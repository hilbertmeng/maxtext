"""Standalone, reproducible forward/backward correctness and timing probe."""
import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas import write_reference, write_residual, c8_read, c8_reference
from layers.rmt_pallas_stage import stage_reference, write_mlp_stage
from layers.rmt_pallas_minor_read import c8_read as minor_read, reference as minor_read_reference
from layers.rmt_pallas_minor import write_residual as minor_write
from layers.rmt_pallas_qk import qk_reference, qk_read
from layers.rmt_pallas_joined import joined_reference, joined_read


def main():
  # TPU's default FP32 dot may truncate operands; compare the VPU path to a
  # full-precision MXU reference. BF16 operands keep their original precision.
  jax.config.update('jax_default_matmul_precision','highest')
  parser=argparse.ArgumentParser()
  parser.add_argument('--arm',choices=['reference','pallas','both'],default='both')
  parser.add_argument('--kernel',choices=['write','c8','stage','joined','qk','minor_write','minor_read'],default='write')
  parser.add_argument('--tokens',type=int,default=8192)
  parser.add_argument('--interpret',action='store_true')
  parser.add_argument('--output',required=True)
  args=parser.parse_args()
  result={'jax':jax.__version__,'devices':str(jax.devices()),'tokens':args.tokens,'arm':args.arm,'write_impl':os.environ.get('RMT_PALLAS_WRITE_IMPL','mxu'),'tile':os.environ.get('RMT_PALLAS_TILE','8'),'qk_norm':os.environ.get('RMT_PALLAS_QK_NORM','gram'),'grouped_dot':os.environ.get('RMT_PALLAS_GROUPED_DOT','0'),'batched_dot':os.environ.get('RMT_PALLAS_BATCHED_DOT','0'),'kernel':args.kernel,'checks':[],'timings':{}}
  def inputs(n,dtype):
    shapes=[(n,48,75),(n,16,48),(n,16,75),(n,16),(16,48)]
    if args.kernel=='c8':shapes=[(n,75,32),(n,16,8),(32,8),(n,16,2)]
    if args.kernel=='stage':shapes.extend([(48,16),(1200,)])
    if args.kernel=='qk':shapes=[(n,32,75),(n,4,32),(n,32,4),(n,32)]
    if args.kernel=='joined':shapes=[(n,48,75),(n,16,8),(48,56),(n,16,2)]
    if args.kernel=='minor_read':shapes=[(n,8,75),(n,16,8),(n,16,2)]
    values=[jax.random.normal(jax.random.key(410+i),s,dtype=dtype) for i,s in enumerate(shapes)]
    gate_index=2 if args.kernel=='minor_read' else 3
    values[gate_index]=jax.nn.sigmoid(values[gate_index]*3-2)
    return values
  reference=lambda m,a,d,g,s:jax.vmap(write_reference,in_axes=(0,0,0,0,None))(m,a,d,g[...,None],s)
  fused=lambda *x:write_residual(*x,interpret=args.interpret)
  labels=('output','d_matrix','d_address','d_data','d_gate','d_static')
  if args.kernel=='minor_write':fused=lambda *x:minor_write(*x,interpret=args.interpret,tile=int(os.environ.get('RMT_PALLAS_TILE','128')),key_contiguous=os.environ.get('RMT_PALLAS_KEY_CONTIGUOUS')=='1')
  if args.kernel=='minor_read':
    reference=jax.vmap(minor_read_reference)
    fused=lambda *x:minor_read(*x,interpret=args.interpret,tile=int(os.environ.get('RMT_PALLAS_TILE','128')))
    labels=('output','d_matrix','d_key','d_gate')
  if args.kernel=='c8':
    reference=lambda m,k,c,g:jax.vmap(c8_reference,in_axes=(0,0,None,0))(m,k,c,g)
    fused=lambda *x:c8_read(*x,interpret=args.interpret)
    labels=('output','d_matrix','d_key','d_compression','d_gates')
  if args.kernel=='stage':
    reference=lambda m,a,d,g,s,r,n:jax.vmap(stage_reference,in_axes=(0,0,0,0,None,None,None))(m,a,d,g[...,None],s,r,n)
    fused=lambda *x:write_mlp_stage(*x,interpret=args.interpret)
    labels=('matrix','read','proxy','d_matrix','d_address','d_data','d_gate','d_static','d_read_key','d_gain')
  if args.kernel=='joined':
    reference=lambda m,k,p,g:jax.vmap(joined_reference,in_axes=(0,0,None,0))(m,k,p,g)
    fused=lambda *x:joined_read(*x,interpret=args.interpret,tile=int(os.environ.get('RMT_PALLAS_TILE','64')))
    labels=('static','dynamic','d_matrix','d_key','d_projection','d_gates')
  if args.kernel=='qk':
    reference=jax.vmap(qk_reference)
    fused=lambda *x:qk_read(*x,interpret=args.interpret,tile=int(os.environ.get('RMT_PALLAS_TILE','32')))
    labels=('output','d_matrix','d_basis','d_mix','d_gates')
  def forward_backward(fn):
    def f(*x):
      y,pb=jax.vjp(fn,*x)
      dy=jax.tree.map(lambda z:jnp.sin(jnp.arange(z.size,dtype=jnp.float32)).reshape(z.shape).astype(z.dtype),y)
      return y,pb(dy)
    return jax.jit(f)
  if args.arm in ('pallas','both'):
    for dtype in ((jnp.float32,) if args.interpret else (jnp.float32,jnp.bfloat16)):
      jax.config.update('jax_default_matmul_precision','highest' if dtype==jnp.float32 else 'default')
      x=inputs(max(16,int(os.environ.get('RMT_PALLAS_TILE','8'))),dtype)
      actual=forward_backward(fused)(*x)
      expected=forward_backward(reference)(*x)
      for label,a,b in zip(labels,jax.tree.leaves(actual),jax.tree.leaves(expected)):
        a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        err=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
        result['checks'].append({'dtype':str(dtype),'part':label,'relative_l2':err,'max_abs':float(np.max(np.abs(a-b)))})
        assert np.isfinite(a).all() and err < (4e-5 if dtype==jnp.float32 else .035),result['checks'][-1]
    print(json.dumps({'checks':result['checks']}),flush=True)
  if not args.interpret:
    jax.config.update('jax_default_matmul_precision','default')
    x=inputs(args.tokens,jnp.bfloat16)
    for name,fn in [('reference',reference),('pallas',fused)]:
      if args.arm not in ('both',name):
        continue
      for mode,run in [('forward',jax.jit(fn)),('forward_backward',forward_backward(fn)),('remat_forward_backward',forward_backward(jax.checkpoint(fn,prevent_cse=True)))]:
        start=time.monotonic()
        compiled=run.lower(*x).compile()
        compilation=time.monotonic()-start
        for _ in range(5):jax.block_until_ready(compiled(*x))
        times=[]
        for _ in range(30):
          start=time.monotonic()
          jax.block_until_ready(compiled(*x))
          times.append(time.monotonic()-start)
        result['timings'][name+'_'+mode]={'median_ms':float(np.median(times)*1000),'mean_ms':float(np.mean(times)*1000),'compile_s':compilation}
        print(json.dumps(result['timings']),flush=True)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

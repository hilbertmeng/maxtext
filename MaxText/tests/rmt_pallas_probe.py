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


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--arm',choices=['reference','pallas','both'],default='both')
  parser.add_argument('--kernel',choices=['write','c8'],default='write')
  parser.add_argument('--tokens',type=int,default=8192)
  parser.add_argument('--interpret',action='store_true')
  parser.add_argument('--output',required=True)
  args=parser.parse_args()
  result={'jax':jax.__version__,'devices':str(jax.devices()),'tokens':args.tokens,'arm':args.arm,'tile':os.environ.get('RMT_PALLAS_TILE','8'),'kernel':args.kernel,'checks':[],'timings':{}}
  def inputs(n,dtype):
    shapes=[(n,48,75),(n,16,48),(n,16,75),(n,16),(16,48)]
    if args.kernel=='c8':shapes=[(n,75,32),(n,16,8),(32,8),(n,16,2)]
    values=[jax.random.normal(jax.random.key(410+i),s,dtype=dtype) for i,s in enumerate(shapes)]
    values[3]=jax.nn.sigmoid(values[3]*3-2)
    return values
  reference=lambda m,a,d,g,s:jax.vmap(write_reference,in_axes=(0,0,0,0,None))(m,a,d,g[...,None],s)
  fused=lambda *x:write_residual(*x,interpret=args.interpret)
  labels=('output','d_matrix','d_address','d_data','d_gate','d_static')
  if args.kernel=='c8':
    reference=lambda m,k,c,g:jax.vmap(c8_reference,in_axes=(0,0,None,0))(m,k,c,g)
    fused=lambda *x:c8_read(*x,interpret=args.interpret)
    labels=('output','d_matrix','d_key','d_compression','d_gates')
  def forward_backward(fn):
    def f(*x):
      y,pb=jax.vjp(fn,*x)
      dy=jnp.sin(jnp.arange(y.size,dtype=jnp.float32)).reshape(y.shape).astype(y.dtype)
      return y,pb(dy)
    return jax.jit(f)
  if args.arm in ('pallas','both'):
    for dtype in ((jnp.float32,) if args.interpret else (jnp.float32,jnp.bfloat16)):
      x=inputs(16,dtype)
      actual=forward_backward(fused)(*x)
      expected=forward_backward(reference)(*x)
      for label,a,b in zip(labels,jax.tree.leaves(actual),jax.tree.leaves(expected)):
        a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        err=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
        result['checks'].append({'dtype':str(dtype),'part':label,'relative_l2':err,'max_abs':float(np.max(np.abs(a-b)))})
        assert np.isfinite(a).all() and err < (4e-5 if dtype==jnp.float32 else .035),result['checks'][-1]
    print(json.dumps({'checks':result['checks']}),flush=True)
  if not args.interpret:
    x=inputs(args.tokens,jnp.bfloat16)
    for name,fn in [('reference',reference),('pallas',fused)]:
      if args.arm not in ('both',name):
        continue
      for mode,run in [('forward',jax.jit(fn)),('forward_backward',forward_backward(fn))]:
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

"""Full write/read-chain equations, every gradient, and TPU timings."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_write_read import write_mlp_read
from layers.rmt_pallas_minor import write_residual
from layers.rmt_pallas_minor_read import c8_read


def norm(x,epsilon=1e-6):
  f=x.astype(jnp.float32)
  return (f*jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+epsilon)).astype(x.dtype)


def reference(m,a,d,g,s,r,c,scale,wk,wg,bias,*,separate=False,interpret=False,tile=128):
  h=d.shape[-2]
  if separate:
    updated=write_residual(m,a,d,g,s,interpret=interpret,tile=tile)
  else:
    updated=m+jnp.einsum('bthv,hk->btkv',d,s)+jnp.einsum('bthk,bthv->btkv',norm(a)*g[...,None],norm(d))
  static=jnp.einsum('btkv,kh->bthv',updated,r)
  raw=updated[...,:h,:].reshape(m.shape[:2]+(-1,))
  x=norm(raw)*scale
  proj=jnp.einsum('btd,df->btf',x,jnp.concatenate((wk,wg),axis=1))
  key=proj[...,:wk.shape[1]].reshape(m.shape[:2]+(h,c.shape[1]))
  gate=jax.nn.sigmoid(proj[...,wk.shape[1]:]+bias)
  compressed=jnp.einsum('btkv,kr->btrv',updated[...,h:,:],c)
  if separate:
    dynamic=c8_read(compressed,key,gate[...,None],interpret=interpret,tile=tile)[...,0,:]
  else:
    dynamic=(.2*gate)[...,None]*jnp.einsum('btrv,bthr->bthv',compressed,norm(key))
  return updated,static+dynamic,x


def inputs(batch,tokens,dtype):
  shapes=[(batch,tokens,48,75),(batch,tokens,16,48),(batch,tokens,16,75),(batch,tokens,16),
          (16,48),(48,16),(32,8),(1200,),(1200,128),(1200,16),(16,)]
  x=[jax.random.normal(jax.random.key(1100+i),s,dtype=dtype) for i,s in enumerate(shapes)]
  x[3]=jax.nn.sigmoid(x[3]-2)
  for i in (4,5,6,8,9):x[i]*=.03
  x[7]=1+.02*x[7];x[10]=x[10]*.1-3
  return x


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--interpret',action='store_true')
  p.add_argument('--tokens',type=int,default=8192)
  p.add_argument('--batch',type=int,default=1)
  p.add_argument('--tile',type=int,default=128)
  p.add_argument('--buffers',type=int,default=1)
  p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
  p.add_argument('--output',required=True)
  p.add_argument('--modes',default='reference,separate,fused')
  args=p.parse_args();dtype=getattr(jnp,args.dtype)
  x=inputs(args.batch,args.tokens,dtype)
  expected=reference(*x)
  dy=tuple(jax.random.normal(jax.random.key(1400+i),v.shape,dtype=dtype) for i,v in enumerate(expected))
  expected_grad=jax.vjp(reference,*x)[1](dy)
  results={}
  for mode in args.modes.split(','):
    if mode=='fused':fn=lambda *z:write_mlp_read(*z,interpret=args.interpret,tile=args.tile,buffers=args.buffers)
    else:fn=lambda *z,mode=mode:reference(*z,separate=mode=='separate',interpret=args.interpret,tile=args.tile)
    # Extract residuals as explicit runtime inputs: do not include forward
    # recomputation in a measurement labelled backward, or embed constants.
    pullback=jax.vjp(fn,*x)[1]
    closed=jax.make_jaxpr(pullback)(dy)
    nconst=len(closed.consts)
    def reverse(*z):return tuple(jax.core.eval_jaxpr(closed.jaxpr,z[:nconst],*z[nconst:]))
    backward_inputs=(*closed.consts,*jax.tree.leaves(dy))
    start=time.monotonic();f=jax.jit(fn).lower(*x).compile();b=jax.jit(reverse).lower(*backward_inputs).compile()
    actual=jax.block_until_ready(f(*x));grads=jax.block_until_ready(b(*backward_inputs))
    errors=[float(jnp.linalg.norm(a.astype(jnp.float32)-e.astype(jnp.float32))/jnp.maximum(jnp.linalg.norm(e.astype(jnp.float32)),1e-10)) for a,e in zip(actual+grads,expected+expected_grad)]
    results[mode]={'compile_s':time.monotonic()-start,'relative_l2':errors}
    print(mode,results[mode],flush=True)
    assert np.isfinite(errors).all() and max(errors)<(5e-5 if dtype==jnp.float32 else .04),errors
    if not args.interpret:
      for stage,fun,z in [('forward',f,x),('backward',b,backward_inputs)]:
        for _ in range(5):jax.block_until_ready(fun(*z))
        times=[]
        for _ in range(40):
          start=time.monotonic();jax.block_until_ready(fun(*z));times.append((time.monotonic()-start)*1000)
        results[mode][stage+'_ms']=float(np.median(times))
      print(json.dumps(results[mode]),flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

"""Projection-inclusive write equations and every explicit pullback gradient."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_projected_write import projected_write
from tests.rmt_fused_write_read_probe import norm


def reference(m,x,d,s,down,up,ub,wg,gb):
  p=jnp.dot(x,jnp.concatenate((down,wg),axis=1))
  hidden=jax.nn.gelu(p[...,:down.shape[1]])
  a=jnp.dot(hidden,up).reshape(d.shape[:-1]+(s.shape[1],))+ub
  g=jax.nn.sigmoid(p[...,down.shape[1]:]+gb)
  return m+jnp.einsum('bthv,hk->btkv',d,s)+jnp.einsum('bthk,bthv->btkv',norm(a)*g[...,None],norm(d))


def main():
  p=argparse.ArgumentParser();p.add_argument('--interpret',action='store_true')
  p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
  p.add_argument('--tokens',type=int,default=256);p.add_argument('--output',required=True)
  args=p.parse_args();t=args.tokens;dt=getattr(jnp,args.dtype)
  shapes=[(1,t,48,75),(1,t,1200),(1,t,16,75),(16,48),(1200,256),(256,768),(16,48),(1200,16),(16,)]
  x=[jax.random.normal(jax.random.key(1600+i),s,dtype=dt) for i,s in enumerate(shapes)]
  for i in (3,4,5,7):x[i]*=.03
  x[6]*=.1;x[8]=x[8]*.1-2
  dy=jax.random.normal(jax.random.key(1610),shapes[0],dtype=dt)
  baseline=reference(*x);bg=jax.vjp(reference,*x)[1](dy)
  fn=lambda *z:projected_write(*z,interpret=args.interpret)
  actual=jax.jit(fn)(*x);ag=jax.jit(lambda *z:jax.vjp(fn,*z)[1](dy))(*x)
  errors=[float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32))/jnp.linalg.norm(b.astype(jnp.float32)))
          for a,b in zip((actual,*ag),(baseline,*bg))]
  result={'relative_l2':errors};print(json.dumps(result),flush=True)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  assert np.isfinite(errors).all() and max(errors)<(5e-5 if dt==jnp.float32 else .04),errors


if __name__=='__main__':main()

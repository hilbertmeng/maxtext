"""Explicit write-backward performance with a runtime upstream cotangent."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_minor import write_residual
from layers.rmt_pallas_minor_read import c8_read


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--tokens',type=int,default=8192)
  p.add_argument('--kernel',choices=['write','read'],default='write')
  p.add_argument('--tile',type=int,default=128)
  p.add_argument('--modes',default='autodiff,analytic,hybrid')
  p.add_argument('--output',required=True)
  args=p.parse_args()
  shapes=[(args.tokens,48,75),(args.tokens,16,48),(args.tokens,16,75),(args.tokens,16),(16,48),(args.tokens,48,75)]
  fn=write_residual
  if args.kernel=='read':
    shapes=[(args.tokens,8,75),(args.tokens,16,8),(args.tokens,16,2),(args.tokens,16,2,75)]
    fn=c8_read
  x=[jax.random.normal(jax.random.key(501+i),s,dtype=jnp.bfloat16) for i,s in enumerate(shapes)]
  gate_index=3 if args.kernel=='write' else 2
  x[gate_index]=jax.nn.sigmoid(x[gate_index]*3-2)
  results={}
  expected=None
  for mode in args.modes.split(','):
    def backward(*z):
      _,pb=jax.vjp(lambda *p:fn(*p,backward=mode,tile=args.tile),*z[:-1])
      return pb(z[-1])
    start=time.monotonic()
    compiled=jax.jit(backward).lower(*x).compile()
    compile_s=time.monotonic()-start
    actual=jax.block_until_ready(compiled(*x))
    if expected is None:expected=actual
    errors=[float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32))/jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)),1e-10)) for a,b in zip(actual,expected)]
    assert all(np.isfinite(errors)) and max(errors)<.02,errors
    for _ in range(5):jax.block_until_ready(compiled(*x))
    times=[]
    for _ in range(60):
      start=time.monotonic()
      jax.block_until_ready(compiled(*x))
      times.append(1000*(time.monotonic()-start))
    results[mode]=dict(median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),compile_s=compile_s,relative_l2_vs_first=errors)
    print(json.dumps(results),flush=True)
  Path(args.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

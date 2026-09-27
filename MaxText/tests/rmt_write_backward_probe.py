"""Explicit write-backward performance with a runtime upstream cotangent."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_minor import write_residual


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--tokens',type=int,default=8192)
  p.add_argument('--modes',default='autodiff,analytic,hybrid')
  p.add_argument('--output',required=True)
  args=p.parse_args()
  shapes=[(args.tokens,48,75),(args.tokens,16,48),(args.tokens,16,75),(args.tokens,16),(16,48),(args.tokens,48,75)]
  x=[jax.random.normal(jax.random.key(501+i),s,dtype=jnp.bfloat16) for i,s in enumerate(shapes)]
  x[3]=jax.nn.sigmoid(x[3]*3-2)
  results={}
  expected=None
  for mode in args.modes.split(','):
    def backward(m,a,d,g,s,dy):
      _,pb=jax.vjp(lambda *z:write_residual(*z,backward=mode),m,a,d,g,s)
      return pb(dy)
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

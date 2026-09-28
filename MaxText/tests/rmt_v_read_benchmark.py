"""Interleaved same-process V/O, V-only, and gated-key V-only TPU timings."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_minor_read import c8_read
from layers.rmt_pallas_v_read import v_read


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--output',required=True)
  args=parser.parse_args()
  shapes=((8192,8,75),(8192,16,8),(8192,16,2),(8192,16,2,75))
  m,k,g,dy=[jax.random.normal(jax.random.key(1201+i),s,dtype=jnp.bfloat16) for i,s in enumerate(shapes)]
  g=jax.nn.sigmoid(g*3-2)
  cases=[]
  for tile in (128,256):
    for label,dest,impl in [('vo',2,c8_read),('v',1,c8_read),('v_fold',1,v_read)]:
      fn=lambda m,k,g,impl=impl,tile=tile:impl(m,k,g,tile=tile)
      def reverse(m,k,g,dy,fn=fn):
        return jax.vjp(fn,m,k,g)[1](dy)
      for stage,func,x in [('forward',fn,(m,k,g[...,:dest])),('backward',reverse,(m,k,g[...,:dest],dy[...,:dest,:]))]:
        compiled=jax.jit(func).lower(*x).compile()
        for _ in range(5):jax.block_until_ready(compiled(*x))
        cases.append((f'{label}_t{tile}_{stage}',compiled,x))
  times={name:[] for name,_,_ in cases}
  rng=np.random.default_rng(87)
  for _ in range(60):
    for i in rng.permutation(len(cases)):
      name,fn,x=cases[i]
      start=time.monotonic();jax.block_until_ready(fn(*x))
      times[name].append((time.monotonic()-start)*1000)
  result={name:dict(median_ms=float(np.median(t)),p10_ms=float(np.quantile(t,.1)),p90_ms=float(np.quantile(t,.9)),samples_ms=t) for name,t in times.items()}
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({name:{k:v for k,v in r.items() if k!='samples_ms'} for name,r in result.items()}),flush=True)


if __name__=='__main__':main()

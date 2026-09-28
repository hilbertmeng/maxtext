"""Time each complete RMT stage's forward and analytic pullback independently.

Standalone timings select tiles; full-training XPlane remains the final arbiter.
The pullback is constructed before timing, so saved forward state is excluded;
recomputation implemented inside the analytic pullback remains included.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from layers.rmt_pallas_attention_read import attention_read
from layers.rmt_pallas_full_write_read import full_write_read
from layers.rmt_pallas_projected_write import projected_write


def inputs(stage, batch, tokens):
  b, t = batch, tokens
  if stage == 'attention':
    shapes = [(b,t,48,75),(48,48),(32,8),(1200,),(1200,1008),(4,32),(48,)]
  else:
    shapes = [(b,t,48,75),(b,t,1200),(b,t,16,75),(16,48),
              (1200,256),(256,768),(16,48),(1200,16),(16,)]
    if stage == 'middle':
      shapes += [(48,16),(32,8),(1200,),(1200,128),(1200,16),(16,)]
  return tuple(jax.random.normal(jax.random.key(2400+i),s,dtype=jnp.bfloat16)*.03
               for i,s in enumerate(shapes))


def timed(fn, args, repeats):
  start = time.monotonic()
  executable = jax.jit(fn).lower(*args).compile()
  compile_s = time.monotonic()-start
  for _ in range(4):
    jax.block_until_ready(executable(*args))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    jax.block_until_ready(executable(*args))
    samples.append(1000*(time.perf_counter()-start))
  return dict(median_ms=statistics.median(samples),samples_ms=samples,
              compile_s=compile_s,hbm_memory=str(executable.memory_analysis()))


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--stages',default='attention,middle,write')
  p.add_argument('--forward-tiles',default='128,256')
  p.add_argument('--reverse-tiles',default='32,64,128,256')
  p.add_argument('--batch',type=int,default=4)
  p.add_argument('--tokens',type=int,default=4096)
  p.add_argument('--repeats',type=int,default=20)
  p.add_argument('--reverse-mode',default='baseline')
  p.add_argument('--write-mode',default='original')
  p.add_argument('--attention-save-small',action='store_true')
  p.add_argument('--output',required=True,type=Path)
  a=p.parse_args()
  result=dict(device=str(jax.devices()[0]),batch=a.batch,tokens=a.tokens,
              libtpu_init_args=os.environ.get('LIBTPU_INIT_ARGS',''),measurements=[])
  for stage in a.stages.split(','):
    x=inputs(stage,a.batch,a.tokens)
    positions=jnp.broadcast_to(jnp.arange(a.tokens,dtype=jnp.int32),(a.batch,a.tokens))
    for phase,tiles in [('forward',a.forward_tiles),('backward',a.reverse_tiles)]:
      for tile in map(int,tiles.split(',')):
        ft=tile if phase=='forward' else 128
        rt=tile if phase=='backward' else 128
        def fn(*z):
          kw=dict(forward_tile=ft,reverse_tile=rt)
          if stage=='attention':return attention_read(*z,positions,save_small=a.attention_save_small,**kw)
          if stage=='middle':return full_write_read(*z,reverse_mode=a.reverse_mode,write_mode=a.write_mode,**kw)
          return projected_write(*z,write_mode=a.write_mode,**kw)
        row=dict(stage=stage,phase=phase,tile=tile,reverse_mode=a.reverse_mode,write_mode=a.write_mode)
        try:
          if phase=='forward':
            row.update(timed(fn,x,a.repeats))
          else:
            # Residuals are explicit arguments, not constants captured in the executable.
            out,pb=jax.vjp(fn,*x)
            dy=jax.tree.map(jnp.ones_like,out)
            leaves,tree=jax.tree.flatten(pb)
            def reverse(*z):return jax.tree.unflatten(tree,z[:-1])(z[-1])
            row.update(timed(reverse,(*leaves,dy),a.repeats))
          row['ok']=True
        except Exception as e:
          row.update(ok=False,error=str(e))
        result['measurements'].append(row)
        print(json.dumps(row),flush=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        jax.clear_caches()


if __name__=='__main__':main()

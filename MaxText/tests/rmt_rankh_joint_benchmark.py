"""Isolate write contractions from projection/read work; use full steps for selection."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from layers.rmt_pallas_write_reverse import backward
from tests.rmt_three_stage_benchmark import timed


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--modes',default='original,row1_major,row1_vloop')
  p.add_argument('--tiles',default='64,128,256')
  p.add_argument('--batch',type=int,default=4)
  p.add_argument('--tokens',type=int,default=4096)
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args();b,t=a.batch,a.tokens
  shapes=[(b,t,16,48),(b,t,16,75),(b,t,16),(16,48),(b,t,48,75)]
  x=tuple(jax.random.normal(jax.random.key(4300+i),shape,dtype=jnp.bfloat16)*.03 for i,shape in enumerate(shapes))
  out=[]
  for mode in a.modes.split(','):
    for tile in map(int,a.tiles.split(',')):
      row=dict(mode=mode,tile=tile,batch=b,tokens=t)
      try:
        row.update(timed(lambda *z:backward(*z,1e-6,tile=tile,write_mode=mode),x,20),ok=True)
      except Exception as e:
        row.update(ok=False,error=str(e))
      out.append(row);a.output.write_text(json.dumps(out,indent=2));print(json.dumps(row),flush=True)
      jax.clear_caches()


if __name__=='__main__':main()

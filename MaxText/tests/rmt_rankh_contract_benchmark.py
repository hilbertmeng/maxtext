"""Four-way pure-contraction probe; reports error and independent compile/timing."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas_rankh_contract import products
from layers.rmt_pallas_minor import _spec
from tests.rmt_three_stage_benchmark import timed


def call(c,d,g,backend,tile=128,head_block=4,interpret=False):
  b,h,k,t=c.shape;v=d.shape[2]
  def kernel(cr,dr,gr,dcr,ddr):
    dcr[...],ddr[...]=products(cr[...],dr[...],gr[...],backend,head_block)
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[_spec(z.shape[1:-1],tile) for z in (c,d,g)],
      out_specs=[_spec((h,k),tile),_spec((h,v),tile)],
      out_shape=(jax.ShapeDtypeStruct(c.shape,jnp.float32),jax.ShapeDtypeStruct(d.shape,jnp.float32)),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rankh_contract_'+backend)(c,d,g)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--backends',default='symmetric,paired,blocked,hybrid')
  p.add_argument('--batch',type=int,default=4);p.add_argument('--tokens',type=int,default=4096)
  p.add_argument('--tile',type=int,default=128);p.add_argument('--head-block',type=int,default=4)
  p.add_argument('--interpret',action='store_true');p.add_argument('--dtype',default='bfloat16')
  p.add_argument('--dump-root',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();dt=getattr(jnp,a.dtype)
  if a.dump_root:
    os.environ['LIBTPU_INIT_ARGS']=os.environ.get('LIBTPU_INIT_ARGS','')+f' --xla_jf_dump_to={a.dump_root}/jf --xla_mosaic_dump_to={a.dump_root}/mosaic --xla_jf_dump_only_matching_hlo=rankh_contract.*'
  shapes=[(a.batch,16,48,a.tokens),(a.batch,16,75,a.tokens),(a.batch,48,75,a.tokens)]
  x=tuple(jax.random.normal(jax.random.key(5310+i),s,dtype=dt)*.03 for i,s in enumerate(shapes))
  c,d,g=x
  expected=(jnp.einsum('bhvt,bkvt->bhkt',d,g,preferred_element_type=jnp.float32),
            jnp.einsum('bhkt,bkvt->bhvt',c,g,preferred_element_type=jnp.float32))
  result=dict(lib_flags=os.environ.get('LIBTPU_INIT_ARGS'),measurements=[])
  for backend in a.backends.split(','):
    row=dict(backend=backend,tile=a.tile,head_block=a.head_block)
    try:
      fn=lambda *z:call(*z,backend,a.tile,a.head_block,a.interpret)
      start=time.monotonic()
      executable=jax.jit(fn).lower(*x).compile()
      row['compile_s']=time.monotonic()-start
      actual=executable(*x)
      errors=[float(jnp.linalg.norm(u-v)/jnp.linalg.norm(v)) for u,v in zip(actual,expected)]
      assert max(errors)<5e-5,errors
      row.update(relative_l2=errors,ok=True)
      if not a.interpret:
        for _ in range(4):jax.block_until_ready(executable(*x))
        samples=[]
        for _ in range(20):
          start=time.perf_counter();jax.block_until_ready(executable(*x))
          samples.append(1000*(time.perf_counter()-start))
        row.update(median_ms=statistics.median(samples),samples_ms=samples,
                   hbm_memory=str(executable.memory_analysis()))
    except Exception as e:row.update(ok=False,error=str(e))
    result['measurements'].append(row);print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps(result,indent=2));jax.clear_caches()
  assert all(x['ok'] for x in result['measurements']),result


if __name__=='__main__':main()

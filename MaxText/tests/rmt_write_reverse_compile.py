"""Compile standalone reverse kernels for the target TPU, without a TPU lease."""
import argparse
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.experimental.topologies import get_topology_desc
from jax.sharding import SingleDeviceSharding
import accelerator_to_spec_map
from layers.rmt_pallas_minor import write_residual
from layers.rmt_pallas_minor_read import c8_read
from layers.rmt_pallas_v_read import v_read


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--topology',default='v5p-16')
  p.add_argument('--batch',type=int,default=16)
  p.add_argument('--modes',default='autodiff,joint_major')
  p.add_argument('--kernel',choices=['write','read'],default='write')
  p.add_argument('--tile',type=int,default=128)
  p.add_argument('--output',required=True)
  args=p.parse_args()
  hw=accelerator_to_spec_map.get_system_characteristics(args.topology)
  topo=get_topology_desc(platform=hw.platform,topology_name=hw.topology_name,
      chip_config_name=hw.chip_config_name,chips_per_host_bounds=hw.chips_per_host_bounds,
      num_slices=1,wrap=hw.wrap)
  sharding=SingleDeviceSharding(topo.devices[0])
  b=args.batch
  shapes=[(b,4096,48,75),(b,4096,16,48),(b,4096,16,75),(b,4096,16),(16,48),(b,4096,48,75)]
  if args.kernel=='read':
    shapes=[(b,4096,8,75),(b,4096,16,8),(b,4096,16,1),(b,4096,16,1,75)]
  x=[jax.ShapeDtypeStruct(s,jnp.bfloat16) for s in shapes]
  results={}
  for mode in args.modes.split(','):
    def reverse(*z):
      if args.kernel=='read':
        fn=(lambda *v:v_read(*v,tile=args.tile)) if mode=='fold_gate' else (lambda *v:c8_read(*v,tile=args.tile,backward=mode))
      else:
        fn=lambda *v:write_residual(*v,backward=mode,tile=args.tile)
      _,pb=jax.vjp(fn,*z[:-1])
      return pb(z[-1])
    start=time.monotonic()
    try:
      c=jax.jit(reverse,in_shardings=sharding,out_shardings=sharding).lower(*x).compile()
      results[mode]={'ok':True,'compile_s':time.monotonic()-start,'memory':str(c.memory_analysis())}
    except Exception as e:
      results[mode]={'ok':False,'compile_s':time.monotonic()-start,'error':str(e)}
    print(json.dumps({mode:results[mode]}),flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

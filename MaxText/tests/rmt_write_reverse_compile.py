"""Compile standalone reverse kernels for the target TPU, without a TPU lease."""
import argparse
import json
import os
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
from layers.rmt_pallas_write_read import write_mlp_read
from layers.rmt_pallas_projected_write import projected_write
from layers.rmt_pallas_full_write_read import full_write_read
from layers.rmt_pallas_attention_read import attention_read


def main():
  p=argparse.ArgumentParser()
  p.add_argument('--topology',default='v5p-16')
  p.add_argument('--batch',type=int,default=16)
  p.add_argument('--modes',default='autodiff,joint_major')
  p.add_argument('--kernel',choices=['write','read','chain','projected','full','attention'],default='write')
  p.add_argument('--tile',type=int,default=128)
  p.add_argument('--backward-tile',type=int,default=0)
  p.add_argument('--compute-tile',type=int,default=0)
  p.add_argument('--reverse-mode',default='baseline')
  p.add_argument('--output',required=True)
  p.add_argument('--save-hlo',action='store_true',help='Save compiled HLO for scoped-memory accounting')
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
  if args.kernel=='chain':
    shapes=[(b,4096,48,75),(b,4096,16,48),(b,4096,16,75),(b,4096,16),
            (16,48),(48,16),(32,8),(1200,),(1200,128),(1200,16),(16,),
            (b,4096,48,75),(b,4096,16,75),(b,4096,1200)]
  if args.kernel=='projected':
    shapes=[(b,4096,48,75),(b,4096,1200),(b,4096,16,75),(16,48),
            (1200,256),(256,768),(16,48),(1200,16),(16,),(b,4096,48,75)]
  if args.kernel=='full':
    shapes=[(b,4096,48,75),(b,4096,1200),(b,4096,16,75),(16,48),
            (1200,256),(256,768),(16,48),(1200,16),(16,),(48,16),(32,8),
            (1200,),(1200,128),(1200,16),(16,),
            (b,4096,48,75),(b,4096,16,75),(b,4096,1200)]
  if args.kernel=='attention':
    shapes=[(b,4096,48,75),(48,48),(32,8),(1200,),(1200,1008),(4,32),(48,),
            (b,4096),(b,4096,48,75),(b,4096,1200)]
  x=[jax.ShapeDtypeStruct(s,jnp.bfloat16) for s in shapes]
  if args.kernel=='attention':x[7]=jax.ShapeDtypeStruct(shapes[7],jnp.int32)
  results={}
  for mode in args.modes.split(','):
    def reverse(*z):
      if args.kernel=='attention':
        fn=lambda *v:attention_read(*v,z[7],forward_tile=args.tile,reverse_tile=args.backward_tile or 128)
        if mode=='forward':return fn(*z[:7])
        return jax.vjp(fn,*z[:7])[1](tuple(z[8:]))
      if args.kernel=='full':
        fn=lambda *v:full_write_read(*v,forward_tile=args.tile,reverse_tile=args.backward_tile or 32,reverse_mode=args.reverse_mode)
        if mode=='forward':return fn(*z[:15])
        return jax.vjp(fn,*z[:15])[1](tuple(z[15:]))
      if args.kernel=='projected':
        fn=lambda *v:projected_write(*v,forward_tile=args.tile,reverse_tile=args.backward_tile or 32)
        if mode=='forward':return fn(*z[:-1])
        return jax.vjp(fn,*z[:-1])[1](z[-1])
      if args.kernel=='chain':
        fn=lambda *v:write_mlp_read(*v,tile=args.tile,backward_tile=args.backward_tile,backward_compute_tile=args.compute_tile)
        if mode=='forward':return fn(*z[:11])
        return jax.vjp(fn,*z[:11])[1](tuple(z[11:]))
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
      if args.save_hlo:
        hlo=Path(args.output).with_suffix('.'+mode+'.hlo.txt')
        hlo.write_text(c.as_text())
        results[mode]['compiled_hlo']=str(hlo)
    except Exception as e:
      results[mode]={'ok':False,'compile_s':time.monotonic()-start,'error':str(e)}
    results[mode]['libtpu_init_args']=os.environ.get('LIBTPU_INIT_ARGS','')
    results[mode]['topology']=args.topology
    results[mode]['batch']=args.batch
    results[mode]['forward_tile']=args.tile
    results[mode]['backward_tile']=args.backward_tile
    results[mode]['compute_tile']=args.compute_tile
    print(json.dumps({mode:results[mode]}),flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

"""Compile the BAM core kernels for a target topology (no lease of that TPU) for dump inspection.

usage (on any installed TPU VM):
  LIBTPU_INIT_ARGS='--xla_mosaic_dump_to=DIR ...' python MaxText/tests/bam_pallas_compile.py \
      --topology v5p-16 --kernel read_bwd --body blocked2 --tile 128
"""
import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental.topologies import get_topology_desc
from jax.sharding import SingleDeviceSharding

sys.path.insert(0, 'MaxText')
import accelerator_to_spec_map  # pylint: disable=wrong-import-position
from layers import bam_pallas as bp  # pylint: disable=wrong-import-position

N, K, V, C, R, QKC = 20, 96, 40, 10, 24, 72


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--topology', default='v5p-16')
  p.add_argument('--kernel', choices=['read_fwd', 'read_bwd', 'write_fwd', 'write_bwd', 'write2_fwd', 'write2_bwd'],
                 required=True)
  p.add_argument('--body', default='blocked2')
  p.add_argument('--tile', type=int, default=128)
  p.add_argument('--tokens', type=int, default=128)
  p.add_argument('--block', type=int, default=4)
  p.add_argument('--vmem', type=int, default=48)
  a = p.parse_args()
  hw = accelerator_to_spec_map.get_system_characteristics(a.topology)
  topo = get_topology_desc(platform=hw.platform, topology_name=hw.topology_name,
                           chip_config_name=hw.chip_config_name,
                           chips_per_host_bounds=hw.chips_per_host_bounds, num_slices=1, wrap=hw.wrap)
  sharding = SingleDeviceSharding(topo.devices[0])
  t = a.tokens
  bf = jnp.bfloat16
  s = lambda *shape: jax.ShapeDtypeStruct(shape, bf)
  if a.kernel.startswith('read'):
    opts = bp._freeze(dict(heads=N, qk_cols=QKC, read_epsilon=1e-4, key_scale=.2, forward_tile=a.tile,
                           reverse_tile=a.tile, vmem_mib=a.vmem, interpret=False, body=a.body,
                           head_block=a.block))
    args = [s(1, V, K, t), jax.ShapeDtypeStruct((4 * N + C, V), jnp.float32), s(1, N, C, t), s(1, N, t),
            s(1, N, C, t), s(1, N, t), s(1, N, C, t), s(1, N, t), s(1, N, t), s(1, N, R, t), s(1, N, R, t)]
    if a.kernel == 'read_fwd':
      fn = lambda *z: bp._read_forward_call(z, dict(opts))
    else:
      args += [s(1, N, K, t)] * 4
      fn = lambda *z: bp._read_backward_call(z[:11], z[11:], dict(opts))
  else:
    opts = bp._freeze(dict(epsilon=1e-6, forward_tile=a.tile, reverse_tile=a.tile, vmem_mib=a.vmem,
                           interpret=False, body=a.body, row_block=a.block, head_block=a.block))
    groups = [s(1, N, K, t), s(1, N, t), s(1, N, V, t)] * (2 if a.kernel.startswith('write2') else 1)
    args = [s(1, V, K, t)] + groups
    if a.kernel.endswith('fwd'):
      fn = lambda m, *g: bp._write_forward_call(m, g, dict(opts))
    else:
      fn = lambda g0, *g: bp._write_backward_call(g0, g, dict(opts))
  start = time.monotonic()
  compiled = jax.jit(fn, in_shardings=sharding, out_shardings=sharding).lower(*args).compile()
  print('COMPILE_JSON ' + json.dumps({'kernel': a.kernel, 'body': a.body, 'tile': a.tile, 'block': a.block,
                                     'topology': a.topology, 'seconds': time.monotonic() - start,
                                     'memory': str(compiled.memory_analysis())}))


if __name__ == '__main__':
  main()

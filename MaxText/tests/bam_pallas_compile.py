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
  p.add_argument('--ablate', default='')
  a = p.parse_args()
  hw = accelerator_to_spec_map.get_system_characteristics(a.topology)
  topo = get_topology_desc(platform=hw.platform, topology_name=hw.topology_name,
                           chip_config_name=hw.chip_config_name,
                           chips_per_host_bounds=hw.chips_per_host_bounds, num_slices=1, wrap=hw.wrap)
  sharding = SingleDeviceSharding(topo.devices[0])
  t = a.tokens
  bf = jnp.bfloat16
  s = lambda *shape: jax.ShapeDtypeStruct(shape, bf)
  rkw = dict(qk_cols=QKC, read_epsilon=1e-4, key_scale=.2, forward_tile=a.tile, reverse_tile=a.tile,
             vmem_mib=a.vmem, body=a.body, head_block=a.block)
  wkw = dict(epsilon=1e-6, forward_tile=a.tile, reverse_tile=a.tile, vmem_mib=a.vmem, body=a.body,
             row_block=a.block, head_block=a.block)
  if a.body == 'v4' and a.kernel == 'read_fwd':
    from layers import bam_pallas_v4 as v4
    np_ = -(-N // 8) * 8
    args = [s(1, K, V, t), jax.ShapeDtypeStruct((4 * np_ + C, V), jnp.float32), s(1, C, N, t), s(1, N, t),
            s(1, C, N, t), s(1, N, t), s(1, C, N, t), s(1, N, t), s(1, N, t), s(1, R, N, t), s(1, R, N, t)]
    opts = dict(heads=N, qk_cols=QKC, read_epsilon=1e-4, key_scale=.2, forward_tile=a.tile, reverse_tile=a.tile,
                vmem_mib=a.vmem, interpret=False, k_block=a.block, ablate=a.ablate)
    fn = lambda *z: v4._read_forward_call(z, opts)
  elif a.body == 'v4' and a.kernel == 'read_bwd':
    from layers import bam_pallas_v4 as v4
    args = [s(1, K, V, t), jax.ShapeDtypeStruct((4 * N + C, V), jnp.float32), s(1, t, N, C), s(1, t, N),
            s(1, t, N, C), s(1, t, N), s(1, t, N, C), s(1, t, N), s(1, t, N), s(1, t, N, R), s(1, t, N, R)]
    read = lambda *z: v4.read(*z, qk_cols=QKC, read_epsilon=1e-4, key_scale=.2, forward_tile=a.tile,
                              reverse_tile=a.tile, vmem_mib=a.vmem, rev_k_block=a.block)
    args += [s(1, t, N, K)] * 4
    fn = lambda *z: jax.vjp(read, *z[:11])[1](tuple(z[11:]))
  elif a.kernel.startswith('read'):
    args = [s(1, V, K, t), jax.ShapeDtypeStruct((4 * N + C, V), jnp.float32), s(1, t, N, C), s(1, t, N),
            s(1, t, N, C), s(1, t, N), s(1, t, N, C), s(1, t, N), s(1, t, N), s(1, t, N, R), s(1, t, N, R)]
    read = lambda *z: bp.read(*z, **rkw)
    if a.kernel == 'read_fwd':
      fn = read
    else:
      args += [s(1, t, N, K)] * 4
      fn = lambda *z: jax.vjp(read, *z[:11])[1](tuple(z[11:]))
  else:
    n_groups = 2 if a.kernel.startswith('write2') else 1
    args = [s(1, V, K, t)] + [s(1, t, N, K), s(1, t, N), s(1, t, N, V)] * n_groups
    write = lambda m, *g: bp.write(m, [g[i:i + 3] for i in range(0, len(g), 3)], **wkw)
    if a.kernel.endswith('fwd'):
      fn = write
    else:
      args += [s(1, V, K, t)]
      fn = lambda *z: jax.vjp(write, *z[:-1])[1](z[-1])
  start = time.monotonic()
  compiled = jax.jit(fn, in_shardings=sharding, out_shardings=sharding).lower(*args).compile()
  print('COMPILE_JSON ' + json.dumps({'kernel': a.kernel, 'body': a.body, 'tile': a.tile, 'block': a.block,
                                     'topology': a.topology, 'seconds': time.monotonic() - start,
                                     'memory': str(compiled.memory_analysis())}))


if __name__ == '__main__':
  main()

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
  p.add_argument('--c', type=int, default=C, help='compressed rows (V for full-M reads)')
  p.add_argument('--gated', action='store_true', help='per-token static-read gates (generalized read)')
  p.add_argument('--hm-reverse', action='store_true', help='head-major dynamic-read reverse (v7)')
  p.add_argument('--kblock', type=int, default=0, help='v7 forward k block (0 = default)')
  p.add_argument('--rolled', default='', help='v7 rolled reverse passes, e.g. b1,b2')
  p.add_argument('--token-io', action='store_true', help='v7 token-major q/k/v/o and cotangents')
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
             vmem_mib=a.vmem, body=a.body, head_block=a.block,
             **({'hm_reverse': True} if a.hm_reverse else {}), **({'k_block': a.kblock} if a.kblock else {}),
             **({'rolled_passes': tuple(a.rolled.split(','))} if a.rolled else {}),
             **({'token_io': True} if a.token_io else {}))
  wkw = dict(epsilon=1e-6, forward_tile=a.tile, reverse_tile=a.tile, vmem_mib=a.vmem, body=a.body,
             row_block=a.block, head_block=a.block)
  if a.kernel.startswith('read'):
    c = a.c
    mshape = (1, K, V, t) if a.body.startswith('v7') else (1, V, K, t)
    args = [s(*mshape), jax.ShapeDtypeStruct((4 * N + c, V), jnp.float32), s(1, t, N, c), s(1, t, N),
            s(1, t, N, c), s(1, t, N), s(1, t, N, c), s(1, t, N), s(1, t, N), s(1, t, N, R), s(1, t, N, R)]
    ng = 4 if a.gated else 0
    args += [s(1, t, N)] * ng
    read = lambda *z: bp.read(*z[:11], static_gates=tuple(z[11:]) or None, **rkw)
    if a.kernel == 'read_fwd':
      fn = read
    else:
      args += [s(1, t, N, K)] * 4
      fn = lambda *z: jax.vjp(read, *z[:11 + ng])[1](tuple(z[11 + ng:]))
  else:
    n_groups = 2 if a.kernel.startswith('write2') else 1
    mshape = (1, K, V, t) if a.body.startswith('v7') else (1, V, K, t)
    args = [s(*mshape)] + [s(1, t, N, K), s(1, t, N), s(1, t, N, V)] * n_groups
    write = lambda m, *g: bp.write(m, [g[i:i + 3] for i in range(0, len(g), 3)], **wkw)
    if a.kernel.endswith('fwd'):
      fn = write
    else:
      args += [s(*mshape)]
      fn = lambda *z: jax.vjp(write, *z[:-1])[1](z[-1])
  start = time.monotonic()
  compiled = jax.jit(fn, in_shardings=sharding, out_shardings=sharding).lower(*args).compile()
  print('COMPILE_JSON ' + json.dumps({'kernel': a.kernel, 'body': a.body, 'tile': a.tile, 'block': a.block,
                                     'topology': a.topology, 'seconds': time.monotonic() - start,
                                     'memory': str(compiled.memory_analysis())}))


if __name__ == '__main__':
  main()

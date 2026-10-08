"""Prototype: token-major write forward with block-diagonal MXU batching (cost probe).

M tm [B,T,V,K]; normalized gated content C [B,T,N,K]; normalized address transposed
AT [B,T,V,N]. For each group of g tokens:
  X = blockdiag_j(AT_j) [(j,v), (j',n)]   (built by lane-tiling AT and masking)
  W = stack_j(C_j)      [(j,n), k]
  M[(j,v), k] += X @ W
usage: python bam_mxu_write_proto.py [--check] [--compile TOPOLOGY] [g] [tile]
"""
import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


def kernel_factory(g):
  def kernel(m_ref, c_ref, at_ref, out_ref):
    t, vdim, kdim = m_ref.shape
    n = c_ref.shape[1]
    rows, cols = g * vdim, g * n
    ri = jax.lax.broadcasted_iota(jnp.int32, (rows, cols), 0) // vdim
    ci = jax.lax.broadcasted_iota(jnp.int32, (rows, cols), 1) // n
    mask = ri == ci

    def body(b, carry):
      t0 = b * g
      at = at_ref[pl.ds(t0, g)].reshape(rows, n)               # [(j,v), n]
      x = jnp.where(mask, jnp.concatenate([at] * g, axis=1), 0).astype(jnp.bfloat16)
      w = c_ref[pl.ds(t0, g)].reshape(cols, kdim).astype(jnp.bfloat16)   # [(j,n), k]
      out = jnp.dot(x, w, preferred_element_type=F32).reshape(g, vdim, kdim)
      out_ref[pl.ds(t0, g)] = (m_ref[pl.ds(t0, g)].astype(F32) + out).astype(out_ref.dtype)
      return carry
    jax.lax.fori_loop(0, t // g, body, 0)
  return kernel


def call(m, c, at, g, tile, interpret=False):
  b, t, v, k = m.shape
  n = c.shape[2]
  spec = lambda *s: pl.BlockSpec((None, tile) + s, lambda bb, i: (bb, i) + (0,) * len(s))
  return pl.pallas_call(
      kernel_factory(g), grid=(b, t // tile),
      in_specs=[spec(v, k), spec(n, k), spec(v, n)], out_specs=spec(v, k),
      out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype), interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel')),
      name='bam_mxu_write')(m, c, at)


def reference(m, c, at):
  return (m.astype(F32) + jnp.einsum('btvn,btnk->btvk', at.astype(F32), c.astype(F32))).astype(m.dtype)


def main():
  argv = sys.argv[1:]
  if '--compile' in argv:
    i = argv.index('--compile')
    argv = argv[:i] + argv[i + 2:]
  args = [a for a in argv if not a.startswith('--')]
  g = int(args[0]) if args else 4
  tile = int(args[1]) if len(args) > 1 else 128
  B, T, V, K, N = 1, 256, 40, 96, 20
  if '--check' in sys.argv:
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    m = jax.random.normal(ks[0], (B, T, V, K)).astype(jnp.bfloat16)
    c = jax.random.normal(ks[1], (B, T, N, K)).astype(jnp.bfloat16)
    at = jax.random.normal(ks[2], (B, T, V, N)).astype(jnp.bfloat16)
    got = call(m, c, at, g, tile, interpret=True).astype(F32)
    want = reference(m, c, at).astype(F32)
    print('max rel err', float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want))))
  if '--compile' in sys.argv:
    sys.path.insert(0, 'MaxText')
    import accelerator_to_spec_map
    from jax.experimental.topologies import get_topology_desc
    from jax.sharding import SingleDeviceSharding
    hw = accelerator_to_spec_map.get_system_characteristics(sys.argv[sys.argv.index('--compile') + 1])
    topo = get_topology_desc(platform=hw.platform, topology_name=hw.topology_name,
                             chip_config_name=hw.chip_config_name,
                             chips_per_host_bounds=hw.chips_per_host_bounds, num_slices=1, wrap=hw.wrap)
    sh = SingleDeviceSharding(topo.devices[0])
    s = lambda *shape: jax.ShapeDtypeStruct(shape, jnp.bfloat16)
    fn = jax.jit(lambda m, c, at: call(m, c, at, g, tile), in_shardings=sh, out_shardings=sh)
    fn.lower(s(1, 128, V, K), s(1, 128, N, K), s(1, 128, V, N)).compile()
    print('COMPILED')


if __name__ == '__main__':
  main()

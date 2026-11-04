"""v6e kernel knobs are implementation-only: tiles, chunks, xhat double buffering and unsegmented Splash.

The formal read feature set (fused full-head QKNorm, M shift, whole-M pre-norm, per-head Q/K/V/O scales,
static gates) with the V6eFast settings matches the v5p defaults in outputs and every gradient; the
write kernel's 256 tiles match 128; ignoring Splash segments is rejected unless segmentation is all ones (Pile without mix_attn, synthetic).
"""
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from bam_config import validate_bam_config  # pylint: disable=wrong-import-position
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position
import bam_pallas_mshift_test as shift_test  # pylint: disable=wrong-import-position

B, T, V, K, N, QKC, RS = shift_test.B, shift_test.T, shift_test.V, shift_test.K, shift_test.N, shift_test.QKC, shift_test.RS
V5P = dict(forward_tile=128, reverse_tile=128, chunk=8)
V6E = dict(forward_tile=256, reverse_tile=128, chunk=64, xhat_double_buffer=True)
V6E_GATE = dict(V6E, fwd_gate_scratch=True)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


class V6eKernelKnobTest(unittest.TestCase):

  def test_read_v6e_settings_match_v5p_defaults(self):
    m, sw, xs, sgs, shift = shift_test.inputs(11)
    ks = iter(jax.random.split(jax.random.PRNGKey(12), 8))
    qdim = QKC + RS
    qk = (jnp.abs(jax.random.normal(next(ks), (B, T, N))) + .5, jnp.abs(jax.random.normal(next(ks), (B, T, N))) + .5,
          1 + .2 * jax.random.normal(next(ks), (qdim,)), 1 + .2 * jax.random.normal(next(ks), (qdim,)), 1e-5)
    scales = jnp.asarray([.8, 1.1, .9, 1.2])
    cts = tuple(jax.random.normal(next(ks), (B, T, N, K if i > 1 else qdim)) for i in range(4))

    def call(m, sw, xs, scales, knobs):
      head = lambda i: scales[i] * jnp.ones(N)
      return v8.read_tm(m, sw, *xs, qk_cols=QKC, read_epsilon=shift_test.EPS, key_scale=shift_test.SCALE,
                        interpret=True, q_scale=1.0, static_gates=sgs, shift=shift, m_norm=1e-5, qk_norm=qk,
                        q_dynamic_scale=head(0), k_dynamic_scale=head(1), v_dynamic_scale=head(2),
                        o_dynamic_scale=head(3), **knobs)

    outs = [call(m, sw, xs, scales, k) for k in (V5P, V6E)]
    gate = call(m, sw, xs, scales, V6E_GATE)
    for i, (a, b) in enumerate(zip(outs[1], gate)):
      self.assertLess(rel(b, a), 1e-6, f'gate scratch output {i}')
    for i, (a, b) in enumerate(zip(*outs)):
      self.assertLess(rel(b, a), 1e-5, f'output {i}')
    loss = lambda knobs: lambda m, sw, xs, s: sum(jnp.sum(o * c) for o, c in zip(call(m, sw, xs, s, knobs)[:4], cts))
    ga, gb = (jax.grad(loss(k), argnums=(0, 1, 2, 3))(m, sw, xs, scales) for k in (V5P, V6E))
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree.leaves(gb)):
      self.assertLess(rel(b, a), 1e-5, f'grad {jax.tree_util.keystr(path)}')

  def test_write_tile_256_matches_128(self):
    ks = iter(jax.random.split(jax.random.PRNGKey(13), 16))
    r = lambda shape: jax.random.normal(next(ks), shape, jnp.float32)
    m = r((B, T, V, K))
    groups = [(r((B, T, N, K)), r((B, T, N)), r((B, T, N, V))) for _ in range(2)]
    ct = r((B, T, V, K))

    def call(m, groups, tile):
      return v8.write_tm(m, groups, epsilon=1e-6, forward_tile=tile, reverse_tile=tile, interpret=True,
                         content_divisors=(None, 1.))
    self.assertLess(rel(call(m, groups, 256), call(m, groups, 128)), 1e-6)
    ga, gb = (jax.grad(lambda m, g: jnp.sum(call(m, g, tile) * ct), argnums=(0, 1))(m, groups) for tile in (128, 256))
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree.leaves(gb)):
      self.assertLess(rel(b, a), 1e-5, f'grad {jax.tree_util.keystr(path)}')

  def test_ignore_segments_requires_all_ones_pile_segmentation(self):
    base = {'bam_enabled': True, 'bam_splash_ignore_segments': True, 'dataset_type': 'pile'}
    validate_bam_config(dict(base, mix_attn=False))
    validate_bam_config(dict(base, mix_attn=False, dataset_type='synthetic'))
    for bad in (dict(base, mix_attn=True), dict(base, mix_attn=False, dataset_type='hf')):
      with self.assertRaises(ValueError):
        validate_bam_config(bad)


if __name__ == '__main__':
  unittest.main()

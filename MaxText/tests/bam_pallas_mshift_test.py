"""CPU interpret-mode check of the v8t read with the in-kernel previous-token M shift.

Reference: mix M in XLA (M_t + g'_t (M_{t-1} - M_t), g' zero at invalid tokens), then the unshifted
v8t read, whose own equivalence to the original model is covered by bam_pallas_model_test.
"""
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position

B, T, V, K, N, QKC, RS = 2, 256, 16, 32, 2, 24, 8
EPS, SCALE = 1e-4, 0.2


def mixed(m, shift):
  prev = jnp.concatenate((m[:, :1], m[:, :-1]), axis=1)
  return m + shift[..., None] * (prev - m)


def inputs(seed=0):
  ks = iter(jax.random.split(jax.random.PRNGKey(seed), 32))
  r = lambda shape, s=1.0: s * jax.random.normal(next(ks), shape, jnp.float32)
  m = r((B, T, V, K))
  sw = r((4 * N, V), .3)
  xs = (r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)),
        r((B, T, N)), r((B, T, N)), r((B, T, N, RS)), r((B, T, N, RS)))
  static_gates = tuple(jax.nn.sigmoid(r((B, T, N))) for _ in range(4))
  shift = jax.nn.sigmoid(r((B, T, V)))
  valid = jnp.ones((B, T), bool).at[:, 0].set(False).at[0, 77].set(False).at[1, 128].set(False)
  return m, sw, xs, static_gates, shift * valid[..., None]


def read(m, sw, xs, static_gates, shift, tile, body='dot'):
  outs = v8.read_tm(m, sw, *xs, qk_cols=QKC, read_epsilon=EPS, key_scale=SCALE, forward_tile=tile,
                    reverse_tile=tile, interpret=True, q_scale=0.125, static_gates=static_gates, shift=shift,
                    shift_body=body)
  return outs


class ShiftedReadTest(unittest.TestCase):

  def assert_close(self, a, b, tol, name):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)
    self.assertLess(err, tol, f'{name}: relative max error {err:.3e}')

  def test_forward_and_gradients(self):
    m, sw, xs, sgs, shift = inputs()
    cts = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), (B, T, N, K)) for i in range(4))
    ct_m = jax.random.normal(jax.random.PRNGKey(20), m.shape)

    def loss(fn):
      def f(m, sw, xs, sgs, shift):
        outs = fn(m, sw, xs, sgs, shift)
        return sum(jnp.sum(o * c) for o, c in zip(outs[:4], cts)) + jnp.sum(outs[4] * ct_m)
      return f

    for tile, body in ((64, 'dot'), (256, 'dot'), (64, 'blend'), (256, 'blend'), (64, 'blend_tm')):
      kernel = lambda m, sw, xs, sgs, shift, tile=tile, body=body: read(m, sw, xs, sgs, shift, tile, body)
      ref = lambda m, sw, xs, sgs, shift, tile=tile: (
          read(mixed(m, shift), sw, xs, sgs, None, tile)[:4] + (m,))
      for i, (a, b) in enumerate(zip(kernel(m, sw, xs, sgs, shift), ref(m, sw, xs, sgs, shift))):
        self.assert_close(a, b, 1e-5, f'{body} tile {tile} output {i}')
      plain = read(m, sw, xs, sgs, None, tile)[3]
      shifted = kernel(m, sw, xs, sgs, shift)[3]
      self.assertGreater(float(jnp.max(jnp.abs(plain - shifted)) / jnp.max(jnp.abs(plain))), 0.05)
      ga = jax.grad(loss(kernel), argnums=(0, 1, 2, 3, 4))(m, sw, xs, sgs, shift)
      gb = jax.grad(loss(ref), argnums=(0, 1, 2, 3, 4))(m, sw, xs, sgs, shift)
      for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree.leaves(gb)):
        self.assert_close(a, b, 1e-5, f'{body} tile {tile} grad {jax.tree_util.keystr(path)}')


class ShiftedModelTest(unittest.TestCase):
  """Model-level: the MediumProp M-shift + post-attention MLP-norm recipe, pure JAX vs the v8t core.

  Kept out of bam_pallas_model_test: one process interpreting all 14 fused models there segfaults."""

  def test_v8t_m_shift_post_attention_controls(self):
    sys.path.insert(0, 'MaxText/tests')
    import bam_pallas_model_test as model_test  # pylint: disable=import-outside-toplevel
    model_test.ModelEquivalenceTest('_run')._run(
        'BamPallasMShiftTinyXlaTest', ('BamPallasMShiftTinyPallasV8TTest', 'BamPallasMShiftTinyPallasV8TFusedTest'))


if __name__ == '__main__':
  unittest.main()

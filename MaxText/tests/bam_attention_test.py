"""Focused tests for BAM runtime read-key transforms."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from layers.attentions import (
    _mix_bam_write_v,
    _select_bam_write_source,
    _shared_bam_fetch_alpha,
    _transform_bam_read_key,
    _update_bam_matrix,
)


class BamReadKeyTransformTest(absltest.TestCase):

  def test_constant_matrix_update_matches_existing_decay(self):
    M_in = jnp.arange(12, dtype=jnp.float32).reshape(1, 1, 3, 4)
    dM = jnp.ones_like(M_in)
    M_out, forget_gate = _update_bam_matrix(M_in, dM, 0.75)
    np.testing.assert_array_equal(M_out, 0.75 * M_in + dM)
    self.assertIsNone(forget_gate)

  def test_dynamic_forget_starts_at_requested_retention_and_has_gradient(self):
    forget_init = 0.01
    logits = jnp.full((1, 2, 1), np.log(forget_init / (1.0 - forget_init)))
    M_in = jnp.ones((1, 2, 3, 4))
    dM = jnp.zeros_like(M_in)
    M_out, forget_gate = _update_bam_matrix(M_in, dM, 1.0, logits)
    np.testing.assert_allclose(forget_gate, forget_init, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(M_out, 1.0 - forget_init, rtol=1e-6, atol=1e-7)
    grad = jax.grad(
        lambda z: _update_bam_matrix(M_in, dM, 1.0, z)[0].sum())(logits)
    self.assertTrue(np.all(np.asarray(grad) < 0.0))

  def test_write_v_mix_starts_as_local_projection(self):
    x_v = jax.random.normal(jax.random.PRNGKey(3), (2, 5, 4, 3))
    o_head = jax.random.normal(jax.random.PRNGKey(4), (2, 5, 4, 6))
    scale = jnp.tile(jnp.array([[1.0, 0.0]]), (4, 1))
    bias = jnp.zeros((4, 3))
    mixed = _mix_bam_write_v(x_v, o_head, 3, scale, bias)
    np.testing.assert_array_equal(mixed, x_v)

  def test_write_v_mix_selects_output_tail_and_bias(self):
    x_v = jnp.zeros((1, 2, 2, 3))
    o_head = jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 2, 3)
    scale = jnp.tile(jnp.array([[0.0, 1.0]]), (2, 1))
    bias = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    mixed = _mix_bam_write_v(x_v, o_head, 0, scale, bias)
    np.testing.assert_array_equal(mixed, o_head + bias)

  def test_shared_fetch_implementations_match_values_and_gradients(self):
    q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8)) / np.sqrt(8)
    k = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4, 8))
    causal = jnp.where(jnp.tril(jnp.ones((1, 1, 5, 5), dtype=bool)), 0.0, -jnp.inf)
    weights = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 5, 5))

    def objective(query, mode, soft_cap):
      logits = jnp.einsum('btnd,bsnd->bnts', query, k)
      if soft_cap:
        logits = jnp.tanh(logits / soft_cap) * soft_cap
      logits = jnp.where(causal == 0, logits, causal)
      alpha = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
      fetch = _shared_bam_fetch_alpha(
          alpha, query, k, causal, 2, mode, True, soft_cap, True)
      return jnp.sum(fetch * weights)

    for soft_cap in (0.0, 3.0):
      values = [objective(q, mode, soft_cap) for mode in ('legacy', 'compact', 'recompute')]
      grads = [jax.grad(objective)(q, mode, soft_cap)
               for mode in ('legacy', 'compact', 'recompute')]
      for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-6, atol=1e-6)
      for grad in grads[1:]:
        np.testing.assert_allclose(grad, grads[0], rtol=1e-5, atol=1e-6)

  def test_write_source_selection(self):
    terms = [jnp.full((2,), value) for value in (1.0, 2.0, 4.0, 8.0)]
    np.testing.assert_array_equal(
        _select_bam_write_source('std', *terms), jnp.full((2,), 1.0))
    np.testing.assert_array_equal(
        _select_bam_write_source('std+cross', *terms), jnp.full((2,), 7.0))
    np.testing.assert_array_equal(
        _select_bam_write_source('std+cross+local_o', *terms), jnp.full((2,), 15.0))
    y_all = jnp.full((2,), 16.0)
    self.assertIs(_select_bam_write_source('std+cross+local_o', *terms, y_all), y_all)

  def test_soft_rms_cap_is_identity_at_zero_and_bounded(self):
    scale = 2.0
    jacobian = jax.jacfwd(
        lambda z: _transform_bam_read_key(z, 'soft_rms_cap', scale))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-6, atol=1e-6)

    large = jnp.array([30.0, 40.0])
    transformed = _transform_bam_read_key(large, 'soft_rms_cap', scale)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    self.assertLess(float(transformed_rms), scale)
    self.assertGreater(float(transformed_rms), 0.99 * scale)

  def test_rms_gate_bias_calibration_preserves_zero_jacobian(self):
    scale = 2.0
    eps = 1e-4
    initial_gate = np.sqrt(eps) / scale
    gate_logits = jnp.full((1,), np.log(initial_gate / (1.0 - initial_gate)))
    jacobian = jax.jacfwd(
        lambda z: _transform_bam_read_key(
            z, 'rms_gate', scale, eps, gate_logits))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-5, atol=1e-5)

  def test_rms_gate_has_requested_rms(self):
    scale = 2.0
    gate = 0.25
    gate_logits = jnp.full((1,), np.log(gate / (1.0 - gate)))
    transformed = _transform_bam_read_key(
        jnp.array([3.0, 4.0]), 'rms_gate', scale, 1e-8, gate_logits)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    np.testing.assert_allclose(transformed_rms, scale * gate, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()

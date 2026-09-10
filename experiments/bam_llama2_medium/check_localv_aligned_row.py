"""Check shared row-coordinate projection without changing column read or routing."""
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import factorized_head_bam_read, _fit_bam_read_to_head


def main():
  keys = jax.random.split(jax.random.key(77), 5)
  matrix = jax.random.normal(keys[0], (1, 3, 32, 32))
  hidden = jnp.zeros((1, 3, 8))
  read_keys = jax.random.normal(keys[1], (1, 3, 4, 64))
  mix = jax.random.normal(keys[2], (1, 3, 16, 2, 4))
  gates = jax.random.normal(keys[3], (1, 3, 16, 2))
  projection = jax.random.normal(keys[4], (32, 8))
  kwargs = dict(rank=4, rank_routing='head_gate_r', key_mode='rms_gate',
                key_gate_logits=gates, key_scale=1., implementation='mul_reduce_btn',
                return_sides=True, rms_epsilon=1e-4)

  def read(p):
    return factorized_head_bam_read(matrix, hidden, lambda _: read_keys,
                                   lambda _: mix, v_projection=p, **kwargs)

  u, v = read(None)
  actual_u, actual_v = read(projection)
  np.testing.assert_array_equal(actual_u, u)
  np.testing.assert_allclose(actual_v, v @ projection, atol=2e-5, rtol=2e-5)
  fitted = _fit_bam_read_to_head((actual_u, actual_v), 32, 64)
  np.testing.assert_array_equal(fitted[..., :32], u)
  np.testing.assert_array_equal(fitted[..., 32:40], actual_v)
  np.testing.assert_array_equal(fitted[..., 40:], 0)
  actual_grad = jax.grad(lambda p: jnp.mean(read(p)[1] ** 2))(projection)
  expected_grad = jax.grad(lambda p: jnp.mean((v @ p) ** 2))(projection)
  # Projection-before-mix changes floating-point summation order.
  relative_error = jnp.linalg.norm(actual_grad - expected_grad) / jnp.linalg.norm(expected_grad)
  assert float(relative_error) < 2e-6, float(relative_error)
  print('ALIGNED_ROW_OK: unchanged col; shared projection forward/gradient; slots 32:40')


if __name__ == '__main__':
  main()

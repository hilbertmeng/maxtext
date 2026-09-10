"""Verify L-only tied row decoding, untouched column/F output, and gradients."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import BamAttention


def main():
  u, c, e = [jax.random.normal(jax.random.key(i), shape) for i, shape in
             enumerate(((1, 3, 16, 32), (1, 3, 16, 8), (32, 8)))]

  def output(projection, local):
    config = SimpleNamespace(
        _abs_k_dim=None, _abs_v_dim=8, _abs_v_row_output='direct',
        _local_o_row_tied_decoder=local, abs_v_cache_projection=projection,
        num_query_heads=16, head_dim=64)
    return BamAttention._expand_full_read.__wrapped__(config, (u, c))

  expected = jnp.concatenate((u, c @ e.T), axis=-1)
  actual = output(e, True)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  np.testing.assert_array_equal(actual[..., :32], u)
  np.testing.assert_array_equal(output(e, False), jnp.pad(jnp.concatenate((u, c), -1),
                                                       ((0, 0), (0, 0), (0, 0), (0, 24))))
  g1 = jax.grad(lambda p: jnp.mean(output(p, True) ** 2))(e)
  g2 = jax.grad(lambda p: jnp.mean(jnp.concatenate((u, c @ p.T), -1) ** 2))(e)
  np.testing.assert_allclose(g1, g2, rtol=1e-6, atol=1e-6)
  print('LOCAL_O_TIED_DECODE_OK: L row only; column and F unchanged; E receives decoder gradient')


if __name__ == '__main__':
  main()

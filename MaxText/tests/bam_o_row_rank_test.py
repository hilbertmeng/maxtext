"""One-sided C row read matches explicit normalized effective keys and VJPs."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import _effective_key_row_read, BamAttention


class ORowRankTest(absltest.TestCase):
  def test_forward_and_vjp(self):
    keys = jax.random.split(jax.random.key(18), 5)
    args = [jax.random.normal(k, shape) for k, shape in zip(keys, (
        (1, 3, 7, 5), (1, 3, 4, 7), (1, 3, 6, 4), (1, 3, 6)))]
    cotangent = jax.random.normal(keys[4], (1, 3, 6, 5))
    def explicit(m, a, h, g):
      key = h @ a
      key = key * jax.lax.rsqrt(jnp.mean(key**2, -1, keepdims=True) + 1e-4)
      return (2 * jax.nn.sigmoid(g)[..., None] * key) @ m
    def actual(m, a, h, g):
      return _effective_key_row_read(m, a, h, g, key_scale=2.,
          rms_epsilon=1e-4, implementation='mul_reduce_btn')
    np.testing.assert_allclose(actual(*args), explicit(*args), atol=3e-6, rtol=3e-5)
    for fn in (actual, explicit):
      value, backward = jax.vjp(fn, *args)
      grads = backward(cotangent)
      if fn is actual:
        reference = grads
      else:
        for x, y in zip(reference, grads):
          np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-4)

  def test_local_and_fetch_modules(self):
    from bam_local_fetch_test import LocalFetchTest
    import max_utils
    helper = LocalFetchTest()
    self.addCleanup(helper.doCleanups)
    cfg = helper.config('BamMediumIndependentLLFBAlignedRowORowRank4CFp32')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    for mode in ('local_qk+local_o', 'local_qk+full'):
      module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
          head_dim=64, max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode=mode, attention_type=cfg.attention_type)
      x = jnp.ones((1, 8, 128), cfg.dtype)
      M = jnp.ones((1, 8, 32, 32), cfg.dtype)
      args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
      p = module.init({'params':jax.random.key(1), 'aqt':jax.random.key(2)},
          *args, M_in=M, deterministic=True, layer_index=2)
      self.assertIn('W_o_row_basis', p['params'])
      self.assertEqual(p['params']['W_o_row_basis']['kernel'].value.shape[-2:], (4,32))
      def loss(params):
        y, m = module.apply({'params':params}, *args, M_in=M, deterministic=True, layer_index=2)
        return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(m.astype(jnp.float32)**2)
      grad = jax.grad(loss)(p['params'])
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad)))


if __name__ == '__main__':
  absltest.main()

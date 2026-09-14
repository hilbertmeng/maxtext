"""Shared raw row bases: destination-independent routing and no second M read."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import factorized_head_bam_read, _effective_key_row_expand, BamAttention


class SharedRowBasisTest(absltest.TestCase):
  def test_shared_forward_and_gradients(self):
    keys = jax.random.split(jax.random.key(71), 8)
    shapes = ((1, 3, 7, 5), (1, 3, 4, 12), (1, 3, 6, 2, 4),
              (1, 3, 6, 2), (5, 3), (1, 3, 6, 4), (1, 3, 6))
    args = [jax.random.normal(k, s) for k, s in zip(keys, shapes)]

    def read(m, a, h, g, e, return_basis=False):
      return factorized_head_bam_read(
          m, None, lambda _: a, lambda _: h,
          rms_epsilon=1e-4, key_mode='rms_gate', key_scale=1.,
          key_gate_logits=g, rank=4, rank_routing='effective_key',
          scale_placement='mix', v_projection=e, return_row_basis=return_basis)

    def shared(m, a, h, g, e, ho, go):
      v, basis = read(m, a, h, g, e, True)
      o = _effective_key_row_expand(*basis, ho, go, key_scale=2., rms_epsilon=1e-4)
      return v, o

    def explicit(m, a, h, g, e, ho, go):
      v = read(m, a, h, g, e)
      effective_key = ho @ a[..., :7]
      effective_key *= jax.lax.rsqrt(jnp.mean(effective_key**2, -1, keepdims=True) + 1e-4)
      o = (2 * jax.nn.sigmoid(go)[..., None] * effective_key) @ (m @ e)
      return v, o

    for dtype, atol, rtol in ((jnp.float32, 2e-5, 2e-4), (jnp.bfloat16, .3, .08)):
      typed = [v.astype(dtype) for v in args]
      actual, pull = jax.vjp(shared, *typed)
      expected, ref_pull = jax.vjp(explicit, *typed)
      # Returning intermediate bases does not alter the V computation at all.
      for x, y in zip(actual[0], expected[0]):
        np.testing.assert_array_equal(x, y)
      np.testing.assert_allclose(actual[1].astype(float), expected[1].astype(float), atol=atol, rtol=rtol)
      cot = jax.tree.map(jnp.ones_like, actual)
      for x, y in zip(pull(cot), ref_pull(cot)):
        np.testing.assert_allclose(x.astype(float), y.astype(float), atol=atol, rtol=rtol)

  def test_modules_and_unchanged_fetch(self):
    from bam_local_fetch_test import LocalFetchTest
    import max_utils
    helper = LocalFetchTest()
    self.addCleanup(helper.doCleanups)
    cfg = helper.config('BamMediumIndependentLLFBAlignedRowSharedRowRank4CFp32')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x = jax.random.normal(jax.random.key(1), (1, 8, 128), cfg.dtype)
    m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    for mode in ('local_qk+local_o', 'local_qk+full'):
      module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
          head_dim=64, max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode=mode, attention_type=cfg.attention_type)
      p = module.init({'params':jax.random.key(3), 'aqt':jax.random.key(4)},
          *args, M_in=m, deterministic=True, layer_index=2)
      self.assertNotIn('W_o_row_basis', p['params'])
      local = mode == 'local_qk+local_o'
      self.assertEqual('W_o_row_mix' in p['params'], local)
      self.assertEqual(p['params']['W_R']['kernel'].value.shape[-1], 8 if local else 40)
      def loss(params):
        y, mm = module.apply({'params':params}, *args, M_in=m, deterministic=True, layer_index=2)
        return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(mm.astype(jnp.float32)**2)
      grad = jax.grad(loss)(p['params'])
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad)))
      if not local:
        cfg.get_keys()['bam_local_o_share_v_row_basis'] = False
        ref = module.init({'params':jax.random.key(3), 'aqt':jax.random.key(4)},
            *args, M_in=m, deterministic=True, layer_index=2)
        for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(ref)):
          np.testing.assert_array_equal(a, b)
        cfg.get_keys()['bam_local_o_share_v_row_basis'] = True


if __name__ == '__main__':
  absltest.main()

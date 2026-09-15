"""LocalV activation, rank selection and independence from LocalO."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import max_utils
from bam_local_fetch_test import LocalFetchTest
from layers.attentions import BamAttention


class LocalVModeTest(absltest.TestCase):
  config = LocalFetchTest.config

  def module(self, cfg, mode, layer_inx=0):
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    return BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
        head_dim=64, max_target_length=8, max_prefill_predict_length=8,
        mesh=mesh, attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        layer_mode=mode, layer_inx=layer_inx, attention_type=cfg.attention_type)

  def metadata(self, cfg, mode, layer_inx=0):
    output, _ = self.module(cfg, mode, layer_inx).init_with_output(
        {'params': jax.random.key(11)},
        method=lambda mod: ({name: arm.rank for name, arm in mod._local_arms.items()},
                            hasattr(mod, 'W_lv_gate')))
    return output

  def test_layer_marker_disables_v_regardless_of_rank(self):
    cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVNonScan')
    for rank in (None, 2, 4):
      cfg.get_keys()['bam_local_v_rank'] = rank
      arms, shared_gate = self.metadata(cfg, 'local_qk+local_o')
      self.assertEqual(set(arms), {'q', 'k'})
      self.assertFalse(shared_gate)

  def test_none_is_shared_and_positive_rank_is_independent(self):
    cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVNonScan')
    cfg.get_keys()['bam_local_q_rank'] = 4
    for rank in (None, 1, 2, 4):
      cfg.get_keys()['bam_local_v_rank'] = rank
      arms, shared_gate = self.metadata(cfg, 'local_qk+local_v+local_o')
      self.assertEqual((arms['q'], arms['k']), (4, 4))
      self.assertEqual(arms.get('v'), rank)
      self.assertEqual(shared_gate, rank is None)

  def test_per_layer_ranks_keep_none_for_v_but_k_follows_q(self):
    cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVNonScan')
    cfg.get_keys().update(bam_local_q_rank=[4, 2, 1, 2],
                          bam_local_k_rank=[None, 1, None, None],
                          bam_local_v_rank=[None, 4, None, 2])
    for i, expected in enumerate(({'q': 4, 'k': 4}, {'q': 2, 'k': 1, 'v': 4},
                                 {'q': 1, 'k': 1}, {'q': 2, 'k': 2, 'v': 2})):
      arms, shared_gate = self.metadata(cfg, 'local_qk+local_v+local_o', i)
      self.assertEqual(arms, expected)
      self.assertEqual(shared_gate, i in (0, 2))

  def test_independent_ranks_must_be_positive_integers(self):
    cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVNonScan')
    for rank in (0, -1, 1.5, True):
      cfg.get_keys()['bam_local_v_rank'] = rank
      with self.assertRaisesRegex(ValueError, 'rank must be positive'):
        self.metadata(cfg, 'local_v')
    cfg.get_keys().update(bam_local_q_rank=None, bam_local_v_rank=2)
    with self.assertRaisesRegex(ValueError, 'rank must be positive'):
      self.metadata(cfg, 'local_qk')

  def test_v_only_forward_and_gradients_do_not_require_qk_or_o(self):
    for rank in (None, 2, 4):
      with self.subTest(rank=rank):
        cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVNonScan')
        cfg.get_keys()['bam_local_v_rank'] = rank
        module = self.module(cfg, 'local_v')
        x = jax.random.normal(jax.random.key(1), (1, 8, 128), cfg.dtype)
        m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), cfg.dtype)
        args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
        params = module.init({'params': jax.random.key(3)}, *args, M_in=m)['params']
        self.assertEqual('W_R' in params, rank is None)
        self.assertEqual('W_local_packed' in params, rank is not None)
        params = jax.tree.map(lambda a: a + .01 * jax.random.normal(
            jax.random.key(4), a.shape, a.dtype), params)
        def loss(p):
          y, state = module.apply({'params': p}, *args, M_in=m)
          return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(state.astype(jnp.float32)**2)
        value, grads = jax.value_and_grad(loss)(params)
        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads)))
        kernel = grads['W_R' if rank is None else 'W_local_packed']['kernel'].value
        self.assertGreater(float(jnp.linalg.norm(kernel.astype(jnp.float32))), 0.)
        jax.clear_caches()

  def test_row_only_shared_keeps_o_reader_and_independent_v_column(self):
    cfg = self.config('BamMediumIndependentLLFBAlignedRowLocalVRowSharedColRank4B')
    module = self.module(cfg, cfg.bam_layer_modes[0])
    x = jax.random.normal(jax.random.key(1), (1, 8, 128), cfg.dtype)
    m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    params = module.init({'params': jax.random.key(3)}, *args, M_in=m)['params']
    self.assertIn('W_R', params)
    self.assertIn('W_local_v_col_packed', params)
    self.assertIn('W_lv_row_gate', params)
    self.assertNotIn('W_lv_gate', params)
    self.assertNotIn('W_lv_bias', params)
    self.assertEqual(params['W_local_v_col_packed']['kernel'].value.shape[-1], 138)
    params = jax.tree.map(lambda a: a + .01 * jax.random.normal(
        jax.random.key(4), a.shape, a.dtype), params)
    def loss(p):
      y, state = module.apply({'params': p}, *args, M_in=m)
      return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(state.astype(jnp.float32)**2)
    value, grads = jax.value_and_grad(loss)(params)
    self.assertTrue(bool(jnp.isfinite(value)))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads)))
    for name in ('W_local_v_col_packed', 'W_lv_row_gate'):
      grad = grads[name]['kernel'].value
      self.assertGreater(float(jnp.linalg.norm(grad.astype(jnp.float32))), 0.)
    fetch_params = self.module(cfg, cfg.bam_layer_modes[2], 2).init(
        {'params': jax.random.key(3)}, *args, M_in=m)['params']
    self.assertNotIn('W_local_v_col_packed', fetch_params)
    self.assertNotIn('W_lv_row_gate', fetch_params)


if __name__ == '__main__':
  absltest.main()

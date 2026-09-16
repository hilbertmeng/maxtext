"""LocalV activation, rank selection and independence from LocalO."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import max_utils
import bam_local_fetch_test
from flax import linen as nn
from layers.attentions import BamAttention
from layers.fusion import BamLayerPair


class LocalVModeTest(absltest.TestCase):
  def config(self, exp):
    return bam_local_fetch_test.LocalFetchTest.config(self, exp)

  def module(self, cfg, mode, layer_inx=0, prune_local_vo_row=False):
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    return BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
        head_dim=64, max_target_length=8, max_prefill_predict_length=8,
        mesh=mesh, attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        layer_mode=mode, layer_inx=layer_inx,
        prune_local_vo_row=prune_local_vo_row,
        attention_type=cfg.attention_type)

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

  def test_xl_row_only_shared_keeps_c_column_and_qk_shared_basis(self):
    cfg = self.config('BamXLSharedBasisLocalVRowSharedColRank4CFp32')
    mode = cfg.bam_layer_modes[0]
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    module = BamAttention(
        config=cfg, num_query_heads=2, num_kv_heads=2,
        head_dim=128, max_target_length=8, max_prefill_predict_length=8,
        mesh=mesh, attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        bam_k=cfg.bam_k, bam_v=cfg.bam_v,
        layer_mode=mode, layer_inx=0, attention_type=cfg.attention_type)
    x = jax.random.normal(jax.random.key(21), (1, 8, 256), cfg.dtype)
    m = jax.random.normal(jax.random.key(22), (1, 8, 64, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    params = module.init({'params': jax.random.key(23)}, *args, M_in=m)['params']
    self.assertIn('W_R', params)
    self.assertIn('W_local_v_col_packed', params)
    self.assertIn('W_lv_row_gate', params)
    self.assertNotIn('W_local_packed_v', params)
    self.assertEqual(params['W_local_v_col_packed']['kernel'].value.shape[-1], 138)
    self.assertEqual(params['W_local_v_col_bias'].value.shape, (4, 32))
    metadata, _ = module.init_with_output(
        {'params': jax.random.key(24)}, method=lambda mod: (
            mod._local_v_col_arm.rank_routing,
            mod._local_v_col_arm.key_scale,
            mod._local_v_col_arm.gram_statistics_dtype,
            mod._share_qk_basis))
    self.assertEqual(metadata[0], 'effective_key')
    self.assertEqual(metadata[1], float(cfg.bam_read_key_scale))
    self.assertEqual(metadata[2], jnp.float32)
    self.assertTrue(metadata[3])
    params = jax.tree.map(lambda a: a + .01 * jax.random.normal(
        jax.random.key(25), a.shape, a.dtype), params)
    def loss(p):
      y, state = module.apply({'params': p}, *args, M_in=m)
      return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(state.astype(jnp.float32)**2)
    value, grads = jax.value_and_grad(loss)(params)
    self.assertTrue(bool(jnp.isfinite(value)))
    for name in ('W_local_v_col_packed', 'W_lv_row_gate'):
      self.assertGreater(float(jnp.linalg.norm(
          grads[name]['kernel'].value.astype(jnp.float32))), 0.)
    jax.clear_caches()

  def test_xl_col_only_removes_local_v_row_gate(self):
    cfg = self.config('BamXLSharedBasisLocalVColOnlyRank4CFp32')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    module = BamAttention(
        config=cfg, num_query_heads=2, num_kv_heads=2,
        head_dim=128, max_target_length=8, max_prefill_predict_length=8,
        mesh=mesh, attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        bam_k=cfg.bam_k, bam_v=cfg.bam_v,
        layer_mode=cfg.bam_layer_modes[0], layer_inx=0,
        attention_type=cfg.attention_type)
    x = jax.random.normal(jax.random.key(31), (1, 8, 256), cfg.dtype)
    m = jax.random.normal(jax.random.key(32), (1, 8, 64, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    params = module.init({'params': jax.random.key(33)}, *args, M_in=m)['params']
    self.assertIn('W_R', params)
    self.assertIn('W_local_v_col_packed', params)
    self.assertNotIn('W_lv_row_gate', params)
    self.assertNotIn('W_lv_row_gate_b0', params)
    params = jax.tree.map(lambda a: a + .01 * jax.random.normal(
        jax.random.key(34), a.shape, a.dtype), params)
    def loss(p):
      y, state = module.apply({'params': p}, *args, M_in=m)
      return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(state.astype(jnp.float32)**2)
    value, grads = jax.value_and_grad(loss)(params)
    self.assertTrue(bool(jnp.isfinite(value)))
    self.assertGreater(float(jnp.linalg.norm(
        grads['W_local_v_col_packed']['kernel'].value.astype(jnp.float32))), 0.)
    jax.clear_caches()

  def test_later_block_prunes_only_shared_v_o_rows(self):
    cfg = self.config('BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly')
    module = self.module(
        cfg, cfg.bam_layer_modes[0], prune_local_vo_row=True)
    x = jax.random.normal(jax.random.key(41), (1, 8, 128), cfg.dtype)
    m = jax.random.normal(jax.random.key(42), (1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    params = module.init({'params': jax.random.key(43)}, *args, M_in=m)['params']
    self.assertEqual(params['W_R']['kernel'].value.shape[-1], 8)
    self.assertEqual(params['W_R_gate']['kernel'].value.shape[-1], 1)
    self.assertNotIn('abs_v_row_decoder', params)
    self.assertNotIn('W_lv_row_gate', params)
    self.assertIn('W_local_v_col_packed', params)
    metadata, _ = module.init_with_output(
        {'params': jax.random.key(44)}, method=lambda mod: (
            mod._fetched_arm.read_side, mod._fetched_arm.prune_row,
            tuple(mod._local_arms)))
    self.assertEqual(metadata, ('col', True, ('q', 'k')))
    params = jax.tree.map(lambda a: a + .01 * jax.random.normal(
        jax.random.key(45), a.shape, a.dtype), params)
    def loss(p):
      y, state = module.apply({'params': p}, *args, M_in=m)
      return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(state.astype(jnp.float32)**2)
    value, grads = jax.value_and_grad(loss)(params)
    self.assertTrue(bool(jnp.isfinite(value)))
    for name in ('W_R', 'W_local_v_col_packed'):
      self.assertGreater(float(jnp.linalg.norm(
          grads[name]['kernel'].value.astype(jnp.float32))), 0.)
    jax.clear_caches()

  def test_first_llf_block_is_full_and_scanned_blocks_prune_vo_rows(self):
    cfg = self.config('BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnly')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    h = jnp.ones((1, 8, 128), cfg.dtype)
    m = jnp.zeros((1, 8, 32, 32), cfg.dtype)
    common = (jnp.ones((1, 8), jnp.int32), jnp.arange(8)[None],
              jnp.ones((1, 8), jnp.int32), None, True, 'train', None,
              None, None, None)
    first = BamLayerPair(
        cfg, mesh, 8, all_global_attention=True, first_block=True)
    first_vars = first.init(
        {'params': jax.random.key(51)}, (h, m), *common,
        jnp.asarray(0, jnp.int32))
    first_carry, _ = first.apply(
        first_vars, (h, m), *common, jnp.asarray(0, jnp.int32))

    later = nn.scan(
        BamLayerPair, variable_axes={'params': cfg.param_scan_axis},
        split_rngs={'params': True, 'dropout': False},
        in_axes=(nn.broadcast,) * 10 + (0,), length=1,
        metadata_params={nn.PARTITION_NAME: 'layers'})(
            cfg, mesh, 8, all_global_attention=True)
    later_vars = later.init(
        {'params': jax.random.key(52)}, first_carry, *common,
        jnp.asarray([1], jnp.int32))
    final_carry, _ = later.apply(
        later_vars, first_carry, *common, jnp.asarray([1], jnp.int32))

    def attention(params, layer):
      return params[layer]['block']['self_attention']

    for layer in ('local_0', 'local_1'):
      first_attention = attention(first_vars['params'], layer)
      later_attention = attention(later_vars['params'], layer)
      self.assertEqual(first_attention['W_R']['kernel'].value.shape[-1], 40)
      self.assertEqual(later_attention['W_R']['kernel'].value.shape[-1], 8)
      self.assertIn('W_lv_row_gate', first_attention)
      self.assertNotIn('W_lv_row_gate', later_attention)
      self.assertIn('abs_v_row_decoder', first_attention)
      self.assertNotIn('abs_v_row_decoder', later_attention)
    fetch = attention(later_vars['params'], 'fetch_2')
    self.assertEqual(fetch['W_R']['kernel'].value.shape[-1], 40)
    self.assertIn('abs_v_row_decoder', fetch)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(x)))
                        for x in jax.tree.leaves(final_carry)))
    self.assertGreater(float(jnp.linalg.norm(
        (final_carry[1] - first_carry[1]).astype(jnp.float32))), 0.)
    jax.clear_caches()

  def test_o_row_gelu_lora_applies_only_to_retained_rows(self):
    cfg = self.config(
        'BamMediumIndependentLLFBAlignedRowLocalVORowFirstBlockOnlyORowR256Gelu')
    cfg.get_keys()['bam_o_row_bottleneck_dim'] = 16
    x = jax.random.normal(jax.random.key(61), (1, 8, 128), cfg.dtype)
    m = jax.random.normal(jax.random.key(62), (1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))

    for name, mode, layer, prune, active in (
        ('first_local', cfg.bam_layer_modes[0], 0, False, True),
        ('later_local', cfg.bam_layer_modes[0], 0, True, False),
        ('fetch', cfg.bam_layer_modes[2], 2, False, True)):
      with self.subTest(name=name):
        module = self.module(cfg, mode, layer, prune)
        params = module.init({'params': jax.random.key(63)}, *args,
                             M_in=m)['params']
        self.assertEqual('W_R_row_down' in params, active)
        self.assertEqual('W_R_row_up' in params, active)
        self.assertEqual(params['W_R']['kernel'].value.shape[-1], 8)
        if active:
          self.assertEqual(
              params['W_R_row_down']['kernel'].value.shape, (128, 16))
          self.assertEqual(
              params['W_R_row_up']['kernel'].value.shape, (16, 2, 1, 32))
          def loss(p):
            y, state = module.apply({'params': p}, *args, M_in=m)
            return (jnp.mean(y.astype(jnp.float32)**2)
                    + jnp.mean(state.astype(jnp.float32)**2))
          value, grads = jax.value_and_grad(loss)(params)
          self.assertTrue(bool(jnp.isfinite(value)))
          self.assertGreater(float(jnp.linalg.norm(
              grads['W_R_row_up']['kernel'].value.astype(jnp.float32))), 0.)
          self.assertEqual(float(jnp.linalg.norm(
              grads['W_R_row_down']['kernel'].value.astype(jnp.float32))), 0.)
        jax.clear_caches()


if __name__ == '__main__':
  absltest.main()

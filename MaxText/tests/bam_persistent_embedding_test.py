"""Embedding-only persistent bank: budget, independent writes, read timing and scan gradients."""
import math
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from bam_prop_test_utils import MLPWriteTest
import max_utils
import train_compile
from layers import attentions, fusion, initializers
from layers.models import EmbeddingBamWrite, Transformer


PARENT = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
EXPERIMENT = 'BamMediumPropMShiftReadG050PersistentEmbedMLastQKVOTruePile'


class ReadProbe(attentions.BamAttention):
  def _matrix_for_read(self, matrix):
    self.sow('intermediates', 'observed_read_matrix', matrix)
    return super()._matrix_for_read(matrix)


class PersistentEmbeddingTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, experiment=EXPERIMENT):
    cfg = self.config(experiment, bam_splash_attention=False, base_emb_dim=150,
                      base_num_query_heads=2, base_num_kv_heads=2,
                      base_num_decoder_layers=9, base_mlp_dim=48, vocab_size=32,
                      dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(
        emb_bam_num_head=2, bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, mlp_dim_by_block=[48, 40, 48],
        bam_persistent_embedding_last_mlp_dim=40,
        bam_layer_modes=['local_qk+local_v+local_o'] * 9)
    return cfg

  def test_formal_budget_and_last_layer_read_write_shapes(self):
    cfg = self.config(EXPERIMENT, bam_splash_attention=False)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
    params = flatten_dict(nn.unbox(shaped[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in params.values()), 432113283)
    root = ('params', 'decoder')
    last = root + ('tail_layers', 'fetch_2', 'block')
    self.assertEqual(params[root + ('layers', 'local_0', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                     (1200, 5, 3802))
    self.assertEqual(params[last + ('mlp', 'wi_0', 'kernel')].shape, (1200, 1, 3617))
    for name in ('W_lq_c8_up', 'W_lk_c8_up'):
      self.assertEqual(params[last + ('self_attention', name, 'kernel')].shape,
                       (256, 1, 16, 48))
    self.assertEqual(params[last + ('self_attention', 'W_R_up', 'kernel')].shape,
                     (256, 1, 16, 1, 48))
    self.assertEqual(params[last + ('self_attention', 'P_loc_up', 'kernel')].shape,
                     (256, 1, 16, 32))
    self.assertEqual(params[root + ('tail_layers', 'local_1', 'block',
                                    'mlp_address_up', 'kernel')].shape[-2:], (16, 32))
    self.assertEqual(params[root + ('embedding_bam_write', 'W_emb_persistent_v_up', 'kernel')].shape,
                     (256, 16, 16))
    self.assertEqual(params[root + ('compose_final_three_source',
                                    'dynamic_dense_conn1', 'kernel')].shape, (1200, 64))
    self.assertFalse(cfg.bam_sqrt_n_scale)
    self.assertFalse(cfg.bam_mudd_tail_blocks)
    self.assertTrue(cfg.bam_mudd_final_output)
    self.assertFalse(cfg.qk_norm)

  def test_shared_content_preserves_parent_and_bank_gates_are_independent(self):
    cfg = self.tiny()
    original_cfg = self.tiny(PARENT)
    inputs = .006 * jax.random.normal(jax.random.key(1), (1, 4, 150))
    def module(config):
      return EmbeddingBamWrite(config, 2, jnp.float32, jnp.float32, None,
                               initializers.get_bam_init(config))
    current, original = module(cfg), module(original_cfg)
    params = nn.unbox(current.init(jax.random.key(3), inputs)['params'])
    old_params = nn.unbox(original.init(jax.random.key(3), inputs)['params'])
    for path, value in flatten_dict(old_params).items():
      np.testing.assert_array_equal(flatten_dict(params)[path], value)
    matrix, persistent = current.apply({'params': params}, inputs)
    np.testing.assert_array_equal(matrix, original.apply({'params': old_params}, inputs))
    self.assertEqual(persistent.shape, (1, 4, 75, 2))
    self.assertFalse(np.allclose(persistent, matrix[..., :2]))
    np.testing.assert_allclose(params['emb_persistent_gw_b0'], math.log(.1 / .9), rtol=1e-6)
    gradients = jax.grad(lambda p: jnp.mean(current.apply({'params': p}, inputs)[1] ** 2))(params)
    np.testing.assert_array_equal(gradients['W_emb_g']['kernel'], 0.)
    np.testing.assert_array_equal(gradients['emb_gw_b0'], 0.)
    for name in ('W_emb_u', 'W_emb_persistent_g', 'W_emb_persistent_v_down', 'W_emb_persistent_v_up'):
      self.assertGreater(float(jnp.sum(gradients[name]['kernel'] ** 2)), 0., name)
    disabled = dict(params, emb_persistent_gw_b0=jnp.full((2,), -100.))
    unchanged, zero = current.apply({'params': disabled}, inputs)
    np.testing.assert_array_equal(unchanged, matrix)
    np.testing.assert_array_equal(zero, 0.)

  def test_only_last_layer_concatenates_after_shift_and_writes_original_bank(self):
    cfg = self.tiny()
    cfg.get_keys()['bam_lambda_decay'] = 1.
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x = jax.random.normal(jax.random.key(6), (1, 4, 150))
    matrix = jax.random.normal(jax.random.key(7), (1, 4, 75, 32))
    persistent = jax.random.normal(jax.random.key(8), (1, 4, 75, 2))
    positions, segments = jnp.arange(4)[None], jnp.ones((1, 4), jnp.int32)
    for layer in (0, 8):
      module = ReadProbe(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=75,
          bam_k=75, bam_v=32, layer_inx=layer,
          max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=jnp.float32, weight_dtype=jnp.float32,
          layer_mode='local_qk+local_v+local_o', read_side='col')
      args = (x, x, positions, segments)
      params = nn.unbox(module.init({'params': jax.random.key(9)}, *args,
                                   M_in=matrix, persistent_M=persistent, deterministic=True)['params'])
      params['gw_b0'] = jnp.full_like(params['gw_b0'], -100.)
      (output, written), updates = module.apply({'params': params}, *args,
          M_in=matrix, persistent_M=persistent, deterministic=True, mutable=['intermediates'])
      shifted = attentions.mix_bam_previous_token(matrix, jnp.full((1, 4, 32), .5), segments, positions)
      expected = jnp.concatenate((shifted, persistent), -1) if layer == 8 else shifted
      np.testing.assert_allclose(updates['intermediates']['observed_read_matrix'][0], expected,
                                 rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(written, matrix)
      self.assertTrue(np.isfinite(np.asarray(output)).all())
      self.assertEqual(params['M_shift_gate']['kernel'].shape, (150, 32))
      if layer == 8:
        grad = jax.grad(lambda p: jnp.mean(module.apply({'params': params}, *args,
            M_in=matrix, persistent_M=p, deterministic=True)[0] ** 2))(persistent)
        self.assertGreater(float(jnp.sum(grad ** 2)), 0.)

  def test_scanned_persistent_and_l0_carry_causality_and_active_gradients(self):
    cfg = self.tiny()
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.asarray([[1, 4, 8, 2]], jnp.int32)
    positions, segments = jnp.arange(4)[None], jnp.ones_like(tokens)
    rng = {name: jax.random.key(i + 11) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(model.init(rng, tokens, positions, tokens, segments,
                                   decoder_segment_ids=segments, enable_dropout=False)['params'])
      flat = flatten_dict(params)
      for path, value in flat.items():
        if path[-1] == 'mix_bias': flat[path] = jnp.asarray([.8, .1, .1])
        elif 'dynamic_dense_conn2' in path: flat[path] = value + .01
      params = unflatten_dict(flat)
      def forward(p, t=tokens, capture=False):
        return model.apply({'params': p}, t, positions, tokens, segments,
            decoder_segment_ids=segments, enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(module, (fusion.FusionDecoderLayer, EmbeddingBamWrite)))
                                  if capture else False,
            mutable=['intermediates'] if capture else False)
      output, captured = forward(params, capture=True)
      leaves = captured['intermediates']['decoder']
      initial_persistent = leaves['embedding_bam_write']['__call__'][0][1]
      first = leaves['layers']['local_0']['__call__'][0][0]
      initial_l0 = first[0][0]
      for stage in ('layers', 'tail_layers'):
        for layer in ('local_0', 'local_1', 'fetch_2'):
          carried = leaves[stage][layer]['__call__'][0][0]
          self.assertEqual(len(carried), 5)
          self.assertEqual(carried[1].shape[-1], 32)
          np.testing.assert_allclose(carried[3], jnp.broadcast_to(initial_l0, carried[3].shape),
                                     rtol=2e-6, atol=2e-6)
          np.testing.assert_array_equal(carried[4], jnp.broadcast_to(initial_persistent, carried[4].shape))
      changed = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, changed)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0])))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      for name in ('W_emb_u', 'W_emb_persistent_g', 'W_emb_persistent_v_down',
                   'W_emb_persistent_v_up', 'dynamic_dense_conn2', 'W_lq_c8_up', 'W_lk_c8_up', 'W_R_up'):
        selected = [v for path, v in gradients.items() if name in path]
        self.assertTrue(selected, name)
        self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)


if __name__ == '__main__':
  unittest.main()

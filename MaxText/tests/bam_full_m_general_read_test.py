"""Full-M Prop SOTA budgets, parameter sharding, scan/tail gradients and health."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer

MEDIUM = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
XL = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'


class FullMGeneralReadTest(MLPWriteTest, unittest.TestCase):
  def check_layout(self, experiment, count, rank, width, heads, layers):
    cfg = self.config(experiment, bam_splash_attention=False)
    self.assertEqual(cfg.emb_dim, width)
    self.assertEqual(cfg.num_query_heads, heads)
    self.assertEqual(cfg.num_decoder_layers, layers)
    self.assertTrue(cfg.scan_layers and cfg.bam_local_full_m_read_share_down)
    self.assertTrue(cfg.bam_general_column_read)
    self.assertFalse(cfg.bam_general_matrix_write)
    self.assertIsNone(cfg.bam_abs_v_compression_dim)
    self.assertEqual(cfg.bam_local_full_m_read_bottleneck_dim, rank)
    self.assertEqual(cfg.bam_static_read_gate_init, .99)
    self.assertEqual(cfg.bam_read_key_scale, .1)
    self.assertEqual(cfg.bam_read_gate_init, .05)
    self.assertGreater(cfg.tensorboard_gcs_rotate_period, 0)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, _, partitions, _ = train_compile.get_shaped_inputs(mesh, cfg)
    params = flatten_dict(nn.unbox(shaped[0].params))
    specs = flatten_dict(nn.unbox(partitions.params))
    self.assertEqual(sum(math.prod(v.shape) for v in params.values()), count)
    factor = math.prod(mesh.shape[a] for a in (
        'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose',
        'tensor_sequence', 'stage', 'expert'))
    local = sum(math.prod(specs[path].shard_shape(v.shape)) for path, v in params.items())
    self.assertLess(local/(count/factor)-1, cfg.sharding_tolerance)
    self.assertEqual(any('final_local_layer' in path for path in params), layers == 28)
    self.assertFalse(any('AbsVCompression' in part for path in params for part in path))
    for path, value in params.items():
      block_axis = () if 'final_local_layer' in path else (layers//3,)
      if 'full_m_read_down' in path:
        self.assertEqual(value.shape, (width,) + block_axis + (rank,))
      if any(n in path for n in ('W_lq_c8_up', 'W_lk_c8_up', 'W_R_up')):
        extra = (1,) if 'W_R_up' in path else ()
        self.assertEqual(value.shape, (rank,) + block_axis + (heads,) + extra + (cfg.bam_v,))
        self.assertEqual(math.prod(specs[path].shard_shape(value.shape)), math.prod(value.shape)//factor)

  def test_medium_exact_budget_and_sharding(self):
    self.check_layout(MEDIUM, 432111104, 256, 1200, 16, 18)

  def test_xl_exact_budget_and_tail_sharding(self):
    self.check_layout(XL, 1432386360, 400, 1920, 20, 28)

  def check_active_reads(self, experiment, width, heads, layers):
    cfg = self.config(experiment, bam_splash_attention=False)
    cfg.get_keys().update(
        base_emb_dim=width, emb_dim=width, base_num_query_heads=heads,
        base_num_kv_heads=heads, num_query_heads=heads, num_kv_heads=heads,
        emb_bam_num_head=heads, bam_mlp_write_num_heads=0,
        bam_write_v_bottleneck_dim=16, emb_bam_v_bottleneck_dim=16,
        bam_mlp_write_address_rank=16, base_num_decoder_layers=layers,
        num_decoder_layers=layers, bam_layer_modes=['local_qk+local_v+local_o']*layers,
        base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32, 24, 32],
        bam_final_local_mlp_dim=32, vocab_size=128,
        dtype='float32', weight_dtype='float32')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 4, 8, 2]], jnp.int32)
    positions = jnp.arange(4)[None]
    segments = jnp.ones_like(tokens)
    rng = {name: jax.random.key(i) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(model.init(rng, tokens, positions, segments, tokens)['params'])

      def forward(p, health=False):
        result = model.apply({'params': p}, tokens, positions, segments, tokens,
                             enable_dropout=False, rngs=rng,
                             mutable=['intermediates'] if health else False)
        output, stats = result if health else (result, None)
        output = output[0] if isinstance(output, tuple) else output
        return (output, stats) if health else output

      initial, stats = forward(params, True)
      cfg.get_keys().update(bam_general_column_read=False)
      parent_params = nn.unbox(model.init(rng, tokens, positions, segments, tokens)['params'])
      child_flat = flatten_dict(params)
      for path, value in flatten_dict(parent_params).items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(child_flat[path]), err_msg='/'.join(path))
      np.testing.assert_allclose(np.asarray(initial), np.asarray(forward(params)), rtol=3e-5, atol=3e-6)
      cfg.get_keys().update(bam_general_column_read=True)
      metrics = {'scalar': {}}
      train.record_bam_concat_health_metrics(metrics, stats, cfg)
      health = {key: value for key, value in metrics['scalar'].items() if '/general_' in key}
      self.assertTrue(health)
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in health.values()))
      for key, value in health.items():
        if key.endswith('/mean') and '_static_gate/' in key:
          np.testing.assert_allclose(float(value), .99, rtol=2e-6)
      for layer in range(layers):
        for name in ('q', 'k', 'vo'):
          self.assertIn(f'bam/concat/general_{name}_after_bias_key_norm/layer_{layer:03d}/min_rms', health)
        self.assertIn(f'bam/concat/general_q_key_alignment/layer_{layer:03d}/mean_cosine', health)
      loss, gradients = jax.jit(jax.value_and_grad(
          lambda p: jnp.mean(forward(p).astype(jnp.float32)**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(gradients)))
      flat = flatten_dict(gradients)
      for name in ('full_m_read_down', 'W_lq_c8_up', 'W_lk_c8_up', 'W_R_up',
                   'general_q_pre_bias', 'general_k_pre_bias', 'general_vo_pre_bias', 'general_v_static_gate'):
        leaves = [v for path, v in flat.items() if name in path]
        self.assertTrue(leaves, name)
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves), 0, name)
      if layers == 4:
        self.assertTrue(any('final_local_layer' in path and 'general_v_static_gate' in path for path in flat))

  def test_medium_parent_initialization_active_gradients_and_health(self):
    self.check_active_reads(MEDIUM, 300, 4, 3)

  def test_xl_parent_initialization_scan_tail_gradients_and_health(self):
    self.check_active_reads(XL, 768, 8, 4)


if __name__ == '__main__':
  unittest.main()

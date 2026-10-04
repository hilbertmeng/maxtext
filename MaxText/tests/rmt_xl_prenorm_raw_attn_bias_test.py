"""XL transfer: exact budget, affine-write initialization, gradients and health."""
import contextlib
import copy
import io
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

EXP = ('RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoO'
       'SharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWriteAttnWriteContentBias')


class XLPreNormRawAttentionBiasTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget_and_transfer_scope(self):
    cfg = self.config(EXP)
    self.assertEqual((cfg.num_decoder_layers, cfg.num_query_heads, cfg.head_dim), (28, 20, 96))
    self.assertEqual(cfg.rmt_dynamic_write_bottleneck_dim, 384)
    self.assertEqual(cfg.rmt_matrix_read_norm, 'none')
    self.assertEqual(cfg.rmt_carry_health_layers, 'all')
    self.assertEqual(cfg.keep_period, 2000)
    self.assertEqual(cfg.max_to_keep, 2)
    self.assertEqual(cfg.mlp_dim, 6644)
    for key in ('rmt_static_qk_zero_init', 'rmt_static_v_zero_init',
                'rmt_embedding_shared_write_norm', 'rmt_embedding_seed_key_zero_init',
                'rmt_mlp_input_pre_norm', 'rmt_mlp_shared_raw_write',
                'rmt_attn_write_content_pre_norm_bias', 'rmt_record_stability_health'):
      self.assertTrue(cfg.get_keys()[key], key)
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(0)))
    count = sum(math.prod(x.shape) for x in jax.tree.leaves(params))
    self.assertEqual(count, 1432399960)
    self.assertLess(abs(count - 1432398720), abs(count - 161280 - 1432398720))
    self.assertLess(abs(count - 1432398720), abs(count + 161280 - 1432398720))
    layers = params['decoder']['layers']
    self.assertEqual(layers['attn_write_content_bias']['bias'].shape, (20, 28, 96))
    self.assertEqual(layers['mlp_input_norm']['scale'].shape, (1920, 28))
    for key in ('static_q_read_bias', 'static_v_read_bias', 'static_mlp_read_bias', 'mlp_write_content_bias'):
      self.assertNotIn(key, layers)
    print('XL_TRANSFER_PARAMS', count, 'MHA_DELTA', count - 1432398720)

  def test_xl_ratio_zero_bias_gradients_and_health(self):
    # Preserve XL row/head/C ratios while shrinking only content width and LoRA work.
    cfg = self.config(EXP, base_num_decoder_layers=2, base_emb_dim=960, head_dim=48,
                      base_mlp_dim=96, vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32, rmt_rope_qk_dim=12,
                          rmt_dynamic_write_bottleneck_dim=64)
    parent_cfg = self.config(EXP, base_num_decoder_layers=2, base_emb_dim=960, head_dim=48,
                             base_mlp_dim=96, vocab_size=128)
    parent_cfg.get_keys().update(dtype=jnp.float32, rmt_rope_qk_dim=12,
                                rmt_dynamic_write_bottleneck_dim=64,
                                rmt_attn_write_content_pre_norm_bias=False)
    model, args = self.model_args(cfg)
    parent, parent_args = self.model_args(parent_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(4), **args)['params'])
      old = nn.unbox(parent.init(jax.random.key(4), **parent_args)['params'])
    stripped = copy.deepcopy(params)
    bias = stripped['decoder']['layers'].pop('attn_write_content_bias')['bias']
    np.testing.assert_array_equal(bias, 0)
    self.assertEqual(jax.tree.structure(stripped), jax.tree.structure(old))
    for a, b in zip(jax.tree.leaves(stripped), jax.tree.leaves(old), strict=True):
      np.testing.assert_array_equal(a, b)
    def loss(p):
      y, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
      return jnp.mean(y[0]), aux
    with contextlib.redirect_stdout(io.StringIO()):
      (value, aux), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
      old_value = jnp.mean(parent.apply({'params': old}, **parent_args)[0])
    np.testing.assert_allclose(value, old_value, rtol=1e-6, atol=1e-6)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((value, aux, grad))))
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['attn_write_content_bias']['bias'])), 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['mlp_input_norm']['scale'])), 0.)
    from train import record_rmt_dynamic_health_metrics
    metrics = {'scalar': {}}
    record_rmt_dynamic_health_metrics(metrics, aux, cfg)
    for layer in range(2):
      self.assertAlmostEqual(float(metrics['scalar'][f'rmt/mlp_input/layer_{layer:03d}/actual_input_rms']), 1., delta=.01)
      self.assertIn(f'rmt/carry/layer_{layer:03d}/output_raw_rms', metrics['scalar'])
      self.assertIn(f'rmt/carry/layer_{layer:03d}/token_mean_energy_fraction', metrics['scalar'])


if __name__ == '__main__':
  unittest.main()

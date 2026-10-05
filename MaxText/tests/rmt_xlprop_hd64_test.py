"""HD64 RMT transfer: exact budget and consumed normalized-write scan path."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from exp import RMTHealthDefaults
from tests.rmt_xlprop_test import XLPropTest

EXP = 'RMTXLPropHD64T4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero'

class HD64RMTTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget_and_transfer(self):
    import exp
    cls = getattr(exp, EXP)
    self.assertTrue(issubclass(cls, RMTHealthDefaults))
    self.assertEqual((cls.max_target_length, cls.learning_rate_schedule_steps, cls.steps), (4096, 24000, 24000))
    self.assertEqual(cls.learning_rate, 2.5e-4)
    cfg = self.config(EXP)
    self.assertEqual((cfg.emb_dim, cfg.num_decoder_layers, cfg.num_query_heads, cfg.head_dim), (1280, 28, 20, 64))
    self.assertEqual((cfg.rmt_reskey_dim, cfg.rmt_dynamic_compression_dim, cfg.rmt_rope_qk_dim), (60, 10, 16))
    self.assertEqual(cfg.rmt_dynamic_write_bottleneck_dim, 384)
    self.assertEqual(cfg.mlp_dim, 4109)
    self.assertEqual(cfg.rmt_carry_health_layers, 'all')
    self.assertEqual((cfg.checkpoint_period, cfg.keep_period, cfg.max_to_keep), (250, 2000, 2))
    self.assertEqual(cfg.rmt_matrix_read_norm, 'none')
    for key in ('rmt_static_qk_zero_init', 'rmt_static_v_zero_init',
                'rmt_embedding_shared_write_norm', 'rmt_embedding_seed_key_zero_init',
                'rmt_dynamic_embedding_write', 'rmt_dynamic_unembedding_direct_read',
                'rmt_record_stability_health'):
      self.assertTrue(cfg.get_keys()[key], key)
    for key in ('rmt_mlp_input_pre_norm', 'rmt_mlp_shared_raw_write',
                'rmt_attn_write_content_pre_norm_bias', 'rmt_block_scan',
                'rmt_dynamic_o_enabled'):
      self.assertFalse(cfg.get_keys().get(key, False), key)
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    count = sum(math.prod(v.shape) for v in jax.tree.leaves(tree))
    self.assertEqual(count, 679685720)
    target = 679645440
    self.assertLess(abs(count-target), abs(count-target-107520))
    self.assertLess(abs(count-target), abs(count-target+107520))
    dec = tree['decoder']
    self.assertEqual(dec['dynamic_embedding_write']['address_up'].shape, (384, 1200))
    self.assertEqual(dec['dynamic_unembedding_read']['key_kernel'].shape, (1280, 800))
    self.assertNotIn('compression', dec['dynamic_unembedding_read'])
    self.assertIn('final_matrix_norm', dec)
    self.assertNotIn('attn_norm', dec['layers'])
    self.assertNotIn('mlp_norm', dec['layers'])
    print('HD64_RMT_PARAMS', count, 'MHA_DELTA', count-target)

  def test_consumed_scan_gradient_and_health(self):
    cfg = self.config(EXP, base_num_decoder_layers=2, base_mlp_dim=96, vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32)
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(3), **args)['params'])
      def loss(p):
        output, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
        return jnp.mean(output[0]), aux
      (value, aux), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value, aux, grad))))
    np.testing.assert_array_equal(params['decoder']['layers']['qkv_key'], 0.)
    np.testing.assert_array_equal(params['decoder']['seed_key'], 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])), 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_unembedding_read']['key_kernel'])), 0.)
    from train import record_rmt_dynamic_health_metrics
    metrics = {'scalar': {}}
    record_rmt_dynamic_health_metrics(metrics, aux, cfg)
    for layer in range(2):
      for suffix in ('output_raw_rms', 'token_mean_energy_fraction'):
        self.assertIn(f'rmt/carry/layer_{layer:03d}/{suffix}', metrics['scalar'])
    self.assertTrue(all(np.isfinite(v).all() for v in metrics['scalar'].values()))

if __name__ == '__main__':
  unittest.main()

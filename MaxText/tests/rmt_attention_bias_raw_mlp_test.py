"""Attention-only affine reads/writes preserve the prenorm/raw-write MLP path."""
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

BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroMLPInputPreNormSharedRawWrite'
ARMS = [(BASE + 'AttnStaticQVReadWriteBias', 59616, True),
        (BASE + 'AttnWriteContentBias', 21600, False)]
ADDED = ('attn_write_content_bias', 'static_q_read_bias', 'static_v_read_bias')

class AttentionBiasRawMLPTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget_and_scope(self):
    for name, extra, read_bias in ARMS:
      cfg = self.config(name)
      model, args = self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(0)))
      self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(p)), 431780672 + extra)
      layers = p['decoder']['layers']
      self.assertIn('attn_write_content_bias', layers)
      self.assertEqual('static_q_read_bias' in layers, read_bias)
      self.assertEqual('static_v_read_bias' in layers, read_bias)
      for key in ('mlp_write_content_bias', 'static_mlp_read_bias', 'static_k_read_bias'):
        self.assertNotIn(key, layers)
      self.assertTrue(cfg.rmt_mlp_input_pre_norm and cfg.rmt_mlp_shared_raw_write)
      self.assertTrue(cfg.rmt_static_write_content_norm)
      self.assertEqual(cfg.mlp_dim, 4100)
      self.assertEqual(cfg.keep_period, 0)
      self.assertTrue(cfg.rmt_record_dynamic_health)
      print('PARAMS', name, 431780672 + extra)

  def test_zero_parity_and_bias_gradients(self):
    for name, _, read_bias in ARMS:
      cfg = self.config(name, base_emb_dim=512, head_dim=32,
                        base_mlp_dim=96, base_num_decoder_layers=2, vocab_size=128)
      cfg.get_keys()['dtype'] = jnp.float32
      model, args = self.model_args(cfg)
      old_cfg = self.config(BASE, base_emb_dim=512, head_dim=32,
                           base_mlp_dim=96, base_num_decoder_layers=2, vocab_size=128)
      old_cfg.get_keys()['dtype'] = jnp.float32
      old, old_args = self.model_args(old_cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p = nn.unbox(model.init(jax.random.key(4), **args)['params'])
        parent = nn.unbox(old.init(jax.random.key(4), **old_args)['params'])
      stripped = copy.deepcopy(p)
      for key in ADDED:
        if key in stripped['decoder']['layers']:
          value = stripped['decoder']['layers'].pop(key)
          for a in jax.tree.leaves(value): np.testing.assert_array_equal(a, 0)
      for a, b in zip(jax.tree.leaves(stripped), jax.tree.leaves(parent), strict=True):
        np.testing.assert_array_equal(a, b)
      self.assertEqual(jax.tree.structure(stripped), jax.tree.structure(parent))
      def loss(params):
        out, aux = model.apply({'params': params}, **args, mutable=['intermediates'])
        return jnp.mean(out[0]), aux
      with contextlib.redirect_stdout(io.StringIO()):
        old_value = jnp.mean(old.apply({'params': parent}, **old_args)[0])
        fn = jax.jit(jax.value_and_grad(loss, has_aux=True))
        (value, aux), grad = fn(p)
      np.testing.assert_allclose(value, old_value, rtol=1e-6, atol=1e-6)
      self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['attn_write_content_bias']['bias'])), 0)
      if read_bias:
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['static_v_read_bias']['bias'])), 0)
        for _ in range(2):
          p = jax.tree.map(lambda v, g: v - 1e-6 * g, p, grad)
          (value, aux), grad = fn(p)
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['static_q_read_bias']['bias'])), 0)
      self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((value, aux, grad))))
      from train import record_rmt_dynamic_health_metrics
      metrics = {'scalar': {}}
      record_rmt_dynamic_health_metrics(metrics, aux, cfg)
      for layer in range(2):
        self.assertIn(f'rmt/carry/layer_{layer:03d}/output_raw_rms', metrics['scalar'])
        self.assertAlmostEqual(float(metrics['scalar'][f'rmt/mlp_input/layer_{layer:03d}/actual_input_rms']), 1., delta=.01)
      self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers']['mlp_input_norm']['scale'])), 0)

if __name__ == '__main__':
  unittest.main()

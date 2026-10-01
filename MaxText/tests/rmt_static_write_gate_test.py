"""Targeted checks for independent static-write gates on SharedWriteNorm."""
import contextlib
import io
import math
import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers import rmt
import tests.rmt_xlprop_test as helpers

PARENT = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm'
EXP = PARENT + 'StaticGate'


class StaticWriteGateTest(unittest.TestCase):
  config = helpers.XLPropTest.config
  model_args = helpers.XLPropTest.model_args

  def test_full_budget_and_unchanged_backbone(self):
    counts = {}
    for name in (PARENT, EXP):
      cfg = self.config(name)
      model, args = self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        params = nn.unbox(jax.eval_shape(
            lambda k: model.init(k, **args)['params'], jax.random.key(1)))
      counts[name] = sum(math.prod(p.shape) for p in jax.tree.leaves(params))
      self.assertEqual(cfg.mlp_dim, 4078)
      self.assertEqual(cfg.DATASET_VARIANT, 'truepile4096')
      self.assertTrue(cfg.rmt_static_write_content_norm)
      self.assertFalse(cfg.get_keys().get('rmt_embedding_shared_content', False))
      self.assertIn('embedding_write_content', params['decoder'])
      self.assertFalse(cfg.get_keys().get('rmt_pallas_write', False))
      self.assertFalse(cfg.get_keys().get('rmt_block_scan', False))
      if name == EXP:
        for arm in ('attn', 'mlp'):
          gate = params['decoder']['layers'][f'static_{arm}_write_gate']
          kernel_shape, bias_shape = [1200, 16], [16]
          kernel_shape.insert(cfg.param_scan_axis, 18)
          bias_shape.insert(cfg.param_scan_axis, 18)
          self.assertEqual(gate['kernel'].shape, tuple(kernel_shape))
          self.assertEqual(gate['bias'].shape, tuple(bias_shape))
    self.assertEqual(counts[EXP] - counts[PARENT], 2 * 18 * 16 * 1201)
    print('PARAM_COUNTS', counts)

  def test_unit_initialization_and_selective_suppression(self):
    cfg = self.config(EXP, base_emb_dim=512, head_dim=32)
    cfg.get_keys()['dtype'] = jnp.float32
    x = jax.random.normal(jax.random.key(1), (1, 2, 512))
    module = rmt.RMTStaticWriteGate(cfg)
    params = nn.unbox(module.init(jax.random.key(2), x)['params'])
    gate, opening = module.apply({'params': params}, x)
    np.testing.assert_array_equal(gate, 1.)
    np.testing.assert_allclose(opening, .9, rtol=1e-6)
    np.testing.assert_array_equal(module.apply({'params': params}, x.astype(jnp.bfloat16))[0], 1.)
    altered = dict(params, bias=jnp.concatenate((jnp.full((8,), -20.), params['bias'][8:])))
    closed, _ = module.apply({'params': altered}, x)
    self.assertLess(float(jnp.max(closed[..., :8])), 1e-7)
    np.testing.assert_array_equal(closed[..., 8:], 1.)
    y = jax.random.normal(jax.random.key(3), (1, 2, 16, 32))
    key = jax.random.normal(jax.random.key(4), (16, 48))
    original = rmt.static_layer_write(y, key, cfg)
    np.testing.assert_array_equal(rmt.static_layer_write(y, key, cfg, gate), original)
    expected = rmt.static_layer_write(y * jnp.concatenate((jnp.zeros((8,)), jnp.ones((8,))))[None, None, :, None], key, cfg)
    np.testing.assert_allclose(rmt.static_layer_write(y, key, cfg, closed), expected,
                               rtol=1e-5, atol=1e-5)

  def test_scanned_parent_equivalence_trainable_gates_and_health(self):
    kwargs = dict(base_num_decoder_layers=2, base_emb_dim=512, head_dim=32,
                  base_mlp_dim=128, vocab_size=128)
    cfg = self.config(EXP, **kwargs)
    cfg.get_keys()['dtype'] = jnp.float32
    model, args = self.model_args(cfg)
    parent_cfg = self.config(PARENT, **kwargs)
    parent_cfg.get_keys()['dtype'] = jnp.float32
    parent_model, _ = self.model_args(parent_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(3), **args)['params'])
      def loss(p):
        output, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
        return jnp.sum(output[0]), (output, aux)
      (value, (output, aux)), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
      parent_output = jax.jit(lambda p: parent_model.apply({'params': p}, **args))(params)
    for a, b in zip(jax.tree.leaves(output), jax.tree.leaves(parent_output)):
      np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((value, aux, grad))))
    for arm in ('attn', 'mlp'):
      self.assertGreater(float(jnp.linalg.norm(
          grad['decoder']['layers'][f'static_{arm}_write_gate']['kernel'])), 0.)
    names = rmt.dynamic_health_names(
        record_write_scale_health=True, record_static_write_gates=True)
    health = aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    self.assertEqual(health.shape, (2, len(names)))
    for arm in ('attn', 'mlp'):
      np.testing.assert_allclose(health[:, names.index(f'{arm}_static_write_gate_mean')], .9, rtol=1e-6)
      np.testing.assert_array_equal(health[:, names.index(f'{arm}_static_write_gate_effective_mean')], 1.)
      np.testing.assert_array_equal(health[:, names.index(f'{arm}_static_write_gate_frac_lt_005')], 0.)
    from train import record_rmt_dynamic_health_metrics
    metrics = {'scalar': {}}
    record_rmt_dynamic_health_metrics(metrics, aux, cfg)
    np.testing.assert_allclose(
        metrics['scalar']['rmt/dynamic/layer_001/mlp_static_write_gate_mean'], .9, rtol=1e-6)
    self.assertEqual(float(metrics['scalar']['rmt/dynamic/layer_001/mlp_static_write_gate_effective_mean']), 1.)
    self.assertIn('rmt/dynamic/layer_001/mlp_write_static_rms', metrics['scalar'])


if __name__ == '__main__':
  unittest.main()

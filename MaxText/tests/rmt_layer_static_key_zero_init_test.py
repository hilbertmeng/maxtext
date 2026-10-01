"""Layer static addresses start at zero, learn, and preserve the parent scope."""
import contextlib
import io
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from layers import rmt
from tests.rmt_xlprop_test import XLPropTest


BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm'
EXP = BASE + 'StaticKeyZeroInit'
KEYS = ('attn_write_key', 'mlp_write_key')


class LayerStaticKeyZeroInitTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget_and_scope(self):
    old, cfg = self.config(BASE), self.config(EXP)
    allowed = {'model_name', 'exp_class', 'compare_runs',
               'rmt_attn_write_key_zero_init', 'rmt_mlp_write_key_zero_init',
               'base_output_directory', 'checkpoint_dir', 'metrics_dir', 'bucket_logging_dir'}
    changed = {k for k, value in cfg.get_keys().items() if value != old.get_keys().get(k)}
    self.assertFalse(changed - allowed, changed)
    self.assertEqual(cfg.mlp_dim, 4078)
    self.assertTrue(cfg.rmt_static_write_content_norm)
    self.assertFalse(cfg.get_keys().get('rmt_embedding_seed_key_zero_init', False))
    self.assertFalse(cfg.get_keys().get('rmt_embedding_shared_content', False))
    self.assertTrue(cfg.scan_layers)
    self.assertFalse(cfg.rmt_block_scan)
    self.assertEqual(cfg.DATASET_VARIANT, 'truepile4096')
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(tree)), 431903072)
    for key in KEYS:
      self.assertEqual(tree['decoder']['layers'][key].shape, (16, 18, 48))
    self.assertIn('embedding_write_content', tree['decoder'])

  def test_initial_equivalence_gradients_and_addresses_can_learn(self):
    configs = [self.config(name, base_num_decoder_layers=2, base_emb_dim=512,
                           head_dim=32, base_mlp_dim=128, vocab_size=128)
               for name in (BASE, EXP)]
    for cfg in configs:
      cfg.get_keys()['dtype'] = jnp.float32
    parent, args = self.model_args(configs[0])
    model, _ = self.model_args(configs[1])
    with contextlib.redirect_stdout(io.StringIO()):
      before = nn.unbox(parent.init(jax.random.key(2), **args)['params'])
      params = nn.unbox(model.init(jax.random.key(2), **args)['params'])
      old_paths = jax.tree_util.tree_flatten_with_path(before)[0]
      new_paths = jax.tree_util.tree_flatten_with_path(params)[0]
      self.assertEqual(len(old_paths), len(new_paths))
      for (old_path, old), (path, new) in zip(old_paths, new_paths):
        self.assertEqual(old_path, path)
        if path[-1] not in tuple(jax.tree_util.DictKey(key) for key in KEYS):
          np.testing.assert_array_equal(new, old)
      for key in KEYS:
        np.testing.assert_array_equal(params['decoder']['layers'][key], 0.)
        before['decoder']['layers'][key] = jnp.zeros_like(before['decoder']['layers'][key])
      expected = parent.apply({'params': before}, **args)[0]

      def loss(p):
        out, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
        value = -jnp.mean(jax.nn.log_softmax(out[0].astype(jnp.float32), axis=-1)[..., 1])
        return value, (out[0], aux)

      (value, (out, aux)), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value, aux, grads))))
    for key in KEYS:
      for layer_grad in jnp.moveaxis(grads['decoder']['layers'][key], 1, 0):
        self.assertGreater(float(jnp.linalg.norm(layer_grad)), 0.)
      params['decoder']['layers'][key] -= 1e-4 * grads['decoder']['layers'][key]
      self.assertGreater(float(jnp.linalg.norm(params['decoder']['layers'][key])), 0.)
    for arm in ('attn', 'mlp'):
      self.assertGreater(float(jnp.linalg.norm(
          grads['decoder']['layers']['dynamic_' + arm + '_write']['address_up'])), 0.)
    names = rmt.dynamic_health_names(True, 16, 48)
    initial = aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    _, updated = model.apply({'params': params}, **args, mutable=['intermediates'])
    health = updated['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    for arm in ('attn', 'mlp'):
      index = names.index(arm + '_write_tail32_ratio')
      self.assertGreater(float(jnp.min(initial[:, index])), 1e8)
      self.assertLess(float(jnp.max(health[:, index])), float(jnp.min(initial[:, index])))


if __name__ == '__main__':
  unittest.main(defaultTest='LayerStaticKeyZeroInitTest')

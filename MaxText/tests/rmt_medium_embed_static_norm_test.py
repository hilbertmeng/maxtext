"""Isolate static embedding normalization on the verified shared-content parent."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests import rmt_xlprop_test as helpers
from layers import rmt

BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedNorm'
EXP = BASE.removesuffix('SharedEmbedNorm') + 'SharedEmbedWriteNorm'


class EmbedStaticNormTest(unittest.TestCase):
  config = helpers.XLPropTest.config
  model_args = helpers.XLPropTest.model_args

  def test_target_budget_and_scope(self):
    cfg = self.config(EXP)
    old = self.config(BASE)
    allowed = {'model_name', 'exp_class', 'compare_runs', 'rmt_embedding_shared_write_norm'}
    changed = {k for k in cfg.get_keys() if cfg.get_keys()[k] != old.get_keys().get(k)}
    # Temporary output directories are harness-only overrides.
    changed.discard('base_output_directory')
    self.assertFalse(changed - allowed, changed)
    self.assertEqual(cfg.mlp_dim, 4100)
    self.assertEqual(cfg.rmt_dynamic_write_bottleneck_dim, 256)
    self.assertTrue(cfg.rmt_embedding_shared_write_norm)
    self.assertTrue(cfg.rmt_embedding_content_norm)
    self.assertTrue(cfg.rmt_layer_write_content_norm)
    self.assertFalse(cfg.get_keys().get('rmt_static_write_content_norm', False))
    self.assertTrue(cfg.scan_layers)
    self.assertFalse(cfg.rmt_block_scan)
    self.assertEqual(cfg.DATASET_VARIANT, 'truepile4096')
    for name in ('rmt_pallas_write', 'rmt_fused_attention_read', 'rmt_fused_write_mlp_read'):
      self.assertFalse(cfg.get_keys().get(name, False))
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)), 431888672)
    self.assertNotIn('embedding_write_content', tree['decoder'])

  def test_shared_normalized_write_equation(self):
    cfg = self.config(EXP)
    cfg.get_keys()['dtype'] = jnp.float32
    x = jax.random.normal(jax.random.key(2), (1, 2, cfg.emb_dim)) * .01
    y = x.reshape(1, 2, cfg.num_query_heads, cfg.head_dim)
    norm = lambda a: rmt.normalizations.rms_norm(a, dtype=a.dtype,
        epsilon=cfg.normalization_layer_epsilon, statistics_dtype=jnp.float32)
    module = rmt.RMTDynamicWrite(cfg, 48, name='dynamic_embedding_write')
    p = module.init(jax.random.key(3), x, norm(y), content_is_normalized=True)
    a, g = module.apply(p, x, norm(y), address_only=True, content_is_normalized=True)
    new_dynamic, _ = module.apply(p, x, norm(y), content_is_normalized=True)
    old_dynamic, _ = module.apply(p, x, y)
    np.testing.assert_allclose(new_dynamic, old_dynamic, rtol=2e-6, atol=2e-6)
    key = jax.random.normal(jax.random.key(4), (16, 48)) * .1
    got = jnp.einsum('btnv,nk->btkv', norm(y), key) + new_dynamic
    expected = jnp.einsum('btnk,btnv->btkv', key + g[..., None] * norm(a), norm(y))
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=3e-6)

  def test_scanned_gradient_and_embedding_health(self):
    cfg = self.config(EXP, base_num_decoder_layers=2, base_emb_dim=512,
                      head_dim=32, base_mlp_dim=128, vocab_size=128)
    cfg.get_keys()['dtype'] = jnp.float32
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p = nn.unbox(model.init(jax.random.key(5), **args)['params'])
      def loss(p):
        out, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
        return jnp.sum(out[0]), aux
      (value, aux), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(p)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value, aux, grad))))
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])), 0)
    health = aux['intermediates']['decoder']['rmt_embedding_health'][0]
    self.assertGreater(float(health[1]), .8)
    self.assertLess(float(health[1]), 1.2)


if __name__ == '__main__':
  unittest.main(defaultTest='EmbedStaticNormTest')

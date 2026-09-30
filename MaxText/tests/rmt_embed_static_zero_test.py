"""Focused zero-static embedding scope, budget and gradient checks."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests import rmt_xlprop_test as helpers

PREFIX = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm'
PARENT = PREFIX + 'Scale006'
EXP = PREFIX + 'Scale0'


class StaticZeroTest(unittest.TestCase):
  config = helpers.XLPropTest.config
  model_args = helpers.XLPropTest.model_args

  def test_full_budget_and_scope(self):
    cfg, old = self.config(EXP), self.config(PARENT)
    allowed = {'model_name', 'exp_class', 'compare_runs', 'rmt_embedding_static_write_scale',
               'base_output_directory', 'checkpoint_dir', 'metrics_dir', 'bucket_logging_dir'}
    changed = {k for k, v in cfg.get_keys().items() if v != old.get_keys().get(k)}
    self.assertFalse(changed - allowed, changed)
    self.assertEqual(cfg.rmt_embedding_static_write_scale, 0.)
    self.assertEqual(cfg.mlp_dim, 4100)
    self.assertTrue(cfg.scan_layers)
    self.assertFalse(cfg.rmt_block_scan)
    self.assertEqual(cfg.DATASET_VARIANT, 'truepile4096')
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)), 431888672)
    self.assertEqual(tree['decoder']['seed_key'].shape, (16, 48))

  def test_zero_static_preserves_dynamic_and_finite_gradient(self):
    cfgs = [self.config(name, base_num_decoder_layers=2, base_emb_dim=512,
                        head_dim=32, base_mlp_dim=128, vocab_size=128)
            for name in (PARENT, EXP)]
    for cfg in cfgs:
      cfg.get_keys()['dtype'] = jnp.float32
    parent, args = self.model_args(cfgs[0])
    new, _ = self.model_args(cfgs[1])
    with contextlib.redirect_stdout(io.StringIO()):
      p = nn.unbox(new.init(jax.random.key(2), **args)['params'])
      _, a = parent.apply({'params': p}, **args, mutable=['intermediates'])
      def loss(p):
        out, aux = new.apply({'params': p}, **args, mutable=['intermediates'])
        return jnp.sum(out[0]), aux
      (value, b), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(p)
    old_health = a['intermediates']['decoder']['rmt_embedding_health'][0]
    health = b['intermediates']['decoder']['rmt_embedding_health'][0]
    self.assertEqual(float(health[1]), 0.)
    np.testing.assert_allclose(health[0], old_health[0], rtol=2e-5)
    np.testing.assert_allclose(health[4:], old_health[4:], rtol=2e-5, atol=1e-7)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((value, b, grad))))
    np.testing.assert_array_equal(grad['decoder']['seed_key'], 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])), 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['token_embedder']['embedding'])), 0.)


if __name__ == '__main__':
  unittest.main(defaultTest='StaticZeroTest')

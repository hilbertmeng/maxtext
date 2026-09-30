"""Static seed scale changes no parameters or dynamic embedding write values."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests import rmt_xlprop_test as helpers

PARENT = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedEmbedWriteNorm'
EXP = PARENT + 'Scale006'


class StaticScaleTest(unittest.TestCase):
  config = helpers.XLPropTest.config
  model_args = helpers.XLPropTest.model_args

  def test_target_budget(self):
    cfg = self.config(EXP)
    self.assertEqual(cfg.mlp_dim, 4100)
    self.assertEqual(cfg.rmt_embedding_static_write_scale, .006)
    self.assertTrue(cfg.rmt_embedding_shared_write_norm)
    self.assertFalse(cfg.get_keys().get('rmt_static_write_content_norm', False))
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)), 431888672)
    self.assertNotIn('embedding_static_write_scale', tree['decoder'])

  def test_seed_scale_equation_and_gradient(self):
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
    old = a['intermediates']['decoder']['rmt_embedding_health'][0]
    new_health = b['intermediates']['decoder']['rmt_embedding_health'][0]
    np.testing.assert_allclose(new_health[0], old[0], rtol=2e-5)
    np.testing.assert_allclose(new_health[1], old[1] * .006, rtol=2e-5)
    np.testing.assert_allclose(new_health[4:], old[4:], rtol=2e-5, atol=1e-7)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((value, b, grad))))
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['seed_key'])), 0.)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])), 0.)


if __name__ == '__main__':
  unittest.main(defaultTest='StaticScaleTest')

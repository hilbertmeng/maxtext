"""Final-read zero initialization: unchanged budget and staged gradient opening."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'

class FinalReadZeroTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_budget_and_parent_equivalence(self):
    cfg = self.config(BASE + 'FinalReadZero')
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(p)), 431759072)
    self.assertEqual(p['decoder']['final_read_key'].shape, (48, 16))
    self.assertEqual(cfg.mlp_dim, 4100)
    self.assertTrue(cfg.rmt_dynamic_unembedding_direct_read)
    self.assertEqual(cfg.rmt_matrix_read_norm, 'none')

  def test_zero_output_and_gradient_unlock(self):
    configs = [self.config(n, base_num_decoder_layers=2, base_emb_dim=512,
                          head_dim=32, base_mlp_dim=128, vocab_size=128)
               for n in (BASE, BASE + 'FinalReadZero')]
    trees = []
    for cfg in configs:
      cfg.get_keys()['dtype'] = jnp.float32
      model, args = self.model_args(cfg)
      args['decoder_target_tokens'] = jnp.array([[2, 3, 4, 5]], jnp.int32)
      with contextlib.redirect_stdout(io.StringIO()):
        trees.append(nn.unbox(model.init(jax.random.key(7), **args)['params']))
    parent, params = trees
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
      keys = [k.key for k in path]
      old = parent
      for k in keys:
        old = old[k]
      if keys == ['decoder', 'final_read_key']:
        np.testing.assert_array_equal(value, 0.)
        self.assertGreater(float(jnp.linalg.norm(old)), 0.)
      else:
        np.testing.assert_array_equal(value, old)

    def loss(p):
      # Transformer returns token cross-entropies, not logits.
      out, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
      return jnp.mean(out[0]), aux

    with contextlib.redirect_stdout(io.StringIO()):
      vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
      (value, aux), grad = vg(params)
      self.assertAlmostEqual(float(value), math.log(128), places=5)
      for key in ('final_read_key',):
        self.assertGreater(float(jnp.linalg.norm(grad['decoder'][key])), 0.)
      self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_unembedding_read']['key_kernel'])), 0.)
      for scope in ('layers', 'dynamic_embedding_write', 'final_matrix_norm', 'lm_head'):
        for v in jax.tree.leaves(grad['decoder'][scope]):
          np.testing.assert_array_equal(v, 0.)
      self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value, aux, grad))))
      for i in range(2):
        params = jax.tree.map(lambda p, g: p - 1e-6 * g, params, grad)
        (value, aux), grad = vg(params)
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value, aux, grad))))
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])), 0.)
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['lm_head']['logits_dense'])), 0.)

if __name__ == '__main__':
  unittest.main()

"""Final-vector skip uses the final proxy, preserves parameters, and backpropagates."""
import contextlib
import io
import unittest
import math
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'

class FinalVectorSkipTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_budget(self):
    cfg = self.config(BASE + 'FinalVectorSkip')
    model,args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree = jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(tree)),431759072)
    self.assertTrue(cfg.rmt_unembedding_vector_skip)

  def test_exact_final_skip_and_gradients(self):
    configs = [self.config(BASE, base_num_decoder_layers=2, base_emb_dim=512,
                          head_dim=32, base_mlp_dim=128, vocab_size=128) for _ in range(2)]
    configs[1].get_keys()['rmt_unembedding_vector_skip'] = True
    trees, captures = [], []
    for cfg in configs:
      cfg.get_keys()['dtype'] = jnp.float32
      model, args = self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        params = nn.unbox(model.init(jax.random.key(13), **args)['params'])
      trees.append(params)
      captured = {}
      def interceptor(next_fun, args_, kwargs, ctx):
        if ctx.module.name == 'lm_head' and ctx.method_name == '__call__':
          captured['hidden'] = args_[0]
        out = next_fun(*args_, **kwargs)
        if ctx.module.name == 'unembedding_vector_norm' and ctx.method_name == '__call__':
          captured['proxy'] = out
        return out
      with contextlib.redirect_stdout(io.StringIO()), nn.intercept_methods(interceptor):
        model.apply({'params': params}, **args, mutable=['intermediates'])
      captures.append(captured)
    for a,b in zip(jax.tree.leaves(trees[0]),jax.tree.leaves(trees[1])):
      np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(captures[0]['proxy'], captures[1]['proxy'])
    np.testing.assert_allclose(captures[1]['hidden'],
                               captures[0]['hidden'] + captures[1]['proxy'],rtol=1e-6)
    def loss(p):
      out,aux = model.apply({'params':p}, **args, mutable=['intermediates'])
      return jnp.mean(out[0])
    with contextlib.redirect_stdout(io.StringIO()):
      value, grad = jax.jit(jax.value_and_grad(loss))(trees[1])
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,grad))))
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['unembedding_vector_norm']['scale'])),0.)

if __name__ == '__main__': unittest.main()

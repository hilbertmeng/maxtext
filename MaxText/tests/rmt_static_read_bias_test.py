"""Affine static reads: exact budget, parent equivalence and trainable biases."""
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
EXP = BASE + 'StaticQVMLPReadBias'
NAMES = ('static_q_read_bias', 'static_v_read_bias', 'static_mlp_read_bias')

class StaticReadBiasTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget(self):
    cfg = self.config(EXP)
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(jax.eval_shape(lambda k: model.init(k, **args)['params'], jax.random.key(1)))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(params)), 431759072 + 59616)
    layers = params['decoder']['layers']
    def scanned_shape(width):
      shape = [cfg.num_query_heads, width]
      shape.insert(cfg.param_scan_axis, cfg.num_decoder_layers)
      return tuple(shape)
    self.assertEqual(layers[NAMES[0]]['bias'].shape, scanned_shape(57))
    for name in NAMES[1:]:
      self.assertEqual(layers[name]['bias'].shape, scanned_shape(75))
    self.assertEqual(cfg.mlp_dim, 4100)
    self.assertFalse(cfg.get_keys().get('rmt_final_read_key_zero_init', False))

  def test_same_initial_function_and_bias_gradients(self):
    trees, models, inputs = [], [], []
    for name in (BASE, EXP):
      cfg = self.config(name, base_num_decoder_layers=2, base_emb_dim=512,
                        head_dim=32, base_mlp_dim=128, vocab_size=128)
      cfg.get_keys()['dtype'] = jnp.float32
      model, args = self.model_args(cfg)
      args['decoder_target_tokens'] = jnp.array([[2,3,4,5]], jnp.int32)
      with contextlib.redirect_stdout(io.StringIO()):
        params = nn.unbox(model.init(jax.random.key(7), **args)['params'])
      trees.append(params); models.append(model); inputs.append(args)
    parent, params = trees
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
      keys = [k.key for k in path]
      if any(n in keys for n in NAMES):
        np.testing.assert_array_equal(value, 0.)
      else:
        old = parent
        for k in keys: old = old[k]
        np.testing.assert_array_equal(value, old)
    def loss(p):
      out, aux = models[1].apply({'params':p}, **inputs[1], mutable=['intermediates'])
      return jnp.mean(out[0]), aux
    with contextlib.redirect_stdout(io.StringIO()):
      old, _ = models[0].apply({'params':parent}, **inputs[0], mutable=['intermediates'])
      vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
      (value, aux), grad = vg(params)
      np.testing.assert_allclose(value, jnp.mean(old[0]), rtol=1e-6, atol=1e-6)
      # Q bias initially has zero gradient because the matrix-derived K is zero.
      for name in NAMES[1:]:
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers'][name]['bias'])), 0.)
      for _ in range(2):
        params = jax.tree.map(lambda p,g:p-1e-6*g, params, grad)
        (value,aux),grad=vg(params)
      self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,aux,grad))))
      for name in NAMES:
        self.assertGreater(float(jnp.linalg.norm(grad['decoder']['layers'][name]['bias'])), 0.)

if __name__ == '__main__': unittest.main()

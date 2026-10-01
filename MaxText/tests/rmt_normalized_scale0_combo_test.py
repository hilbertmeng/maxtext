"""Focused checks for disabling only the combined model's static embedding write."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import tests.rmt_xlprop_test as helpers

EXP = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedScale0'
PARENT = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNormEmbedSeedZero'


class NormalizedScale0ComboTest(unittest.TestCase):
  config = helpers.XLPropTest.config
  model_args = helpers.XLPropTest.model_args

  def test_full_budget_and_only_two_effective_flag_changes(self):
    cfg = self.config(EXP)
    parent = self.config(PARENT)
    differences = {k for k in cfg.get_keys()
                   if repr(cfg.get_keys()[k]) != repr(parent.get_keys().get(k))}
    self.assertEqual(differences - {'model_name', 'exp_class', 'compare_runs',
                                   'base_output_directory', 'tensorboard_dir',
                                   'checkpoint_dir', 'metrics_dir', 'bucket_logging_dir'},
                     {'rmt_embedding_seed_key_zero_init', 'rmt_embedding_static_write_scale'})
    self.assertFalse(cfg.rmt_embedding_seed_key_zero_init)
    self.assertEqual(cfg.rmt_embedding_static_write_scale, 0.)
    self.assertEqual((cfg.checkpoint_period,cfg.keep_period,cfg.max_to_keep),(200,1000,2))
    self.assertEqual(cfg.DATASET_VARIANT, 'truepile4096')
    self.assertEqual(cfg.mlp_dim, 4100)
    model,args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    self.assertEqual(sum(math.prod(p.shape) for p in jax.tree.leaves(params)),431888672)
    self.assertEqual(params['decoder']['seed_key'].shape,(16,48))
    self.assertNotIn('embedding_write_content',params['decoder'])

  def test_static_seed_has_no_forward_or_gradient_effect(self):
    cfg = self.config(EXP,base_num_decoder_layers=2,base_emb_dim=512,
                      head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype'] = jnp.float32
    model,args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(2),**args)['params'])
      self.assertGreater(float(jnp.linalg.norm(params['decoder']['seed_key'])),0.)
      def loss(p):
        output,aux = model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(output[0]),(output,aux)
      (value,(output,aux)),grad = jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
      changed = dict(params)
      changed['decoder'] = dict(params['decoder'])
      changed['decoder']['seed_key'] = jnp.ones_like(params['decoder']['seed_key'])*100.
      altered = jax.jit(lambda p:model.apply({'params':p},**args))(changed)
    self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((value,aux,grad))))
    np.testing.assert_array_equal(grad['decoder']['seed_key'],0.)
    np.testing.assert_array_equal(aux['intermediates']['decoder']['rmt_embedding_health'][0][1],0.)
    for x,y in zip(jax.tree.leaves(output),jax.tree.leaves(altered)):
      np.testing.assert_allclose(x,y,rtol=2e-6,atol=2e-6)
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])),0.)


if __name__ == '__main__':
  unittest.main()

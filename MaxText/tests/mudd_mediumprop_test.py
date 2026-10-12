"""MediumProp Mudd: odd head width, independent histories and exact budget."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests import mudd_xlprop_test as shared

class MuddMediumPropTest(unittest.TestCase):
  cfg=shared.MuddXLPropTest.cfg
  model=shared.MuddXLPropTest.model
  check_history=shared.MuddXLPropTest.check_history

  def test_full_shape_budget_and_backbone(self):
    cfg=self.cfg(exp_class='MuddLlama2MediumPropTruePile')
    self.assertTrue(cfg.mudd_full_history)
    self.assertTrue(cfg.bam_mha_control)
    self.assertFalse(cfg.scan_layers)
    self.assertEqual(cfg.head_dim,75)
    self.assertEqual(cfg.bam_partial_rope_nope_dim,1)
    self.assertEqual(cfg.DATASET_VARIANT,'truepile4096')
    model,args=self.model(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
    self.check_history(tree,18)
    self.assertEqual(sum(math.prod(p.shape) for p in jax.tree.leaves(tree)),434076747)
    self.assertEqual(sum(round(3200*(i/17+.5)/128)*128 for i in range(18)),18*3200)
    dec=nn.unbox(tree)['decoder']
    for i in range(18):
      attention=dec[f'layers_{i}']['block']['self_attention']
      self.assertEqual(attention['key']['kernel'].shape,(1200,16,75))
      self.assertEqual(attention['value']['kernel'].shape,(1200,16,75))

  def test_odd_head_rope_history_and_all_gradients(self):
    cfg=self.cfg(exp_class='MuddLlama2MediumPropTruePile',base_num_decoder_layers=3,
      base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,head_dim=75,
      base_mlp_dim=256,vocab_size=128)
    model,args=self.model(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      self.check_history(p,3)
      loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.sum(model.apply({'params':p},**args)[0])))(p)
    self.assertTrue(np.isfinite(loss))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(g)))
    for i in range(3):
      block=g['decoder'][f'layers_{i}']['block']
      for route in ['query','key','value']:
        self.assertGreater(float(jnp.linalg.norm(block['self_attention'][route]['kernel'])),0.)
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(block['mlp'])),0.)
    again=model.apply({'params':p},**args)[0]
    np.testing.assert_allclose(jnp.sum(again),loss,rtol=2e-3)

if __name__=='__main__':unittest.main(defaultTest='MuddMediumPropTest')

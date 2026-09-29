"""Learned matrix read gains: budget, initial equivalence, gradients and health."""
import contextlib
import io
import math
import re
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest
from layers import rmt

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm'
NEW=BASE+'LearnedScale'

class MatrixScaleTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_parameter_budget_and_decay(self):
    cfg=self.config(NEW)
    self.assertEqual(cfg.base_mlp_dim,4078)
    self.assertEqual(cfg.rmt_matrix_read_norm,'all')
    self.assertTrue(cfg.rmt_vector_pre_norm)
    self.assertTrue(cfg.get_keys().get('rmt_layer_write_content_norm',True))
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
    self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)),431903072)
    for arm in ['attn','mlp']:
      self.assertEqual(nn.unbox(tree)['decoder']['layers'][arm+'_norm']['scale'].shape,(48,18,75))
      name='params/decoder/layers/'+arm+'_norm/scale'
      wd=cfg.adam_weight_decay
      for pat,value in cfg.wd_mults:
        if re.findall(pat,name):wd=value
      self.assertEqual(wd,0.)

  def test_gain_one_matches_parameter_free(self):
    cfg=self.config(NEW)
    m=jax.random.normal(jax.random.key(0),(1,3,48,75)).astype(jnp.bfloat16)
    module=rmt.MatrixRMSNorm(cfg)
    p=module.init(jax.random.key(1),m)
    np.testing.assert_array_equal(module.apply(p,m),rmt.matrix_read_rms_norm(m,cfg.normalization_layer_epsilon))

  def test_scanned_initial_equivalence_gradient_and_health(self):
    cfg=self.config(NEW,base_num_decoder_layers=2,base_emb_dim=512,head_dim=32,
                    base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),aux
      (v,aux),grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
      cfg.get_keys()['rmt_matrix_read_learned_scale']=False
      q=jax.tree.map(lambda x:x,p)
      for arm in ['attn','mlp']:del q['decoder']['layers'][arm+'_norm']
      expected=model.apply({'params':q},**args)[0]
      cfg.get_keys()['rmt_matrix_read_learned_scale']=True
    np.testing.assert_allclose(v,jnp.sum(expected),rtol=1e-6,atol=1e-6)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(grad)))
    from train import record_rmt_dynamic_health_metrics
    metrics={'scalar':{}}
    record_rmt_dynamic_health_metrics(metrics,aux,cfg)
    for arm in ['attn','mlp']:
      g=grad['decoder']['layers'][arm+'_norm']['scale']
      self.assertGreater(float(jnp.linalg.norm(g)),0.)
      stats=aux['intermediates']['decoder']['layers'][arm+'_norm']['scale_health'][0]
      np.testing.assert_array_equal(stats,jnp.tile(jnp.array([1.,0.,1.,1.,0.,0.]),(2,1)))
      self.assertEqual(float(metrics['scalar']['rmt/matrix_scale/layer_001/'+arm+'_mean']),1.)

if __name__=='__main__':unittest.main(defaultTest='MatrixScaleTest')

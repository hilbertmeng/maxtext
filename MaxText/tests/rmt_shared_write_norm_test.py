"""Shared normalized layer contents: scope, budget, equations and gradients."""
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

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScale'
NEW=BASE+'SharedWriteNorm'

class SharedWriteNormTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_budget_and_scope(self):
    cfg=self.config(NEW)
    self.assertEqual(cfg.base_mlp_dim,4078)
    self.assertTrue(cfg.rmt_static_write_content_norm)
    self.assertTrue(cfg.rmt_matrix_read_learned_scale)
    self.assertTrue(cfg.get_keys().get('rmt_layer_write_content_norm',True))
    self.assertFalse(cfg.get_keys().get('rmt_embedding_shared_content',False))
    self.assertFalse(cfg.rmt_block_scan)
    self.assertEqual(cfg.DATASET_VARIANT,'truepile4096')
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(p)),431903072)
    self.assertIn('embedding_write_content',p['decoder'])

  def test_layer_write_scale_invariance_and_embedding_unchanged(self):
    cfg=self.config(NEW)
    cfg.get_keys()['dtype']=jnp.float32
    x=jax.random.normal(jax.random.key(1),(1,2,cfg.emb_dim))
    y=jax.random.normal(jax.random.key(2),(1,2,16,75))*2
    key=jax.random.normal(jax.random.key(3),(16,48))*.1
    module=rmt.RMTDynamicWrite(cfg,48,name='dynamic_attn_write')
    p=module.init(jax.random.key(4),x,y)
    address,gate=module.apply(p,x,y,address_only=True)
    norm=lambda v:rmt.normalizations.rms_norm(v,dtype=v.dtype,epsilon=cfg.normalization_layer_epsilon)
    write=lambda y:rmt.static_layer_write(y,key,cfg)+module.apply(p,x,y)[0]
    expected=jnp.einsum('btnk,btnv->btkv',key+gate[...,None]*norm(address),norm(y))
    np.testing.assert_allclose(write(y),expected,atol=3e-6,rtol=3e-6)
    np.testing.assert_allclose(write(y*3),write(y),atol=3e-6,rtol=3e-6)
    g=jax.grad(lambda y:jnp.sum(write(y)**2))(y)
    self.assertTrue(np.isfinite(g).all())
    self.assertLess(abs(float(jnp.sum(g*y))),1e-3)
    embed=rmt.RMTDynamicWrite(cfg,48,name='dynamic_embedding_write')
    ep=embed.init(jax.random.key(5),x,y)
    before=embed.apply(ep,x,y)[0]
    cfg.get_keys()['rmt_static_write_content_norm']=False
    np.testing.assert_array_equal(before,embed.apply(ep,x,y)[0])
    np.testing.assert_allclose(rmt.static_layer_write(y,key,cfg),jnp.einsum('btnv,nk->btkv',y,key))

  def test_scanned_forward_all_gradients_and_health(self):
    cfg=self.config(NEW,base_num_decoder_layers=2,base_emb_dim=512,head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),aux
      (v,aux),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
    self.assertTrue(np.isfinite(v))
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((g,aux))))
    for arm in ['attn','mlp']:
      self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers'][arm+'_write_key'])),0.)
      self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers']['dynamic_'+arm+'_write']['address_up'])),0.)

if __name__=='__main__':unittest.main()

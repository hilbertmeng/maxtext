"""Shared raw embedding contents: parameter budget, write equation, gradients."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest
from layers import rmt

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormRawWrite'
NEW=BASE+'SharedEmbed'

class SharedEmbeddingTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_parameter_budget(self):
    for name,expected in [(BASE,431773472),(NEW,431759072)]:
      cfg=self.config(name)
      self.assertFalse(cfg.rmt_layer_write_content_norm)
      self.assertEqual(cfg.rmt_matrix_read_norm,'all')
      self.assertEqual(cfg.DATASET_VARIANT,'truepile4096')
      self.assertFalse(cfg.rmt_block_scan)
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
      self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)),expected)
      decoder=nn.unbox(tree)['decoder']
      self.assertEqual('embedding_write_content' in decoder,name==BASE)

  def test_shared_write_equation(self):
    cfg=self.config(NEW)
    cfg.get_keys()['dtype']=jnp.float32
    x=jax.random.normal(jax.random.key(1),(1,2,cfg.emb_dim))
    y=x.reshape(1,2,cfg.num_query_heads,cfg.head_dim)
    module=rmt.RMTDynamicWrite(cfg,48,name='dynamic_embedding_write')
    p=module.init(jax.random.key(2),x,y)
    address,g=module.apply(p,x,y,address_only=True)
    address=rmt.normalizations.rms_norm(address,dtype=address.dtype,
        epsilon=cfg.normalization_layer_epsilon,statistics_dtype=jnp.float32)
    actual,_=module.apply(p,x,y)
    np.testing.assert_allclose(actual,jnp.einsum('btnk,btnv->btkv',g[...,None]*address,y),rtol=2e-6,atol=2e-6)
    # Common raw content permits adding addresses before one contraction.
    static=jax.random.normal(jax.random.key(3),(cfg.num_query_heads,48))*.1
    expected=jnp.einsum('btnk,btnv->btkv',static+g[...,None]*address,y)
    np.testing.assert_allclose(expected,jnp.einsum('btnv,nk->btkv',y,static)+actual,rtol=2e-5,atol=2e-5)

  def test_scanned_forward_gradient_and_health(self):
    cfg=self.config(NEW,base_num_decoder_layers=2,base_emb_dim=512,
                    head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),aux
      (value,aux),grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
    self.assertTrue(np.isfinite(value))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(aux)))
    self.assertNotIn('embedding_write_content',grad['decoder'])
    self.assertGreater(float(jnp.linalg.norm(grad['decoder']['dynamic_embedding_write']['address_up'])),0.)

if __name__=='__main__': unittest.main(defaultTest='SharedEmbeddingTest')

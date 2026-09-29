"""Focused regression for unnormalized layer-write contents."""
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

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNorm'
NEW=BASE+'RawWrite'

class RawWriteTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_parameter_tree_and_formal_settings(self):
    trees=[]
    for name in [BASE,NEW]:
      cfg=self.config(name)
      assert cfg.rmt_matrix_read_norm=='all' and cfg.rmt_vector_pre_norm
      assert cfg.DATASET_VARIANT=='truepile4096' and not cfg.rmt_block_scan
      assert cfg.base_mlp_dim==4078 and cfg.rmt_record_dynamic_health
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
      self.assertEqual(sum(math.prod(x.shape) for x in jax.tree.leaves(tree)),431773472)
      trees.append(jax.tree.map(lambda x:(x.shape,str(x.dtype)),tree))
    self.assertEqual(trees[0],trees[1])

  def test_write_equations_and_embedding_unchanged(self):
    cfg=self.config(NEW)
    cfg.get_keys().update(dtype=jnp.float32)
    x=jax.random.normal(jax.random.key(1),(1,2,cfg.emb_dim))
    y=jax.random.normal(jax.random.key(2),(1,2,cfg.num_query_heads,cfg.head_dim))*.3
    for name in ['dynamic_attn_write','dynamic_mlp_write','dynamic_embedding_write']:
      module=rmt.RMTDynamicWrite(cfg,48,name=name)
      params=module.init(jax.random.key(3),x,y)
      address,gate=module.apply(params,x,y,address_only=True)
      address=rmt.normalizations.rms_norm(address,dtype=address.dtype,
          epsilon=cfg.normalization_layer_epsilon,statistics_dtype=jnp.float32)
      out,_=module.apply(params,x,y)
      scaled,_=module.apply(params,x,y*3)
      if name!='dynamic_embedding_write':
        expected=jnp.einsum('btnk,btnv->btkv',gate[...,None]*address,y)
        np.testing.assert_allclose(out,expected,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(scaled,out*3,rtol=2e-6,atol=2e-6)
      else:
        cfg.get_keys()['rmt_layer_write_content_norm']=True
        legacy,_=module.apply(params,x,y)
        np.testing.assert_array_equal(out,legacy)
        cfg.get_keys()['rmt_layer_write_content_norm']=False

  def test_scanned_forward_gradient_and_health(self):
    cfg=self.config(NEW,base_num_decoder_layers=2,base_emb_dim=512,head_dim=32,
                    base_mlp_dim=128,vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),aux
      (value,aux),grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    self.assertTrue(np.isfinite(value))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(aux)))
    for module in ['dynamic_embedding_write','dynamic_unembedding_read']:
      self.assertGreater(sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grad['decoder'][module])),0)

if __name__=='__main__':unittest.main(defaultTest='RawWriteTest')

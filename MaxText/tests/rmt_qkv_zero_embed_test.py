"""Focused QKV-zero embedding compositions: budget, effective seed and learning."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInit'
class QKVEmbedTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_budget_and_scopes(self):
    for suffix in ('EmbedSeedZero','EmbedScale0'):
      cfg=self.config(BASE+suffix)
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
      self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(p)),431759072)
      self.assertNotIn('embedding_write_content',p['decoder'])
      self.assertIn('final_matrix_norm',p['decoder'])
      self.assertEqual(cfg.mlp_dim,4100)
      self.assertEqual(cfg.rmt_matrix_read_norm,'none')
      for name in ('rmt_static_qk_zero_init','rmt_static_v_zero_init','rmt_vector_pre_norm',
                   'rmt_static_write_content_norm','rmt_embedding_shared_write_norm',
                   'rmt_embedding_shared_content','rmt_record_write_scale_health'):
        self.assertTrue(getattr(cfg,name),name)
      self.assertFalse(cfg.get_keys().get('rmt_block_scan',False))
      self.assertFalse(cfg.get_keys().get('rmt_pallas_fused_write',False))

  def test_effective_seed_forward_and_gradient(self):
    outputs=[]; grads=[]; trees=[]
    for suffix in ('EmbedSeedZero','EmbedScale0'):
      cfg=self.config(BASE+suffix,base_num_decoder_layers=2,base_emb_dim=512,
                      head_dim=32,base_mlp_dim=128,vocab_size=128)
      cfg.get_keys()['dtype']=jnp.float32
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        p=nn.unbox(model.init(jax.random.key(3),**args)['params'])
        def loss(params):
          out,aux=model.apply({'params':params},**args,mutable=['intermediates'])
          return -jnp.mean(jax.nn.log_softmax(out[0].astype(jnp.float32),axis=-1)[...,1]),(out[0],aux)
        vg=jax.jit(jax.value_and_grad(loss,has_aux=True))
        (value,(out,aux)),g=vg(p)
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,out,aux,g))))
        self.assertGreater(float(jnp.linalg.norm(g['decoder']['dynamic_embedding_write']['address_up'])),0.)
        updated=jax.tree.map(lambda v,d:v-1e-6*d,p,g)
        (value2,aux2),g2=vg(updated)
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value2,aux2,g2))))
        self.assertGreater(float(jnp.linalg.norm(g2['decoder']['layers']['qkv_key'][:2])),0.)
      trees.append(p);outputs.append(out);grads.append(g)
    np.testing.assert_allclose(outputs[0],outputs[1],rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(trees[0]['decoder']['seed_key'],0.)
    self.assertGreater(float(jnp.linalg.norm(trees[1]['decoder']['seed_key'])),0.)
    self.assertGreater(float(jnp.linalg.norm(grads[0]['decoder']['seed_key'])),0.)
    np.testing.assert_array_equal(grads[1]['decoder']['seed_key'],0.)

if __name__=='__main__':unittest.main()

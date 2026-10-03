"""Zero only static QK; preserve V, embedding, VectorNorm and normalized writes."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

BASE = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOMPreNormLearnedScaleSharedWriteNorm'
EXP = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit'

class QKZeroNoMNormTest(unittest.TestCase):
  config = XLPropTest.config
  model_args = XLPropTest.model_args

  def test_full_budget_and_scope(self):
    old, cfg = self.config(BASE), self.config(EXP)
    allowed = {'model_name','exp_class','compare_runs','rmt_static_qk_zero_init',
               'rmt_matrix_read_norm','rmt_matrix_read_learned_scale','rmt_record_write_scale_health',
               'keep_period','max_to_keep','base_output_directory','checkpoint_dir','metrics_dir','bucket_logging_dir'}
    changed = {k for k,v in cfg.get_keys().items() if v != old.get_keys().get(k)}
    self.assertFalse(changed-allowed, changed-allowed)
    self.assertEqual(cfg.mlp_dim,4078)
    self.assertTrue(cfg.rmt_vector_pre_norm)
    self.assertTrue(cfg.rmt_static_write_content_norm)
    self.assertFalse(cfg.get_keys().get('rmt_probe_matrix_pre_norm',False))
    self.assertFalse(cfg.get_keys().get('rmt_embedding_shared_content',False))
    self.assertTrue(cfg.scan_layers)
    self.assertFalse(cfg.rmt_block_scan)
    self.assertEqual(cfg.DATASET_VARIANT,'truepile4096')
    self.assertFalse(any(v for k,v in cfg.get_keys().items() if k.startswith('rmt_pallas')))
    counts=[]
    for c in (old,cfg):
      model,args=self.model_args(c)
      with contextlib.redirect_stdout(io.StringIO()):
        tree=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
      counts.append(sum(math.prod(v.shape) for v in jax.tree.leaves(tree)))
    self.assertEqual(counts,[431903072,431773472])
    self.assertNotIn('attn_norm',tree['decoder']['layers'])
    self.assertNotIn('mlp_norm',tree['decoder']['layers'])
    self.assertIn('attn_vector_norm',tree['decoder']['layers'])
    self.assertIn('final_matrix_norm',tree['decoder'])

  def test_forward_gradients_and_only_qk_initializer_changes(self):
    cfg=self.config(EXP,base_num_decoder_layers=2,base_emb_dim=512,head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      cfg.get_keys()['rmt_static_qk_zero_init']=False
      reference=nn.unbox(model.init(jax.random.key(2),**args)['params'])
      cfg.get_keys()['rmt_static_qk_zero_init']=True
      for (path,v),(path2,w) in zip(jax.tree_util.tree_flatten_with_path(params)[0],jax.tree_util.tree_flatten_with_path(reference)[0]):
        self.assertEqual(path,path2)
        if path[-1]==jax.tree_util.DictKey('qkv_key'):
          # Scan layer axis is inserted at axis1: (3,L,H,K).
          np.testing.assert_array_equal(v[:2],0)
          np.testing.assert_array_equal(v[2],w[2])
          self.assertGreater(float(jnp.linalg.norm(v[2])),0.)
        else: np.testing.assert_array_equal(v,w)
      reference['decoder']['layers']['qkv_key']=reference['decoder']['layers']['qkv_key'].at[:2].set(0)
      expected=model.apply({'params':reference},**args)[0]
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return -jnp.mean(jax.nn.log_softmax(out[0].astype(jnp.float32),axis=-1)[...,1]),(out[0],aux)
      (value,(out,aux)),grads=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    np.testing.assert_allclose(out,expected,rtol=2e-5,atol=2e-5)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,aux,grads))))
    for arm in range(2):
      self.assertGreater(float(jnp.linalg.norm(grads['decoder']['layers']['qkv_key'][arm])),0.)
    self.assertGreater(float(jnp.linalg.norm(grads['decoder']['dynamic_embedding_write']['address_up'])),0.)

if __name__=='__main__':unittest.main(defaultTest='QKZeroNoMNormTest')

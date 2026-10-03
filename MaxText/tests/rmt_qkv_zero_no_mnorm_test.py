"""QKV-zero is QK-zero with only static V initialization changed."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest

BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKZeroInit'
EXP=BASE.replace('QKZeroInit','QKVZeroInit')
class QKVZeroTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_scope_budget(self):
    old,cfg=self.config(BASE),self.config(EXP)
    changed={k for k,v in cfg.get_keys().items() if v!=old.get_keys().get(k)}
    self.assertFalse(changed-{'model_name','exp_class','compare_runs','rmt_static_v_zero_init','base_output_directory','checkpoint_dir','metrics_dir','bucket_logging_dir'})
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    self.assertEqual(sum(math.prod(v.shape) for v in jax.tree.leaves(tree)),431773472)
    self.assertEqual(cfg.mlp_dim,4078)
    self.assertEqual(cfg.rmt_matrix_read_norm,'none')
    self.assertTrue(cfg.rmt_vector_pre_norm and cfg.rmt_static_write_content_norm)
    self.assertIn('final_matrix_norm',tree['decoder'])

  def test_exact_initializer_delta_forward_gradient(self):
    cfg=self.config(EXP,base_num_decoder_layers=2,base_emb_dim=512,head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(3),**args)['params'])
      cfg.get_keys()['rmt_static_v_zero_init']=False
      old=nn.unbox(model.init(jax.random.key(3),**args)['params'])
      cfg.get_keys()['rmt_static_v_zero_init']=True
      for (path,v),(path2,w) in zip(jax.tree_util.tree_flatten_with_path(p)[0],jax.tree_util.tree_flatten_with_path(old)[0]):
        self.assertEqual(path,path2)
        if path[-1]==jax.tree_util.DictKey('qkv_key'):
          np.testing.assert_array_equal(v,0)
          np.testing.assert_array_equal(w[:2],0)
          self.assertGreater(float(jnp.linalg.norm(w[2])),0.)
        else:np.testing.assert_array_equal(v,w)
      old['decoder']['layers']['qkv_key']=jnp.zeros_like(old['decoder']['layers']['qkv_key'])
      expected=model.apply({'params':old},**args)[0]
      def loss(params):
        out,aux=model.apply({'params':params},**args,mutable=['intermediates'])
        return -jnp.mean(jax.nn.log_softmax(out[0].astype(jnp.float32),axis=-1)[...,1]),(out[0],aux)
      (value,(out,aux)),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
    np.testing.assert_allclose(out,expected,rtol=2e-5,atol=2e-5)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,aux,g))))
    for arm in range(3):
      for layer in range(2):
        self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers']['qkv_key'][arm,layer])),0.)
    self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers']['dynamic_vo']['key_kernel'])),0.)

if __name__=='__main__':unittest.main(defaultTest='QKVZeroTest')

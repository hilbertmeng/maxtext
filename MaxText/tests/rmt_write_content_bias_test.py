"""Affine write content: zero-init parity, per-layer budget, trainable effect."""
import contextlib, io, math, unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest
BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZeroStaticQVMLPReadBias'
NEW=BASE+'WriteContentBias'
class WriteBiasTest(unittest.TestCase):
 config=XLPropTest.config
 model_args=XLPropTest.model_args
 def test_budget(self):
  counts=[]
  for name in [BASE,NEW]:
   cfg=self.config(name);model,args=self.model_args(cfg)
   with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(0)))
   counts.append(sum(math.prod(x.shape) for x in jax.tree.leaves(p)))
  self.assertEqual(counts[1]-counts[0],43200);print('PARAMS',counts)
 def test_zero_parity_and_gradient(self):
  cfg=self.config(NEW,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128)
  model,args=self.model_args(cfg)
  with contextlib.redirect_stdout(io.StringIO()):p=nn.unbox(model.init(jax.random.key(4),**args)['params'])
  lp=p['decoder']['layers']
  for key in ['attn_write_content_bias','mlp_write_content_bias']:np.testing.assert_array_equal(lp[key],0)
  def loss(p):return jnp.mean(model.apply({'params':p},**args)[0])
  with contextlib.redirect_stdout(io.StringIO()):v,g=jax.jit(jax.value_and_grad(loss))(p)
  self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((v,g))))
  for key in ['attn_write_content_bias','mlp_write_content_bias']:self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers'][key])),0)
  oldcfg=self.config(BASE,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128)
  old,args=self.model_args(oldcfg)
  import copy
  oldp=copy.deepcopy(p)
  for key in ['attn_write_content_bias','mlp_write_content_bias']:del oldp['decoder']['layers'][key]
  with contextlib.redirect_stdout(io.StringIO()):
   oldv=jnp.mean(old.apply({'params':oldp},**args)[0])
   newv=loss(p)
  np.testing.assert_array_equal(newv,oldv)
if __name__=='__main__':unittest.main()

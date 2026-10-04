"""Targeted MLP input norm/raw write contrast with unaffected attention/embedding."""
import contextlib,io,math,unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import rmt
from tests.rmt_xlprop_test import XLPropTest
BASE='RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoOSharedWriteNormQKVZeroInitEmbedSeedZero'
A=BASE+'MLPInputPreNorm';B=A+'SharedRawWrite'
class MLPInputNormTest(unittest.TestCase):
 config=XLPropTest.config
 model_args=XLPropTest.model_args
 def test_full_budget(self):
  for name in [A,B]:
   cfg=self.config(name);model,args=self.model_args(cfg)
   with contextlib.redirect_stdout(io.StringIO()):
    p=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(0)))
   count=sum(math.prod(x.shape) for x in jax.tree.leaves(p))
   self.assertEqual(count,431759072+18*1200)
   self.assertFalse(cfg.get_keys().get('rmt_block_scan',False));print(name,count)
 def test_content_scaling(self):
  x=jax.random.normal(jax.random.key(1),(1,4,512));y=jax.random.normal(jax.random.key(2),(1,4,16,32));key=jnp.ones((16,48))
  for name,linear in [(A,False),(B,True)]:
   cfg=self.config(name,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128);cfg.get_keys()['dtype']=jnp.float32
   for arm in ['dynamic_mlp_write','dynamic_attn_write','dynamic_embedding_write']:
    module=rmt.RMTDynamicWrite(cfg,48,name=arm)
    with contextlib.redirect_stdout(io.StringIO()):p=module.init(jax.random.key(3),x,y)
    a=module.apply(p,x,y)[0];b=module.apply(p,x,y*3)[0]
    np.testing.assert_allclose(b,a*(3 if linear and arm=='dynamic_mlp_write' else 1),rtol=2e-4,atol=2e-4)
   a=rmt.static_layer_write(y,key,cfg,raw_content=linear);b=rmt.static_layer_write(3*y,key,cfg,raw_content=linear)
   np.testing.assert_allclose(b,a*(3 if linear else 1),rtol=2e-4,atol=2e-4)
 def test_scan_gradient_and_input(self):
  for name in [A,B]:
   cfg=self.config(name,base_emb_dim=512,head_dim=32,base_mlp_dim=96,base_num_decoder_layers=2,vocab_size=128);cfg.get_keys()['dtype']=jnp.float32
   model,args=self.model_args(cfg)
   with contextlib.redirect_stdout(io.StringIO()):
    p=nn.unbox(model.init(jax.random.key(4),**args)['params'])
    def loss(p):
     y,aux=model.apply({'params':p},**args,mutable=['intermediates']);return jnp.mean(y[0]),aux
    (v,aux),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
   self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((v,aux,g))))
   from train import record_rmt_dynamic_health_metrics
   metrics={'scalar':{}};record_rmt_dynamic_health_metrics(metrics,aux,cfg)
   for l in range(2):self.assertAlmostEqual(float(metrics['scalar'][f'rmt/mlp_input/layer_{l:03d}/actual_input_rms']),1.,delta=.01)
   self.assertGreater(float(jnp.linalg.norm(g['decoder']['layers']['mlp_input_norm']['scale'])),0.)
if __name__=='__main__':unittest.main()

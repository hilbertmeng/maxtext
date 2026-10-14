"""XL MHA formal budget and actual standard-attention norm gradients."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer
NEW='Llama2XLPropTruePileMHAQKNorm'
class XLMHAQKNormTest(MLPWriteTest,unittest.TestCase):
 def mesh(self,c):return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
 def test_formal_budget(self):
  totals=[]
  for exp in ['Llama2XLPropTruePileMHA',NEW]:
   c=self.config(exp);args,*_=train_compile.get_shaped_inputs(self.mesh(c),c);f=flatten_dict(nn.unbox(args[0].params));total=sum(int(np.prod(v.shape)) for v in f.values());totals.append(total)
   if exp==NEW:
    norms=[v for p,v in f.items() if 'qk_norm' in p];self.assertEqual(sum(int(np.prod(v.shape)) for v in norms),5376);self.assertTrue(all(96 in v.shape for v in norms));self.assertEqual(c.learning_rate_schedule_steps,50000);self.assertEqual(c.checkpoint_period,250);self.assertEqual(c.tensorboard_gcs_rotate_period,500);self.assertFalse(c.record_training_health_metrics)
   print('XL_MHA_QKNORM_PARAMS',exp,total,flush=True)
  self.assertEqual(totals,[1432398720,1432404096])
 def test_standard_attention_scan_gradients_causality_and_health(self):
  c=self.config(NEW,attention='dot_product',base_emb_dim=192,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32');mesh=self.mesh(c);m=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   p=m.init(jax.random.key(42),tok,pos,tok,mask,mask,enable_dropout=False)['params']
   def f(p,t):return m.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False)[0]
   loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,tok)**2)))(p);flat=flatten_dict(nn.unbox(g));self.assertTrue(np.isfinite(float(loss)));self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
   for name in ['q_norm','k_norm']:self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if name in path),0)
   np.testing.assert_allclose(f(p,tok)[:,:3],f(p,tok.at[:,3].set(5))[:,:3],atol=1e-6,rtol=1e-6)
if __name__=='__main__':unittest.main()

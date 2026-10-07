"""Focused full-budget and consumed-gradient gates for Prop compressed LocalQK."""
import functools,os,unittest
from unittest import mock
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train,train_compile
from layers import attentions,quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest
XL=os.environ.get('PROP_DIRECT_QK_SCALE')=='xl'
PARENT=('BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile' if XL else 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile')
EXP=PARENT.replace('TruePile','DirectC10TruePile' if XL else 'DirectC8TruePile')

class PropDirectQKTest(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_full_budget_and_key_shapes(self):
  counts=(1432440120,1432381880) if XL else (432096128,432093824)
  for exp,count in zip((PARENT,EXP),counts):
   c=self.config(exp);self.assertEqual(c.DATASET_VARIANT,'truepile4096');self.assertFalse(c.qk_norm)
   self.assertEqual((c.bam_k,c.bam_v,c.bam_abs_v_compression_dim,c.bam_local_qk_col_output_dim),(96,40,10,72) if XL else (75,32,8,57))
   self.assertEqual(c.bam_mlp_write_address_rank,384 if XL else 256)
   self.assertFalse(getattr(c,'bam_local_vo_separate_last_block',False))
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   actual=sum(int(np.prod(v.shape)) for v in flat.values());self.assertEqual(actual,count,exp)
   keys=[(p,v.shape) for p,v in flat.items() if 'W_lq_c8' in p or 'W_lk_c8' in p]
   if exp==EXP:
    self.assertEqual(len(keys),8 if XL else 6)
    for p,shape in keys:self.assertIn((1920 if XL else 1200),shape);self.assertEqual(shape[-2:],(20,10) if XL else (16,8));self.assertEqual(p[-1],'kernel')
    self.assertFalse(any('W_lq_bias' in p or 'W_local_packed' in p for p in flat))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
    for l in range(c.num_decoder_layers):
     self.assertIn(f'bam/concat/local_q_gate/layer_{l:03d}/mean',metrics)
     self.assertIn(f'bam/concat/local_k_gate/layer_{l:03d}/mean',metrics)
   else:self.assertFalse(keys)
   print('PROP_DIRECT_FULL_BUDGET_OK',exp,actual,flush=True)
 def test_forward_and_consumed_gradients(self):
  c=self.config(EXP,dtype='float32',weight_dtype='float32');hd=96 if XL else 75;nl=7 if XL else 6
  c.get_keys().update(base_emb_dim=hd*2,emb_dim=hd*2,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=nl,num_decoder_layers=nl,base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=[64]*3,bam_final_local_mlp_dim=64,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*nl,bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tokens=jnp.array([[1,2,3,4]],jnp.int32);args=(tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens));seen=[];original=attentions._attention_op
  def checked(q,k,v,*a,**kw):
   self.assertEqual((q.shape[-1],k.shape[-1],v.shape[-1]),(hd,hd,hd));seen.append(1);return original(q,k,v,*a,**kw)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(attentions,'_attention_op',checked):
   params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*args,enable_dropout=False)['params']
   def loss(p):return jnp.mean(model.apply({'params':p},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
   value,grad=jax.jit(jax.value_and_grad(loss))(params)
  self.assertTrue(seen);self.assertTrue(np.isfinite(float(value)));self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad)));flat=flatten_dict(nn.unbox(grad))
  for name in ('W_lq_c8','W_lk_c8','W_lq_gate','W_lk_gate','static_q_key','static_k_key','mlp_address_down','mlp_address_up'):
   vals=[v for p,v in flat.items() if name in p];self.assertTrue(vals,name);self.assertGreater(sum(float(jnp.sum(v*v)) for v in vals),0.,name)
  print('PROP_DIRECT_FINITE_CONSUMED_GRADIENTS_OK',EXP,float(value),flush=True)
 def test_legacy_c8_regression(self):
  from bam_attention_test import BamReadKeyTransformTest
  case=BamReadKeyTransformTest('test_direct_c8_qk_independent_keys_static_and_gradients');case.test_direct_c8_qk_independent_keys_static_and_gradients()

if __name__=='__main__':unittest.main()

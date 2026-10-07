"""Focused merge ordering, unchanged full budget, and scanned consumed gradients."""
import functools, unittest
from unittest import mock
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions, quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP='BamMediumPropK75EmbedVOnlyQK75AddBeforeRoPEAllLocalMLPWriteIndependentEveryThirdTruePile'

class MergeProbe(attentions.BamAttention):
 @nn.compact
 def __call__(self,q,k,qm,km,pos,reference=False):
  if not reference:return self._add_local_qk(q,k,qm,km,positions=pos)
  def one(s,m,name):
   tail=self.apply_rotary_embedding(m[...,57:]+s,pos,name=name,embedding_dims=18)
   return jnp.concatenate((m[...,:57],tail),axis=-1)
  return one(q,qm,'query_rotary'),one(k,km,'key_rotary')

class AddBeforeRoPETest(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_full_budget_and_health(self):
  shapes=[]
  for exp in (PARENT,EXP):
   c=self.config(exp);self.assertFalse(c.qk_norm);self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901]);self.assertEqual(c.DATASET_VARIANT,'truepile4096')
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432096128)
   shapes.append({p:v.shape for p,v in flat.items()})
   if exp==EXP:
    self.assertTrue(c.bam_local_qk_add_before_rope);self.assertEqual(c.bam_partial_rope_nope_dim,57);self.assertEqual(c.bam_local_qk_col_output_dim,75)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
    self.assertIn('bam/concat/matrix_qk_scores/layer_000/nope_over_rope',metrics)
   print('FULL_BUDGET_OK',exp,flush=True)
  self.assertEqual(*shapes)
 def test_merge_then_rotate_and_parent_regression(self):
  c=self.config(EXP,dtype='float32',weight_dtype='float32');c.get_keys()['bam_record_concat_health']=False
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  a=MergeProbe(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
  q,k=[jax.random.normal(jax.random.key(i),(1,4,2,18)) for i in (1,2)]
  qm,km=[jax.random.normal(jax.random.key(i),(1,4,2,75)) for i in (3,4)];pos=jnp.array([[0,2,5,9]])
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   params=a.init(jax.random.key(5),q,k,qm,km,pos)
   result=a.apply(params,q,k,qm,km,pos)
   references=a.apply(params,q,k,qm,km,pos,reference=True)
   for actual,ref,m in zip(result,references,(qm,km)):
    np.testing.assert_allclose(actual,ref,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(actual[...,:57],m[...,:57])
   c.get_keys()['bam_local_qk_add_before_rope']=False
   legacy=a.apply(params,q,k,qm,km,pos)
   np.testing.assert_array_equal(legacy[0],jnp.concatenate((qm,q),axis=-1));np.testing.assert_array_equal(legacy[1],jnp.concatenate((km,k),axis=-1))
  print('MERGE_ORDER_LEGACY_REGRESSION_OK',flush=True)
 def test_scanned_forward_and_consumed_gradients(self):
  c=self.config(EXP,dtype='float32',weight_dtype='float32');c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=6,num_decoder_layers=6,base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=[64]*3,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*6,bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tokens=jnp.array([[1,2,3,4]],jnp.int32);args=(tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens));seen=[];rotations=[];original=attentions._attention_op;rotate=attentions.BamAttention.apply_rotary_embedding
  def checked(q,k,v,*a,**kw):
   self.assertEqual((q.shape[-1],k.shape[-1],v.shape[-1]),(75,75,75));seen.append(1);return original(q,k,v,*a,**kw)
  def checked_rotate(mod,x,*a,**kw):rotations.append(x.shape[-1]);return rotate(mod,x,*a,**kw)
  merge=attentions.BamAttention._add_local_qk
  def checked_merge(mod,*a,**kw):
   before=len(rotations);result=merge(mod,*a,**kw);self.assertEqual(len(rotations)-before,2);return result
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(attentions,'_attention_op',checked),mock.patch.object(attentions.BamAttention,'apply_rotary_embedding',checked_rotate),mock.patch.object(attentions.BamAttention,'_add_local_qk',checked_merge):
   params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*args,enable_dropout=False)['params']
   def loss(p):return jnp.mean(model.apply({'params':p},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
   value,grad=jax.jit(jax.value_and_grad(loss))(params)
  self.assertTrue(seen);self.assertTrue(rotations);self.assertEqual(set(rotations),{18});self.assertTrue(np.isfinite(float(value)));self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad)))
  flat=flatten_dict(nn.unbox(grad))
  for name in ('query','key','static_q_key','static_k_key','W_local_packed','mlp_address_up'):
   vals=[v for p,v in flat.items() if name in p];self.assertTrue(vals,name);self.assertGreater(sum(float(jnp.sum(v*v)) for v in vals),0.,name)
  print('SCANNED_FINITE_CONSUMED_GRADIENTS_ONCE_ROPE_OK',float(value),flush=True)

if __name__=='__main__':unittest.main()

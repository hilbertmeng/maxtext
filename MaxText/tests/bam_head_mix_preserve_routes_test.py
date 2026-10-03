"""Strict H/gate replacement retains summed LocalO writes and old-M reads."""
import functools
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions
from bam_mlp_write_test import MLPWriteTest,PREFIX
import unittest

BASE=PREFIX+'StaticEveryThirdTruePile'
EXP=PREFIX+'StaticEveryThirdHeadMixPreserveRoutesTruePile'
class PreserveRoutesTest(MLPWriteTest):
 def test_budget_health(self):
  c=self.config(EXP);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
  flat=flatten_dict(args[0].params)
  self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432128192)
  self.assertEqual(c.mlp_dim_by_block,[4296,4291,4296])
  self.assertFalse(c.bam_local_vo_separate_c8_keys)
  self.assertFalse(c.bam_local_o_post_write)
  self.assertFalse(c.bam_local_o_updated_query)
  self.assertTrue(c.bam_mlp_write_static_address)
  self.assertFalse(any('self_attention' in p and 'out' in p for p in flat))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
  for i in range(18):self.assertIn(f'bam/concat/output_head_gate/layer_{i:03d}/mean',metrics)

 def test_parent_memory_equal_and_summed_output_projection(self):
  modules=[];params=[]
  x=jax.random.normal(jax.random.key(10),(1,4,150));M=jax.random.normal(jax.random.key(11),(1,4,75,32));pos=jnp.arange(4)[None];mask=jnp.ones((1,4),jnp.int32)
  for exp in (BASE,EXP):
   c=self.config(exp,dtype='float32',weight_dtype='float32');c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,bam_write_v_bottleneck_dim=16,bam_record_concat_health=False,bam_record_write_health=False,bam_record_address_health=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):p=nn.unbox(a.init(jax.random.key(12),x,x,pos,mask,M_in=M,deterministic=True)['params'])
   modules.append(a);params.append(p)
  # Use an identity output in the parent solely to expose its original summed head output.
  p0,p1=params
  for k,v in p0.items():
   if k!='out':
    for a,b in zip(jax.tree.leaves(v),jax.tree.leaves(p1[k])):np.testing.assert_array_equal(a,b)
  p0['out']['kernel']=jnp.eye(150).reshape(2,75,150)
  # Nonzero LocalO makes the route assertion sensitive to bypassing LocalO.
  for p in params:
   for k in p:
    if 'static' in k and 'o' in k.lower():
     p[k]=jax.tree.map(lambda v:jnp.ones_like(v)*.03,p[k])
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   old,mold=modules[0].apply({'params':p0},x,x,pos,mask,M_in=M,deterministic=True)
   new,mnew=modules[1].apply({'params':p1},x,x,pos,mask,M_in=M,deterministic=True)
   gate=1.1*jax.nn.sigmoid(jnp.einsum('btd,dn->btn',x,p1['output_head_gate']['kernel'])+p1['output_head_gate_bias'])
   expected=(jnp.einsum('btnk,nm->btmk',old.reshape(1,4,2,75),p1['output_head_mix'])*gate[...,None]).reshape(old.shape)
   np.testing.assert_allclose(new,expected,rtol=2e-5,atol=2e-5)
   np.testing.assert_array_equal(mnew,mold)
   d0=modules[0].apply({'params':p0},x,x,pos,mask,M_in=M,deterministic=True,defer_write=True)
   d1=modules[1].apply({'params':p1},x,x,pos,mask,M_in=M,deterministic=True,defer_write=True)
   for u,v in zip(d0[2],d1[2]):np.testing.assert_array_equal(u,v)
   np.testing.assert_allclose(d1[0],expected,rtol=2e-5,atol=2e-5)
   def loss(p):
    o,m=modules[1].apply({'params':p},x,x,pos,mask,M_in=M,deterministic=True)
    return jnp.mean(o**2)+jnp.mean(m**2)
   v,g=jax.jit(jax.value_and_grad(loss))(p1)
   self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves((v,g))))
   self.assertGreater(float(jnp.linalg.norm(g['output_head_mix'])),0.)
   self.assertGreater(float(jnp.linalg.norm(g['output_head_gate']['kernel'])),0.)

if __name__=='__main__':unittest.main()

"""Updated LocalO query: shared MLP norm, read timing, and untouched raw writes."""
import functools
from unittest import mock
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions, fusion, linears, normalizations
from bam_mlp_write_test import MLPWriteTest, PREFIX

POST=PREFIX+'IndependentEveryThirdPostWriteLocalORawWOTruePile'
PRE=PREFIX+'IndependentEveryThirdPreWriteLocalORawWOTruePile'
ARMS=[POST.replace('TruePile','UpdatedQueryTruePile'),PRE.replace('TruePile','UpdatedQueryTruePile')]

class Capture(attentions.BamAttention):
 def _query_chunk_op(self,*args,**kw):
  result=super()._query_chunk_op(*args,**kw)
  self.sow('probe','raw',result[0])
  return result
 def _read_fetched_m(self,m,x,*args,**kw):
  self.sow('probe','o_query',x)
  result=super()._read_fetched_m(m,x,*args,**kw)
  return result
 def _deferred_write_factors(self,y,x,*args,**kw):
  self.sow('probe','write_query',x)
  result=super()._deferred_write_factors(y,x,*args,**kw)
  self.sow('probe','write_content',result[0]);self.sow('probe','write_address',result[1])
  return result
class CaptureMLP(linears.MlpBlock):
 def __call__(self,x,*args,**kw):
  self.sow('probe','input',x)
  return super().__call__(x,*args,**kw)

class UpdatedQueryTest(MLPWriteTest):
 def test_full_budget_and_health(self):
  for exp in ARMS:
   c=self.config(exp)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
   flat=flatten_dict(args[0].params)
   self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432139328)
   norms=[v for p,v in flat.items() if 'post_self_attention_layer_norm' in p and p[-1]=='scale']
   self.assertEqual(sum(int(np.prod(v.shape)) for v in norms),18*1200)
   self.assertEqual(c.mlp_dim_by_block,[3859,3732,3859])
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
   for i in range(18):
    self.assertIn(f'bam/concat/local_o_query_change_amplitude/layer_{i:03d}/bam_over_standard',metrics)
    self.assertIn(f'bam/concat/local_o_gate/layer_{i:03d}/mean',metrics)
   print('UPDATED_QUERY_FULL_BUDGET_HEALTH_OK',exp,flush=True)

 def test_fusion_norm_routing_and_gradient(self):
  for exp in ARMS:
   c=self.config(exp,dtype='float32',weight_dtype='float32')
   c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,bam_write_v_bottleneck_dim=16,
     mlp_dim_by_block=[128]*3,bam_record_concat_health=False,bam_record_write_health=False,
     bam_record_address_health=False,bam_lambda_decay=.73)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   x=jax.random.normal(jax.random.key(1),(1,4,150))*2
   m0=jax.random.normal(jax.random.key(2),(1,4,75,32))
   tokens=jnp.ones((1,4),jnp.int32)
   call=(x,tokens,jnp.arange(4)[None],tokens,None,True,'train',tokens)
   for layer in [0,1]:
    a=fusion.SubDecoderLayer(c,mesh,layer_inx=layer,sliding_window_size=4)
    with mock.patch.object(attentions,'BamAttention',Capture),mock.patch.object(linears,'MlpBlock',CaptureMLP),mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
     p=nn.unbox(a.init(jax.random.key(3),*call,M_in=m0)['params'])
     p['post_self_attention_layer_norm']['scale']=jnp.linspace(.2,.7,150)
     p['pre_self_attention_layer_norm']['scale']=jnp.linspace(.1,.3,150)
     (out,mout),probes=a.apply({'params':p},*call,M_in=m0,mutable=['probe','intermediates'],capture_intermediates=lambda mod,name:isinstance(mod,normalizations.RMSNorm) and name=='__call__')
     pr=probes['probe']['self_attention'];raw=pr['raw'][0]
     projected=jnp.einsum('btnk,nkd->btd',raw,p['self_attention']['out']['kernel'])
     def norm(z,name):
      scale=p[name]['scale']+(0 if c.direct_scale else 1)
      return normalizations.rms_norm(z,dtype=c.dtype,epsilon=c.normalization_layer_epsilon)*scale
     xmid=x+projected
     np.testing.assert_allclose(pr['o_query'][0],norm(xmid,'post_self_attention_layer_norm'),rtol=2e-5,atol=2e-5)
     np.testing.assert_allclose(pr['write_query'][0],norm(x,'pre_self_attention_layer_norm'),rtol=2e-5,atol=2e-5)
     matt=.73*m0+jnp.einsum('btnk,btnv->btkv',pr['write_content'][0],pr['write_address'][0])
     read_m=matt if c.bam_local_o_post_write else m0
     def expected_o(mod):
      att=Capture(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel=c.attention,dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
      def read(inst):
       mr=inst._matrix_for_read(read_m)
       d,g=inst._read_fetched_m(inst._compress_m(mr),norm(xmid,'post_self_attention_layer_norm'),ungated=True)
       return inst._gate_local_output(d,g)+inst._static_column(mr,'o')
      return att.apply({'params':p['self_attention']},method=read)
     local=expected_o(None).reshape(x.shape)
     z=norm(xmid+local,'post_self_attention_layer_norm')
     np.testing.assert_allclose(probes['probe']['mlp']['input'][0],z,rtol=3e-5,atol=3e-5)
     norms=probes['intermediates']['post_self_attention_layer_norm']['__call__']
     self.assertEqual(len(norms),2)
     np.testing.assert_allclose(norms[0],norm(xmid,'post_self_attention_layer_norm'),rtol=2e-5,atol=2e-5)
     np.testing.assert_allclose(norms[1],z,rtol=3e-5,atol=3e-5)
     if layer==0:np.testing.assert_allclose(mout,matt,rtol=2e-5,atol=2e-5)
     def loss(params):
      o,m=a.apply({'params':params},*call,M_in=m0)
      return jnp.mean(o**2)+jnp.mean(m**2)
     v,g=jax.jit(jax.value_and_grad(loss))(p)
     self.assertTrue(np.isfinite(float(v)))
     self.assertTrue(all(np.isfinite(np.asarray(t)).all() for t in jax.tree.leaves(g)))
     self.assertGreater(float(jnp.sum(g['post_self_attention_layer_norm']['scale']**2)),0)
    print('UPDATED_QUERY_FUSION_OK',exp,layer,flush=True)

 def test_parent_initialization_equality(self):
  trees=[]
  for exp in [POST,PRE,*ARMS]:
   c=self.config(exp,dtype='float32',weight_dtype='float32')
   c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,bam_write_v_bottleneck_dim=16,mlp_dim_by_block=[128]*3,bam_record_concat_health=False,bam_record_write_health=False,bam_record_address_health=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   a=fusion.SubDecoderLayer(c,mesh,layer_inx=0,sliding_window_size=4)
   x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32));tok=jnp.ones((1,4),jnp.int32)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    p=a.init(jax.random.key(3),x,tok,jnp.arange(4)[None],tok,None,True,'train',tok,M_in=m)['params']
   trees.append(flatten_dict(nn.unbox(p)))
  for tree in trees[1:]:
   self.assertEqual(trees[0].keys(),tree.keys())
   for path,v in trees[0].items():np.testing.assert_array_equal(v,tree[path],err_msg=str(path))
  print('RAW_WO_POST_PRE_INITIALIZATION_IDENTICAL',flush=True)

if __name__=='__main__':unittest.main()

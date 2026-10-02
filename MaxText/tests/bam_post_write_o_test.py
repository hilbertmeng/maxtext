"""Post-write O dependency, carry decay once, and exact full-model budget."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions
from bam_mlp_write_test import MLPWriteTest, PREFIX
EXP=PREFIX+'IndependentEveryThirdPostWriteLocalONoWOTruePile'
class PostWriteTest(MLPWriteTest):
 def test_budget_health(self):
  c=self.config(EXP)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
  flat=flatten_dict(args[0].params)
  count=sum(int(np.prod(v.shape)) for v in flat.values())
  self.assertEqual(count,432139328)
  self.assertFalse(any('output_head_mix' in p or ('self_attention' in p and 'out' in p) for p in flat))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
  for l in range(18):
   self.assertIn(f'bam/concat/local_o_gate/layer_{l:03d}/mean',metrics)
   self.assertIn(f'bam/concat/post_write_delta_amplitude/layer_{l:03d}/bam_over_standard',metrics)
  print('POST_WRITE_BUDGET_HEALTH_OK',count,flush=True)
 def test_routes_gradients(self):
  c=self.config(EXP,dtype='float32',weight_dtype='float32')
  c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,bam_write_v_bottleneck_dim=16,bam_record_concat_health=False,bam_record_write_health=False,bam_record_address_health=False,bam_lambda_decay=.73)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
  x=jax.random.normal(jax.random.key(10),(1,4,150));M=jax.random.normal(jax.random.key(11),(1,4,75,32));pos=jnp.arange(4)[None];mask=jnp.ones((1,4),jnp.int32);call=(x,x,pos,mask)
  def read_o(mod,m,x):
   m=mod._matrix_for_read(m)
   read,logits=mod._read_fetched_m(mod._compress_m(m),x,ungated=True)
   return (mod._gate_local_output(read,logits)+mod._static_column(m,'o')).reshape(x.shape)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   params=nn.unbox(a.init(jax.random.key(12),*call,M_in=M,deterministic=True)['params'])
   self.assertGreater(float(jnp.sum(params['static_o_key']**2)),0)
   self.assertEqual(params['static_o_key'].shape,(32,2))
   out,m=a.apply({'params':params},*call,M_in=M,deterministic=True)
   reference=a.apply({'params':params},m,x,method=read_o)
   np.testing.assert_allclose(out,reference,rtol=1e-5,atol=1e-5)
   old=a.apply({'params':params},M,x,method=read_o)
   self.assertGreater(float(jnp.max(jnp.abs(out-old))),1e-5)
   deferred=a.apply({'params':params},*call,M_in=M,deterministic=True,defer_write=True)
   np.testing.assert_allclose(out,deferred[0],rtol=1e-5,atol=1e-5)
   np.testing.assert_array_equal(M,deferred[1])
   y=jax.random.normal(jax.random.key(15),(1,4,2,75));g=jnp.full((1,4,2),.23);addr=jax.random.normal(jax.random.key(16),(1,4,2,32))
   merged=a.apply({'params':params},y,g,deferred[2],M,None,addr,method=a.merge_mlp_write)
   yn=a.apply({'params':params},y,method=lambda mod,z:mod.write_data_norm(z))
   an=a.apply({'params':params},addr,method=lambda mod,z:mod.write_address_norm(z))
   scale=2**-.5 if c.bam_sqrt_n_scale else 1.
   np.testing.assert_allclose(merged,m+jnp.einsum('btnk,btnv->btkv',scale*g[...,None]*yn,an),rtol=1e-5,atol=1e-5)
   changed=dict(params,static_o_key=jnp.zeros_like(params['static_o_key']))
   out2,m2=a.apply({'params':changed},*call,M_in=M,deterministic=True)
   np.testing.assert_array_equal(m,m2)
   self.assertGreater(float(jnp.max(jnp.abs(out-out2))),1e-5)
   def objective(p):
    o,m=a.apply({'params':p},*call,M_in=M,deterministic=True)
    return jnp.mean(o**2)+jnp.mean(m**2)
   val,grad=jax.jit(jax.value_and_grad(objective))(params)
   self.assertTrue(np.isfinite(float(val)))
   self.assertTrue(all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad)))
   self.assertGreater(float(jnp.sum(grad['static_o_key']**2)),0)
  print('POST_WRITE_ROUTES_DECAY_GRAD_OK',flush=True)
if __name__=='__main__':unittest.main()

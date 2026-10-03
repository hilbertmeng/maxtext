"""Controlled post-write variants: zero static O, gated H or raw-only W_O."""
import functools, os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions
from bam_mlp_write_test import MLPWriteTest, PREFIX
import unittest

ONE=PREFIX+'IndependentEveryThirdPostWriteLocalOGatedHeadMixNoWOTruePile'
TWO=PREFIX+'IndependentEveryThirdPostWriteLocalORawWOTruePile'
THREE=PREFIX+'IndependentEveryThirdPreWriteLocalORawWOTruePile'
ACTIVE=[os.environ['BAM_CONTROLLED_TEST_EXP']] if 'BAM_CONTROLLED_TEST_EXP' in os.environ else [ONE,TWO]
PARENT=PREFIX+'IndependentEveryThirdHeadMixNoWOSeparateVOKeysTruePile'

class Capture(attentions.BamAttention):
 def _query_chunk_op(self,*args,**kw):
  result=super()._query_chunk_op(*args,**kw)
  self.sow('probe','raw_attention',result[0])
  return result

class ControlledPostWriteTest(MLPWriteTest):
 def test_exact_budget_health(self):
  for exp,count,widths in [(ONE,432101024,[4253,4126,4253]),(TWO,432139328,[3859,3732,3859]),(THREE,432139328,[3859,3732,3859])]:
   if exp not in ACTIVE:continue
   c=self.config(exp)
   self.assertEqual(c.mlp_dim_by_block,widths)
   self.assertEqual(c.DATASET_VARIANT,'truepile4096')
   self.assertTrue(c.scan_layers)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
   flat=flatten_dict(args[0].params)
   self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),count)
   for name in ['output_head_mix','output_head_gate','output_head_gate_bias']:
    self.assertEqual(any(name in p for p in flat),exp==ONE,name)
   self.assertEqual(any('self_attention' in p and 'out' in p for p in flat),exp!=ONE)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
   for l in range(18):
    if c.bam_local_o_post_write:self.assertIn(f'bam/concat/post_write_delta_amplitude/layer_{l:03d}/bam_over_standard',metrics)
    self.assertIn(f'bam/concat/local_o_gate/layer_{l:03d}/mean',metrics)
   print('CONTROLLED_BUDGET_HEALTH_OK',exp,count,widths,flush=True)

 def test_paths_decay_and_gradients(self):
  for exp in ACTIVE:
   c=self.config(exp,dtype='float32',weight_dtype='float32')
   c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,bam_write_v_bottleneck_dim=16,bam_record_concat_health=False,bam_record_write_health=False,bam_record_address_health=False,bam_lambda_decay=.73)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   a=Capture(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
   x=jax.random.normal(jax.random.key(10),(1,4,150))
   m0=jax.random.normal(jax.random.key(11),(1,4,75,32))
   call=(x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
   def read_o(mod,m):
    m=mod._matrix_for_read(m)
    d,g=mod._read_fetched_m(mod._compress_m(m),x,ungated=True)
    return mod._gate_local_output(d,g)+mod._static_column(m,'o')
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    p=nn.unbox(a.init(jax.random.key(12),*call,M_in=m0,deterministic=True)['params'])
    np.testing.assert_array_equal(p['static_o_key'],jnp.zeros((32,2)))
    (out,m1),probe=a.apply({'params':p},*call,M_in=m0,deterministic=True,mutable=['probe'])
    raw=probe['probe']['raw_attention'][0]
    factors=a.apply({'params':p},raw,x,method=a._deferred_write_factors)
    np.testing.assert_allclose(m1,.73*m0+jnp.einsum('btnk,btnv->btkv',factors[0],factors[1]),rtol=1e-5,atol=1e-5)
    read_matrix=m1 if c.bam_local_o_post_write else m0
    local=a.apply({'params':p},read_matrix,method=read_o)
    if exp==ONE:
     expected=a.apply({'params':p},raw,local,x,method=a._mix_attention_output).reshape(x.shape)
     # Exact best-parent trajectory except that O reads the updated M.
     c.get_keys()['bam_local_o_post_write']=False
     old_out,old_m=a.apply({'params':p},*call,M_in=m0,deterministic=True)
     old_local=a.apply({'params':p},m0,method=read_o)
     expected_delta=(local-old_local).reshape(x.shape)
     np.testing.assert_allclose(out-old_out,expected_delta,rtol=2e-5,atol=2e-5)
     np.testing.assert_array_equal(m1,old_m)
     c.get_keys()['bam_local_o_post_write']=True
     changed=dict(p,output_head_mix=jnp.zeros_like(p['output_head_mix']))
    else:
     projected=jnp.einsum('btnk,nkd->btd',raw,p['out']['kernel'])
     expected=projected+local.reshape(x.shape)
     changed=dict(p,out=jax.tree.map(jnp.zeros_like,p['out']))
    if exp==THREE:
     c.get_keys()['bam_local_o_post_write']=True
     post_out,post_m=a.apply({'params':p},*call,M_in=m0,deterministic=True)
     post_local=a.apply({'params':p},m1,method=read_o)
     np.testing.assert_allclose(post_out-out,(post_local-local).reshape(x.shape),rtol=2e-5,atol=2e-5)
     np.testing.assert_allclose(m1,post_m,rtol=1e-5,atol=1e-5)
     c.get_keys()['bam_local_o_post_write']=False
    np.testing.assert_allclose(out,expected,rtol=1e-5,atol=1e-5)
    o2,m2=a.apply({'params':changed},*call,M_in=m0,deterministic=True)
    np.testing.assert_array_equal(m1,m2)
    np.testing.assert_allclose(o2,local.reshape(x.shape),rtol=1e-5,atol=1e-5)
    # Static O must bypass H/W_O and cannot contaminate the memory write.
    p_static=dict(p,static_o_key=jax.random.normal(jax.random.key(13),(32,2))*.1)
    o3,m3=a.apply({'params':p_static},*call,M_in=m0,deterministic=True)
    l3=a.apply({'params':p_static},read_matrix,method=read_o)
    np.testing.assert_allclose(o3-out,(l3-local).reshape(x.shape),rtol=2e-5,atol=2e-5)
    np.testing.assert_array_equal(m1,m3)
    od,md,fd=a.apply({'params':p},*call,M_in=m0,deterministic=True,defer_write=True)
    np.testing.assert_array_equal(md,m0)
    np.testing.assert_allclose(od,out,rtol=1e-5,atol=1e-5)
    y=jax.random.normal(jax.random.key(15),raw.shape)
    gate=jnp.full(raw.shape[:-1],.23)
    addr=jax.random.normal(jax.random.key(16),(1,4,2,32))
    merged=a.apply({'params':p},y,gate,fd,m0,None,addr,method=a.merge_mlp_write)
    yn=a.apply({'params':p},y,method=lambda mod,z:mod.write_data_norm(z))
    an=a.apply({'params':p},addr,method=lambda mod,z:mod.write_address_norm(z))
    scale=2**-.5 if c.bam_sqrt_n_scale else 1.
    np.testing.assert_allclose(merged,m1+jnp.einsum('btnk,btnv->btkv',scale*gate[...,None]*yn,an),rtol=1e-5,atol=1e-5)
    def loss(params):
     o,m=a.apply({'params':params},*call,M_in=m0,deterministic=True)
     return jnp.mean(o**2)+jnp.mean(m**2)
    val,grad=jax.jit(jax.value_and_grad(loss))(p)
    self.assertTrue(np.isfinite(float(val)))
    self.assertTrue(all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grad)))
    self.assertGreater(float(jnp.sum(grad['static_o_key']**2)),0)
   print('CONTROLLED_PATHS_GRAD_OK',exp,flush=True)

if __name__=='__main__':unittest.main()

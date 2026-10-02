"""Head mixing, old-M reads, raw attention writes, and exact budgets."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, train, train_compile
from layers import attentions
from bam_mlp_write_test import MLPWriteTest, PREFIX

BASE=PREFIX+'IndependentEveryThird'
EXPS=[BASE+'HeadMixNoWOTruePile',BASE+'HeadMixNoWOSeparateVOKeysTruePile',BASE+'HeadMixNoWOSeparateVOKeysNoGateTruePile']

class HeadMixTest(MLPWriteTest):
  def test_budgets_and_health(self):
    for exp,count in zip(EXPS,[432122624,432101024,432143936]):
      c=self.config(exp)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),count)
      self.assertFalse(any('self_attention' in p and 'out' in p for p in flat))
      self.assertEqual(any('W_R_v' in p for p in flat),c.bam_local_vo_separate_c8_keys)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
      for l in range(18):
        self.assertIn(f'bam/concat/output_head_gate/layer_{l:03d}/mean',metrics)
        self.assertIn(f'bam/concat/output_head_mix_weights/layer_{l:03d}/negative_fraction',metrics)
        self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in metrics,l%3==1)
      print('HEAD_MIX_BUDGET_HEALTH_OK',exp,count,flush=True)

  def test_output_and_raw_write_routes(self):
    for exp in EXPS:
      c=self.config(exp,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
          bam_write_v_bottleneck_dim=16,bam_record_concat_health=False,
          bam_record_write_health=False,bam_record_address_health=False)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,
          head_dim=75,bam_k=75,bam_v=32,max_target_length=4,
          max_prefill_predict_length=4,mesh=mesh,
          attention_kernel='dot_product_chunk',dtype=c.dtype,
          layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
      x=jax.random.normal(jax.random.key(10),(1,4,150))
      M=jax.random.normal(jax.random.key(11),(1,4,75,32))
      pos=jnp.arange(4)[None];mask=jnp.ones((1,4),jnp.int32)
      call=(x,x,pos,mask)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        params=a.init(jax.random.key(12),*call,M_in=M,deterministic=True)['params']
        plain=nn.unbox(params)
        H=plain['output_head_mix']
        if c.bam_output_head_gate:
          np.testing.assert_array_equal(plain['output_head_gate']['kernel'],jnp.zeros_like(plain['output_head_gate']['kernel']))
          np.testing.assert_allclose(1.1*jax.nn.sigmoid(plain['output_head_gate_bias']),1,atol=1e-6)
        else:
          self.assertNotIn('output_head_gate',plain)
          self.assertNotIn('output_head_gate_bias',plain)
        self.assertFalse(np.array_equal(H,np.eye(2)))
        raw=jax.random.normal(jax.random.key(13),(1,4,2,75));local=jax.random.normal(jax.random.key(14),raw.shape)
        output=a.apply({'params':params},raw,local,x,method=a._mix_attention_output)
        np.testing.assert_allclose(output,jnp.einsum('btnk,nm->btmk',raw,H)+local,rtol=1e-5,atol=1e-5)
        # Change only the vector-output path: M and deferred write factors must be identical.
        flat=flatten_dict(plain)
        changed=unflatten_dict({p:(jnp.zeros_like(v) if p==('output_head_mix',) else v) for p,v in flat.items()})
        out,m=a.apply({'params':plain},*call,M_in=M,deterministic=True)
        out2,m2=a.apply({'params':changed},*call,M_in=M,deterministic=True)
        self.assertGreater(float(jnp.max(jnp.abs(out-out2))),1e-5)
        np.testing.assert_array_equal(m,m2)
        deferred=a.apply({'params':plain},*call,M_in=M,deterministic=True,defer_write=True)
        deferred2=a.apply({'params':changed},*call,M_in=M,deterministic=True,defer_write=True)
        np.testing.assert_array_equal(deferred[1],M)
        for v,w in zip(deferred[2],deferred2[2]):np.testing.assert_array_equal(v,w)
        f=deferred[2]
        reconstructed=c.bam_lambda_decay*M+jnp.einsum('btnk,btnv->btkv',f[0],f[1])
        np.testing.assert_allclose(m,reconstructed,rtol=1e-5,atol=1e-5)
        def objective(p):
          result=a.apply({'params':p},*call,M_in=M,deterministic=True)
          return jnp.mean(result[0]**2)+jnp.mean(result[1]**2)
        value,grad=jax.jit(jax.value_and_grad(objective))(plain)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.all(np.isfinite(v)) for v in jax.tree.leaves(grad)))
        self.assertGreater(float(jnp.sum(grad['output_head_mix']**2)),0)
        if c.bam_output_head_gate:
          self.assertGreater(float(jnp.sum(grad['output_head_gate']['kernel']**2)),0)
      print('HEAD_MIX_ROUTES_GRAD_OK',exp,flush=True)

if __name__=='__main__':unittest.main()

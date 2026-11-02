"""GELU bottleneck adapter identity, nonlinear oracle, sharding and fused-model gradients."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_content_adapter_test import AdapterFusedTest, PREFIX
from bam_standard_value_test import StandardValueTest
from layers.bam_content_adapter import HeadContentAdapter, relative_update_norm, parameter_health

PARENT=PREFIX+'ContentWriteOAdapterR16TruePile'
RUN=PREFIX+'ContentWriteOAdapterR16GeluTruePile'
class GeluTest(MLPWriteTest,unittest.TestCase):
  def test_identity_nonlinear_oracle_and_initial_learning(self):
    x=jax.random.normal(jax.random.key(1),(1,7,2,75));mod=HeadContentAdapter(2,75,16,dtype=jnp.float32,activation='gelu')
    p=nn.unbox(mod.init(jax.random.key(2),x));np.testing.assert_array_equal(mod.apply(p,x),x)
    g=jax.grad(lambda q:jnp.sum(mod.apply(q,x)**2))(p)
    self.assertGreater(float(jnp.linalg.norm(g['params']['up'])),0);np.testing.assert_array_equal(g['params']['down'],0)
    p['params']['up']=jax.random.normal(jax.random.key(3),p['params']['up'].shape)*.05
    def oracle(q,z):return z+jnp.einsum('...nr,nrd->...nd',nn.gelu(jnp.einsum('...nd,ndr->...nr',z,q['params']['down'])),q['params']['up'])
    np.testing.assert_allclose(mod.apply(p,x),oracle(p,x),rtol=1e-6,atol=1e-6)
    g1=jax.grad(lambda q:jnp.sum(mod.apply(q,x)**2))(p);g2=jax.grad(lambda q:jnp.sum(oracle(q,x)**2))(p)
    for a,b in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
    self.assertGreater(float(jnp.max(jnp.abs(mod.apply(p,2*x)-2*mod.apply(p,x)))),.001)
    # The reported nonlinear health is the adapter Jacobian at zero, not B@C globally.
    zero=jnp.zeros((2,75));jac=jax.jacfwd(lambda z:mod.apply(p,z))(zero)
    delta=jnp.einsum('ndr,nrk->ndk',p['params']['down'],p['params']['up'])
    for h in range(2):np.testing.assert_allclose(jac[h,:,h,:].T,jnp.eye(75)+.5*delta[h],atol=1e-6)
    cfg=SimpleNamespace(param_scan_axis=1,num_decoder_layers=18,bam_content_adapter_activation='gelu')
    health=parameter_health({'params':{'final_local_layer':{'content_o_adapter':p['params']}}},cfg)
    self.assertEqual(len(health),1);np.testing.assert_allclose(health['bam/content_adapter/o/layer_017/jacobian0_update_over_identity'],.5*relative_update_norm(p['params']))

  def test_formal_parent_tree_budget_and_health(self):
    a=self.config(RUN);b=self.config(PARENT)
    for c in (a,b):c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(a),a.mesh_axes)
    aa,*_=train_compile.get_shaped_inputs(mesh,a);bb,*_=train_compile.get_shaped_inputs(mesh,b)
    fa=flatten_dict(nn.unbox(aa[0].params));fb=flatten_dict(nn.unbox(bb[0].params))
    self.assertEqual({k:v.shape for k,v in fa.items()},{k:v.shape for k,v in fb.items()})
    self.assertEqual(sum(math.prod(x.shape) for x in fa.values()),432113410)
    self.assertEqual((a.mlp_dim,a.bam_memory_mlp_dim,a.bam_last_mlp_dim),(2496,1121,3888))
    self.assertEqual(a.compare_runs,[PARENT]);self.assertTrue(a.bam_pallas_core and a.bam_pallas_write)
    self.assertTrue(a.record_training_health_metrics and not a.record_internal_nn_metrics)
    self.assertFalse(a.bam_content_adapter_v);self.assertEqual(a.tensorboard_gcs_rotate_period,500)

  @unittest.skipUnless(jax.device_count()==8,'launch separately on8 virtual CPU devices')
  def test_formal_eight_device_sharding(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);a,_,s,_=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(nn.unbox(a[0].params));parts=flatten_dict(nn.unbox(s.params));total=sum(math.prod(x.shape) for x in flat.values());local=sum(math.prod(parts[p].shard_shape(x.shape)) for p,x in flat.items());self.assertLess(local/(total/8)-1,c.sharding_tolerance)
    for p,x in flat.items():
      if any(str(k).startswith('content_') and str(k).endswith('_adapter') for k in p):self.assertEqual(math.prod(parts[p].shard_shape(x.shape)),math.prod(x.shape)//8)
    print('GELU_FORMAL_SHARDING',total,local,local/(total/8)-1,flush=True)

class GeluFusedTest(MLPWriteTest,unittest.TestCase):
  tiny=AdapterFusedTest.tiny
  def test_whole_model_all_gradients(self):
    self.variant='ContentWriteOAdapterR16Gelu'
    StandardValueTest.check_pallas(self,True)

if __name__=='__main__':unittest.main()

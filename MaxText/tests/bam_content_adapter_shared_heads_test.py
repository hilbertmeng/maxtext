"""Shared full-write adapters: exact tying, summed gradients, budget and real FSDP layout."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_content_adapter_test import AdapterFusedTest,PREFIX
from bam_standard_value_test import StandardValueTest
from layers.bam_content_adapter import HeadContentAdapter,relative_update_norm,parameter_health
PARENT=PREFIX+'ContentWriteAdapterFullTruePile'
RUN=PREFIX+'ContentWriteAdapterFullSharedHeadsTruePile'

class SharedHeadsTest(MLPWriteTest,unittest.TestCase):
  def test_identity_tying_and_summed_head_gradients(self):
    x=jax.random.normal(jax.random.key(1),(1,7,4,75));s=HeadContentAdapter(4,75,dtype=jnp.float32,shared_heads=True);a=HeadContentAdapter(4,75,dtype=jnp.float32)
    ps=nn.unbox(s.init(jax.random.key(2),x));pa=nn.unbox(a.init(jax.random.key(3),x))
    self.assertEqual(ps['params']['delta'].shape,(75,75));np.testing.assert_array_equal(s.apply(ps,x),x)
    ps['params']['delta']=jax.random.normal(jax.random.key(4),(75,75))*.02;pa['params']['delta']=jnp.broadcast_to(ps['params']['delta'],(4,75,75))
    np.testing.assert_allclose(s.apply(ps,x),a.apply(pa,x),rtol=1e-6,atol=1e-6)
    target=jax.random.normal(jax.random.key(5),x.shape)
    gs,dxs=jax.grad(lambda q,z:jnp.sum(s.apply(q,z)*target),argnums=(0,1))(ps,x);ga,dxa=jax.grad(lambda q,z:jnp.sum(a.apply(q,z)*target),argnums=(0,1))(pa,x)
    np.testing.assert_allclose(gs['params']['delta'],jnp.sum(ga['params']['delta'],axis=0),rtol=1e-5,atol=1e-5);np.testing.assert_allclose(dxs,dxa,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(relative_update_norm(ps['params']),relative_update_norm(pa['params']),rtol=1e-6)
    cfg=SimpleNamespace(param_scan_axis=1,num_decoder_layers=18)
    tree={'params':{'decoder':{'layers':{'self_attention':{'content_write_adapter':{'delta':jnp.stack([ps['params']['delta']]*17,axis=1)}}}}}}
    h=parameter_health(tree,cfg);self.assertEqual(len(h),17);np.testing.assert_allclose(h['bam/content_adapter/write/layer_016/update_over_identity'],relative_update_norm(ps['params']))

  def test_formal_budget_and_unchanged_controls(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params));adapt={p:v for p,v in flat.items() if 'content_write_adapter' in p}
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432110635);self.assertEqual(sum(math.prod(v.shape) for v in adapt.values()),95625)
    self.assertEqual([v.shape for v in adapt.values()],[(75,17,75)])
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3877))
    self.assertLess(abs(3600*2510/(3600*1128+458000)-2),.001)
    self.assertFalse(c.bam_content_adapter_o or c.bam_content_adapter_v);self.assertEqual(c.bam_content_adapter_rank,0)
    self.assertEqual(c.bam_common_vo_read_scale_init,1.);self.assertEqual(c.compare_runs,[PARENT]);self.assertTrue(c.bam_pallas_core and c.bam_pallas_write)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics);self.assertEqual(c.tensorboard_gcs_rotate_period,500);self.assertNotIn('bam_mlp_dim_by_layer',c.get_keys())
    self.assertFalse(any('final_local_layer' in p for p in adapt))
    print('SHARED_FULL_WRITE_BUDGET',432110635,95625,'parent remainder',-1575,flush=True)

  @unittest.skipUnless(jax.device_count()==8,'launch separately on8 virtual CPU devices')
  def test_formal_eight_device_replication_tolerance(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);a,_,s,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(a[0].params));parts=flatten_dict(nn.unbox(s.params));total=sum(math.prod(x.shape) for x in flat.values());local=sum(math.prod(parts[p].shard_shape(x.shape)) for p,x in flat.items());overhead=local/(total/8)-1
    self.assertLess(overhead,c.sharding_tolerance)
    for p,x in flat.items():
      if 'content_write_adapter' in p:self.assertEqual(parts[p].shard_shape(x.shape),x.shape)
    print('SHARED_FULL_WRITE_SHARDING',total,local,overhead,flush=True)

class SharedHeadsFusedTest(MLPWriteTest,unittest.TestCase):
  tiny=AdapterFusedTest.tiny
  def test_whole_model_all_gradients(self):
    self.variant='ContentWriteAdapterFullSharedHeads';StandardValueTest.check_pallas(self,True)

if __name__=='__main__':unittest.main()

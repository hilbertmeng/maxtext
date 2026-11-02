"""Budget, identity, ordering, effective-Delta health and full-model fused gradients."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, pyconfig, train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers.bam_content_adapter import HeadContentAdapter, relative_update_norm, parameter_health

PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile'
PREFIX=PARENT[:-8]
CASES=[('ContentWriteAdapterFull',0,False,False,2494,1119,3904,1530000),
       ('ContentWriteAdapterR16',16,False,False,2503,1124,3910,652800),
       ('ContentWriteOAdapterFull',0,True,False,2477,1111,3879,3150000),
       ('ContentWriteOAdapterR16',16,True,False,2496,1121,3888,1344000),
       ('ContentWriteVAdapterR16',16,False,True,2496,1121,3888,1344000),
       ('ContentWriteVOAdapterR16',16,True,True,2489,1117,3883,2035200)]

class AdapterTest(MLPWriteTest, unittest.TestCase):
  def test_identity_and_learning(self):
    x=jax.random.normal(jax.random.key(1),(2,3,2,75))
    for rank in (0,16):
      mod=HeadContentAdapter(2,75,rank,dtype=jnp.float32)
      p=nn.unbox(mod.init(jax.random.key(2),x))
      np.testing.assert_array_equal(mod.apply(p,x),x)
      g=jax.grad(lambda q:jnp.sum(mod.apply(q,x)**2))(p)
      name='delta' if rank==0 else 'up'
      self.assertGreater(float(jnp.linalg.norm(g['params'][name])),0)
      if rank:
        self.assertGreater(float(jnp.linalg.norm(p['params']['down'])),0)
        np.testing.assert_array_equal(g['params']['down'],0)
        q=jax.tree.map(lambda a,b:a-.001*b,p,g)
        g2=jax.grad(lambda q:jnp.sum(mod.apply(q,x)**2))(q)
        self.assertGreater(float(jnp.linalg.norm(g2['params']['down'])),0)

  def test_v_adapter_commutes_with_attention(self):
    mod=HeadContentAdapter(2,75,16,dtype=jnp.float32)
    x=jax.random.normal(jax.random.key(3),(1,9,2,75));p=nn.unbox(mod.init(jax.random.key(4),x))
    p['params']['up']=jax.random.normal(jax.random.key(5),p['params']['up'].shape)*.03
    weights=jax.nn.softmax(jax.random.normal(jax.random.key(6),(1,2,7,9)),-1)
    mix=lambda y:jnp.einsum('bnts,bsnd->btnd',weights,y)
    np.testing.assert_allclose(mod.apply(p,mix(x)),mix(mod.apply(p,x)),atol=1e-6,rtol=1e-5)

  def test_effective_delta_health_including_scan_and_tail(self):
    p={'down':jax.random.normal(jax.random.key(7),(2,75,16)),
       'up':jax.random.normal(jax.random.key(8),(2,16,75))}
    delta=jnp.einsum('ndr,nrk->ndk',p['down'],p['up'])
    np.testing.assert_allclose(relative_update_norm(p),relative_update_norm({'delta':delta}),rtol=1e-6)
    cfg=SimpleNamespace(param_scan_axis=1,num_decoder_layers=18)
    tree={'params':{'decoder':{'layers':{'self_attention':{'content_write_adapter':jax.tree.map(lambda a:jnp.stack([a,a*2],axis=1),p)}},
             'final_local_layer_0':{'self_attention':{'content_o_adapter':p}}}}}
    health=parameter_health(tree,cfg)
    self.assertEqual(len(health),3)
    np.testing.assert_allclose(health['bam/content_adapter/write/layer_001/update_over_identity'],4*relative_update_norm(p))
    self.assertIn('bam/content_adapter/o/layer_017/update_over_identity',health)

  def test_formal_budgets_and_parent_settings(self):
    for suffix,rank,o,v,w,m,t,extra in CASES:
      c=self.config(PREFIX+suffix+'TruePile');c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      adapter={p:x for p,x in flat.items() if any(str(a).startswith('content_') and str(a).endswith('_adapter') for a in p)}
      total=sum(math.prod(x.shape) for x in flat.values());actual=sum(math.prod(x.shape) for x in adapter.values())
      print('ADAPTER_BUDGET',suffix,total,actual,'widths',w,m,t,flush=True)
      self.assertEqual(total,432112210+(1200 if rank else 0));self.assertEqual(actual,extra)
      self.assertEqual(c.bam_common_vo_read_scale_init,1.)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.scan_layers);self.assertNotIn('bam_mlp_dim_by_layer',c.get_keys())
      self.assertIsNone(c.bam_attention_write_content_divisor);self.assertEqual(c.bam_mlp_write_content_divisor,1.)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      self.assertEqual(c.compare_runs,[PARENT]);self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(w,m,t))
      self.assertFalse(any('content_write_adapter' in p and 'final_local_layer_0' in p for p in adapter))
      self.assertEqual(len([p for p in adapter if 'content_write_adapter' in p]),1 if rank==0 else 2)

# Reuse the whole-model all-gradient comparison, without inheriting unrelated tests.
class AdapterFusedTest(MLPWriteTest, unittest.TestCase):
  def tiny(self,period=3,keep=True,length=128):
    c=self.config(PREFIX+self.variant+'TruePile',bam_splash_attention=False,
        base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],
        bam_memory_mlp_dim=16,bam_final_local_layer_count=1,
        bam_final_local_mlp_dim=None,bam_last_mlp_dim=32,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c
  def check(self,variant):
    self.variant=variant
    StandardValueTest.check_pallas(self,True)
  def test_write_full_all_gradients(self):self.check('ContentWriteAdapterFull')
  def test_write_r16_all_gradients(self):self.check('ContentWriteAdapterR16')
  def test_write_o_full_all_gradients(self):self.check('ContentWriteOAdapterFull')
  def test_write_o_r16_all_gradients(self):self.check('ContentWriteOAdapterR16')
  def test_write_v_r16_all_gradients(self):self.check('ContentWriteVAdapterR16')
  def test_write_vo_r16_all_gradients(self):self.check('ContentWriteVOAdapterR16')

if __name__=='__main__':unittest.main()

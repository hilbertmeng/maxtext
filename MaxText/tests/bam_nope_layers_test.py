"""Periodic pure-M Q/K with exact local MLP refunds and unchanged v8t kernels."""
import math
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers.attentions import BamAttention

PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile'
PREFIX=PARENT.removesuffix('TruePile')+'NoPEEvery3'
RUNS=[PREFIX+'RefundOrdinaryTruePile',PREFIX+'RefundRatio2to1TruePile']

class NoPELayersTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,period=3,keep=True,length=4):
    c=self.config(RUNS[0 if keep else 1],bam_splash_attention=False,
        base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=6,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_layer_modes=['local_qk+local_v+local_o']*6,
        bam_attention_write_layers=list(range(5)),bam_mlp_write_layers=list(range(5)),
        mlp_dim_by_block=[32,32,40 if keep else 36],
        bam_memory_mlp_dim=16,bam_memory_mlp_dim_by_layer=None if keep else [16,16,20]*2,
        bam_memory_mlp_last_write_dim=0,bam_final_local_layer_count=3,
        bam_final_local_mlp_dim=None,bam_last_mlp_dim=48,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_formal_parameter_budgets_and_scanned_patterns(self):
    for run in [PARENT]+RUNS:
      c=self.config(run);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432121408,run)
      self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if p[-1]=='local_o_dynamic_norm_scale'),9216)
      self.assertFalse(c.qk_norm or c.record_internal_nn_metrics or c.bam_record_m_shift_health)
      self.assertTrue(c.record_training_health_metrics and c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      if run==PARENT:continue
      self.assertTrue(c.scan_layers and c.bam_pair_scan and c.bam_pallas_core and c.bam_pallas_write)
      self.assertEqual((c.bam_local_fetch_block_size,c.bam_final_local_layer_count,c.bam_last_mlp_dim),(3,3,4096))
      self.assertEqual(c.bam_local_o_dynamic_norm_scale_init,.5)
      self.assertEqual(c.compare_runs,[PARENT])
      self.assertNotIn('bam_mlp_dim_by_layer',c.get_keys())
      selected=[(p,v) for p,v in flat.items() if 'fetch_2' in p or 'final_local_layer_2' in p]
      self.assertTrue(selected)
      self.assertFalse(any('query' in p or 'key' in p for p,v in selected))
      for key in ('W_lq_c8_up','W_lk_c8_up','static_q_key','static_k_key','W_R_up','M_shift_gate'):
        self.assertTrue(any(key in p for p,v in selected),key)
      for p,v in flat.items():
        if 'final_local_layer_2' in p:
          self.assertFalse(set(p)&{'memory_mlp','P_loc_down','P_loc_up','W_gw','mlp_address_up'})
      self.assertTrue(any('query' in p and 'local_0' in p for p in flat))
    print('NOPE_EVERY3_BOTH_EXACT_BUDGET_432121408',flush=True)

  def test_deleted_projections_match_zero_parent_and_m_reads_remain_live(self):
    c=self.tiny();mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    def module(cfg):
      return BamAttention(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=75,
          bam_k=75,bam_v=32,read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=2,
          max_target_length=4,max_prefill_predict_length=4,mesh=mesh,
          attention_kernel='dot_product_chunk',dtype=cfg.dtype)
    oldc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_standard_qk_drop_every=0)))
    new,old=module(c),module(oldc)
    x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    args=(x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(new.init(jax.random.key(3),*args,M_in=m));op=nn.unbox(old.init(jax.random.key(3),*args,M_in=m))
      nf,of=flatten_dict(p),flatten_dict(op)
      removed=set(of)-set(nf);self.assertEqual({q[-2] for q in removed},{'query','key'})
      for q,v in nf.items():np.testing.assert_array_equal(v,of[q])
      op=unflatten_dict({q:jnp.zeros_like(v) if q in removed else v for q,v in of.items()})
      a=new.apply(p,*args,M_in=m,deterministic=True);b=old.apply(op,*args,M_in=m,deterministic=True)
      for u,v in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(u,v,rtol=1e-6,atol=1e-7)
      gradient=jax.grad(lambda mm:jnp.mean(new.apply(p,*args,M_in=mm,deterministic=True)[0]**2))(m)
      self.assertGreater(float(jnp.sum(gradient**2)),0)
    print('NOPE_REMOVED_PARAMS_EQUIVALENT_ZERO_PARENT_AND_LIVE_M_GRAD',flush=True)

  def test_native_pallas_ordinary_refund_all_gradients(self):
    StandardValueTest.check_pallas(self,True)

  def test_native_pallas_ratio_refund_all_gradients(self):
    StandardValueTest.check_pallas(self,False)

if __name__=='__main__':unittest.main()

"""L15/L17 QKVM: exact QK budget and live intermediate writing."""
import math
import unittest
from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import numpy as np
from bam_mudd_l15_vm_test import L15VMTest
from layers import bam_mudd
import max_utils
import train_compile

EXP='BamMediumPropMShiftReadG050MuddQKVMSharedRouterL15L17L2AnchorTruePile'


class L15QKVMTest(L15VMTest):
  def test_both_qkvm_formal_budget_and_live_writes(self):
    cfg=self.config(EXP,bam_splash_attention=False)
    cfg.get_keys().update(max_target_length=256,max_prefill_predict_length=256,query_chunk_size=256)
    self.assertEqual([i for i in range(18) if bam_mudd.is_qk_layer(cfg,i)],[15,17])
    self.assertEqual([i for i in range(18) if bam_mudd.skips_attention_write(cfg,i)],[17])
    self.assertEqual([i for i in range(18) if bam_mudd.skips_mlp_write(cfg,i)],[16])
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    state,*_=train_compile.get_shaped_inputs(mesh,cfg)
    flat=flatten_dict(nn.unbox(state[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432342420)
    self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items()
        if 'qkvm_compose' in p or 'compose_final_three_source' in p),232740)
    prefix=('params','decoder','tail_layers')
    for layer,width in zip(('local_0','local_1','fetch_2'),(2898,3803,3026)):
      self.assertEqual(flat[prefix+(layer,'block','mlp','wi_0','kernel')].shape,(1200,1,width))
    for layer,writing in (('local_0',True),('fetch_2',False)):
      block=prefix+(layer,'block')
      self.assertEqual(flat[block+('qkvm_compose','mlp','dynamic_dense_conn2','kernel')].shape,(64,1,4,4))
      for name in ('query','key','value'):
        self.assertEqual(flat[block+('self_attention',name,'kernel')].shape,(1200,1,16,75))
      for name in ('P_loc_down','P_loc_up','W_gw','gw_b0'):
        self.assertEqual(any(p[:len(block)]==block and name in p for p in flat),writing)
      self.assertTrue(any(p[:len(block)]==block and 'static_o_key' in p for p in flat))
    self.assertFalse(any(p[:len(prefix)]==prefix and 'mlp_address_down' in p for p in flat))

  def test_both_qkvm_identity_causality_and_all_routes_gradients(self):
    cfg,mesh,model,params,*_=self.build(EXP)
    flat=flatten_dict(params)
    for p,value in flat.items():
      if 'qkvm_compose' in p and 'dynamic_dense_conn2' in p:
        np.testing.assert_array_equal(np.asarray(value),np.zeros(value.shape))
      if 'qkvm_compose' in p and 'dense_proj2.bias' in p:
        np.testing.assert_array_equal(np.asarray(value),np.broadcast_to([0.,0.,0.,1.],value.shape))
    self.check_l15_model(EXP)

  def test_both_qkvm_v8t_bridge_preserves_live_write(self):
    self.check_v8t_tail_layout_bridge(((EXP,6),))


if __name__=='__main__':
  unittest.main()

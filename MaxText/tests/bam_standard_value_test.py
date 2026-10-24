"""Plain residual V with BAM Q/K/O: budgets, branch provenance, and fused gradients."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_mha_writer_test import MHAWriterTest
from common_types import MODEL_MODE_TRAIN
from layers import fusion
from layers.models import Transformer

PREFIX='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1StandardVEvery3'

class StandardValueTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,period=3,keep=True,length=4):
    c=self.config(PREFIX+('Keep' if keep else 'Drop')+'MemoryTruePile',
        bam_splash_attention=False,base_emb_dim=150,base_num_query_heads=2,
        base_num_kv_heads=2,base_num_decoder_layers=6,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32')
    writers=[i for i in range(5) if keep or (i+1)%3!=0]
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_mha_writer_layers=None,bam_standard_value_layers=[2,5],
        bam_layer_modes=['local_qk+local_v+local_o']*6,
        bam_attention_write_layers=list(range(5)),bam_mlp_write_layers=writers,
        bam_mlp_dim_by_layer=[32]*6,bam_memory_mlp_dim_by_layer=[16 if i in writers else 0 for i in range(6)],
        bam_memory_mlp_last_write_dim=0,bam_final_local_layer_count=3,
        bam_final_local_mlp_dim=None,bam_last_mlp_dim=0,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_formal_budgets_and_no_mudd_or_v_read_parameters(self):
    for kind,total in [('Keep',432112816),('Drop',432112416)]:
      c=self.config(PREFIX+kind+'MemoryTruePile');c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),total)
      self.assertFalse(c.bam_mha_writer_layers);self.assertFalse(c.bam_mudd_tail_blocks or c.bam_mudd_final_output)
      self.assertEqual(c.bam_attention_write_layers,list(range(17)));self.assertIn(16,c.bam_mlp_write_layers)
      self.assertEqual(c.bam_memory_mlp_dim_by_layer[16],1140)
      if kind=='Keep':self.assertLess(abs(3600*2276/(3600*1010+458000)-2),.005)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertFalse(any('compose' in '/'.join(p) for p in flat))
      selected=[(p,v) for p,v in flat.items() if 'fetch_2' in p or 'final_local_layer_2' in p]
      for path,v in selected:
        self.assertFalse(set(path)&{'W_lv_gate','W_lv_gate_b0','static_v_key','general_v_static_gate','general_v_static_gate_bias'})
      self.assertTrue(any('value' in p for p,v in selected))

  def test_v_comes_from_residual_while_qko_read_m(self):
    c=self.tiny();mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    mod=fusion.SubDecoderLayer(c,mesh,quant=None,layer_inx=2)
    x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    pos=jnp.arange(4)[None];seg=jnp.ones((1,4),jnp.int32);args=(x,seg,pos,seg,None,True,MODEL_MODE_TRAIN,None)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,is_global=True)['params']);seen=[]
      def capture(next_fun,args,kwargs,ctx):
        value=next_fun(*args,**kwargs)
        if ctx.method_name=='kv_projection' and kwargs.get('proj_name')=='value':seen.append((args[0],value))
        return value
      with nn.intercept_methods(capture):
        y,mm=mod.apply({'params':p},*args,M_in=m,is_global=True)
        z,nnm=mod.apply({'params':p},*args,M_in=-m,is_global=True)
      self.assertEqual(len(seen),2)
      np.testing.assert_array_equal(seen[0][0],seen[1][0]);np.testing.assert_array_equal(seen[0][1],seen[1][1])
      self.assertGreater(float(jnp.max(jnp.abs(y-z))),1e-6)
      a=p['self_attention'];self.assertEqual(a['value']['kernel'].shape,(150,2,75));self.assertEqual(a['query']['kernel'].shape,(150,2,18))
      for key in ['full_m_read_down','W_lq_c8_up','W_lk_c8_up','W_R_up','static_o_key','M_shift_gate']:self.assertIn(key,a)
      for key in ['W_lv_gate','static_v_key','general_v_static_gate']:self.assertNotIn(key,a)
      grad=jax.grad(lambda mm:jnp.sum(mod.apply({'params':p},*args,M_in=mm,is_global=True)[0]**2))(m)
      self.assertGreater(float(jnp.sum(grad*grad)),0)

  def test_scan_causality_and_last_preterminal_writes_are_live(self):
    for keep in (True,False):
      c=self.tiny(keep=keep);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(4),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
        call=lambda q,t:model.apply({'params':q},t,pos,t,seg,seg,enable_dropout=False)[0]
        np.testing.assert_allclose(call(p,tok)[:,:3],call(p,tok.at[:,3].set(7))[:,:3],rtol=1e-6,atol=1e-6)
        grad=jax.jit(jax.grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
      for path,v in flatten_dict(grad).items():
        self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
        if 'final_local_layer_1' in path and ('memory_mlp' in path or 'P_loc_up' in path):self.assertGreater(float(jnp.sum(v*v)),0,str(path))
      end=[p for p in flatten_dict(p) if 'final_local_layer_2' in p]
      for path in end:self.assertFalse(set(path)&{'memory_mlp','P_loc_down','P_loc_up','W_gw','mlp_address_up'})
      for key in ['full_m_read_down','value','W_R_up','static_o_key']:self.assertTrue(any(key in p for p in end),key)

  def test_pallas_keep_all_gradients(self):MHAWriterTest.check_pallas(self,True)
  def test_pallas_drop_all_gradients(self):MHAWriterTest.check_pallas(self,False)

if __name__=='__main__':unittest.main()

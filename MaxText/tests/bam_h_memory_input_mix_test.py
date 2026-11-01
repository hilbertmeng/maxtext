"""H and unit-initialized H: memory-only input mixing and fused-write gradients."""
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
from common_types import MODEL_MODE_TRAIN
from layers import fusion
from layers.models import Transformer

RUNS = ['BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedMLPInputMixG099MemoryOnlyWriteControlMixTruePile', 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedMLPInputMixComplementaryMemoryOnlyWriteControlMixTruePile', 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMLPInputMixG099MemoryOnlyWriteControlMixTruePile', 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMLPInputMixComplementaryMemoryOnlyWriteControlMixTruePile']

class HMemoryInputMixTest(MLPWriteTest, unittest.TestCase):
  variant = 0

  def tiny(self,period=3,keep=True,length=4):
    c=self.config(RUNS[self.variant],base_emb_dim=150,base_num_query_heads=2,
        base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,
        vocab_size=32,dtype='float32',weight_dtype='float32',bam_splash_attention=False)
    c.get_keys().update(bam_pallas_core=False,mlp_dim=32,bam_last_mlp_dim=32,
        bam_memory_mlp_dim=16,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],
        bam_write_v_bottleneck_dim=8,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
        max_target_length=length,max_prefill_predict_length=length,query_chunk_size=length)
    return c

  def test_formal_budgets_parent_contract_and_no_terminal_gate(self):
    for index,run in enumerate(RUNS):
      c=self.config(run)
      c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      gates={p:v for p,v in flat.items() if any('mlp_input_mix_gate' in x for x in p)}
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432132627)
      self.assertEqual(sum(math.prod(v.shape) for v in gates.values()),20417)
      self.assertFalse(any('final_local_layer' in p for p in gates))
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertTrue(c.bam_common_vo_read_scale)
      self.assertEqual(c.bam_common_vo_read_scale_init,None if index<2 else 1.)
      self.assertEqual(c.bam_embedding_write_content_divisor,.0016052833)
      self.assertEqual(c.bam_mlp_write_content_divisor,1.)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertTrue(c.bam_mlp_input_mix_write_controls)
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_fused_inputs)
      self.assertTrue(c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertFalse(c.bam_record_concat_health or c.bam_record_m_shift_health)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      print('H_MEMORY_MIX_BUDGET_OK',index,432132627,flush=True)

  def test_single_gate_real_branch_inputs_controls_and_gradients(self):
      for variant in range(4):
        index = 1
        self.variant = variant
        c = self.tiny()
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        block = fusion.SubDecoderLayer(c,mesh,quant=None,layer_inx=0)
        x = jax.random.normal(jax.random.key(30),(1,4,150))
        matrix = jax.random.normal(jax.random.key(31),(1,4,75,32))
        seg = jnp.ones((1,4),jnp.int32)
        args = (x,seg,jnp.arange(4)[None],seg,None,True,MODEL_MODE_TRAIN,None)
        seen = {}
        def capture(next_fun,args,kwargs,context):
          out = next_fun(*args,**kwargs)
          if context.method_name == '__call__' and context.module.name in (
              'self_attention','post_self_attention_layer_norm','mlp_input_mix_gate',
              'mlp','memory_mlp','mlp_address_down','mlp_write_gate'):
            seen.setdefault(context.module.name,[]).append((args,kwargs,out))
          return out
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          p = nn.unbox(block.init(jax.random.key(32),*args,M_in=matrix))
          np.testing.assert_array_equal(p['params']['mlp_input_mix_gate']['kernel'],0.)
          np.testing.assert_allclose(p['params']['mlp_input_mix_gate_bias'],math.log(99.) if variant % 2 == 0 else 0.,rtol=1e-6)
          gate = p['params']['mlp_input_mix_gate']['kernel']
          p['params']['mlp_input_mix_gate']['kernel'] = .05*jax.random.normal(jax.random.key(33),gate.shape)
          p['params']['mlp_input_mix_gate_bias'] = jnp.array([-.4])
          with nn.intercept_methods(capture):out = block.apply(p,*args,M_in=matrix)
          def objective(pp):
            vector, memory = block.apply(pp,*args,M_in=matrix)
            return jnp.mean(vector**2) + jnp.mean(memory**2)
          grad = jax.grad(objective)(p)
        norm = seen['post_self_attention_layer_norm']
        self.assertEqual(len(norm),2)
        a = seen['self_attention'][0][2][0]
        np.testing.assert_allclose(norm[0][0][0],x+a,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(seen['mlp_input_mix_gate'][0][0][0],norm[0][2])
        g = c.bam_mlp_input_mix_gate_scale*jax.nn.sigmoid(norm[0][2]@p['params']['mlp_input_mix_gate']['kernel']-.4)
        attention_gate = 1. if variant % 2 == 0 else 2.-g
        np.testing.assert_allclose(norm[1][0][0],g*x+attention_gate*a,rtol=1e-6,atol=1e-6)
        ordinary = norm[1][2] if index == 0 else norm[0][2]
        memory = norm[0][2] if index == 0 else norm[1][2]
        np.testing.assert_array_equal(seen['mlp'][0][0][0],ordinary)
        np.testing.assert_array_equal(seen['memory_mlp'][0][0][0],memory)
        np.testing.assert_array_equal(seen['mlp_address_down'][0][0][0],memory)
        np.testing.assert_array_equal(seen['mlp_write_gate'][0][0][0],memory)
        np.testing.assert_allclose(out[0],x+a+seen['mlp'][0][2],rtol=1e-6,atol=1e-6)
        for k in ['mlp_input_mix_gate','mlp_input_mix_gate_bias']:
          for value in jax.tree.leaves(grad['params'][k]):
            self.assertTrue(np.isfinite(value).all())
            self.assertGreater(float(jnp.sum(value*value)),0.)

  def test_single_gate_actual_health_export(self):
      from train import record_bam_mlp_input_mix_health_metrics
      for variant in range(4):
        index = 1
        self.variant = variant
        c = self.tiny()
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        model = Transformer(c,mesh,quant=None)
        tok = jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          p = nn.unbox(model.init(jax.random.key(34),tok,pos,seg,tok))
          _,health = model.apply(p,tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
        metrics = {'scalar':{}}
        record_bam_mlp_input_mix_health_metrics(metrics,health,c)
        self.assertEqual(len(metrics['scalar']),6 if index == 0 else 4)
        other = 'memory' if index == 0 else 'ordinary'
        self.assertFalse(any('/'+other+'/' in k for k in metrics['scalar']))
        for key,value in metrics['scalar'].items():
          expected = (1/1.99 if variant % 2 == 0 else .5) if key.endswith('mean') else 0.
          self.assertAlmostEqual(float(value),expected,places=6)

  def test_complementary_initial_function_matches_each_own_parent(self):
    for variant in (1,3):
      self.variant=variant;c=self.tiny()
      pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
          bam_mlp_input_mix=False,bam_record_mlp_input_mix_health=False)))
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      child,parent=Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p0=nn.unbox(parent.init(jax.random.key(35),tok,pos,seg,tok))
        p1=nn.unbox(child.init(jax.random.key(35),tok,pos,seg,tok))
        left,right=flatten_dict(p0['params']),flatten_dict(p1['params'])
        for path in left:np.testing.assert_array_equal(left[path],right[path],err_msg=str(path))
        y0=parent.apply(p0,tok,pos,seg,tok,enable_dropout=False)[0]
        y1=child.apply(p1,tok,pos,seg,tok,enable_dropout=False)[0]
        np.testing.assert_allclose(y1,y0,rtol=1e-6,atol=1e-6)

  def test_pallas_depth_sigmoid_all_gradients(self):
    self.variant=0;StandardValueTest.check_pallas(self,True)
  def test_pallas_depth_complementary_all_gradients(self):
    self.variant=1;StandardValueTest.check_pallas(self,True)
  def test_pallas_one_sigmoid_all_gradients(self):
    self.variant=2;StandardValueTest.check_pallas(self,True)
  def test_pallas_one_complementary_all_gradients(self):
    self.variant=3;StandardValueTest.check_pallas(self,True)

if __name__=='__main__':unittest.main()

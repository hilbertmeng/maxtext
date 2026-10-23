"""Dedicated memory FFNs: branch isolation, live gradients, scan and v8t writes."""
import copy
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import fusion, quantizations, bam_pallas, normalizations, linears, initializers
from layers.models import Transformer
from common_types import MODEL_MODE_TRAIN

A='BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile'
B='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile'
SH='BamMediumPropMShiftReadG050SharedHiddenMemoryMLPTruePile'
C='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio3to1TruePile'
HALF='BamMediumPropMShiftReadG050HalfMemoryMLPTruePile'
R8='BamMediumPropMShiftReadG050SharedHiddenRank8WriteTruePile'
R8X='BamMediumPropMShiftReadG050SharedHiddenRank8InputControlsTruePile'
I8='BamMediumPropMShiftReadG050DedicatedMemoryMLPRank8R128H768TruePile'

class DedicatedMemoryMLPTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budgets_and_kernel_settings(self):
    for exp,normal,memory in [(A,207360000,31251600),(B,168843600,69768000),(C,188244000,50367600)]:
      c=self.config(exp)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      total=sum(int(np.prod(v.shape)) for v in flat.values())
      ordinary=sum(int(np.prod(v.shape)) for p,v in flat.items() if 'mlp' in p)
      mem=sum(int(np.prod(v.shape)) for p,v in flat.items() if 'memory_mlp' in p)
      self.assertEqual((total,ordinary,mem),(432113280,normal,memory))
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs)
      self.assertEqual((c.bam_pallas_body,c.bam_pallas_vmem_mib),('v8t',48))
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)

  def tiny(self, exp=A, layers=6):
    c=self.config(exp, bam_splash_attention=False, base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=layers,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,mlp_dim_by_block=[32,32],
        bam_memory_mlp_dim=16,bam_memory_mlp_last_write_dim=12,
        bam_final_local_mlp_dim=32,bam_last_mlp_dim=40,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,bam_layer_modes=c.bam_layer_modes[:layers])
    if getattr(c,'bam_memory_mlp_share_hidden',False):
      c.get_keys().update(bam_memory_mlp_dim=32,bam_memory_mlp_last_write_dim=0)
    if getattr(c,'bam_memory_mlp_half',False):
      c.get_keys().update(bam_memory_mlp_dim=c.emb_dim,bam_memory_mlp_last_write_dim=0)
    if getattr(c,'bam_memory_write_heads',0):
      c.get_keys().update(bam_memory_write_heads=1)
    return c

  def test_rank8_formal_budgets_and_no_redundant_controls(self):
    for exp,total in [(R8,432122600),(R8X,432122168)]:
      c=self.config(exp)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),total)
      self.assertEqual(c.bam_memory_write_heads,8)
      self.assertFalse(any(any(n in p for n in ['memory_mlp','mlp_address_down','mlp_address_up','mlp_write_gate']) for p in flat))
      self.assertFalse(any('final_local_layer_1' in p and any('memory_' in n or n=='mlp_address_bias' or n=='mlp_write_gate_bias' for n in p) for p in flat))
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual((c.tensorboard_gcs_rotate_period,c.bam_write_eps),(500,.1))

  def test_independent_rank8_formal_budget_and_shapes(self):
    c=self.config(I8)
    c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,_,_,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
    size=lambda v:int(np.prod(v.shape))
    self.assertEqual(sum(size(v) for v in flat.values()),432112824)
    self.assertEqual(sum(size(v) for p,v in flat.items() if 'mlp' in p),203893200)
    self.assertEqual(sum(size(v) for p,v in flat.items() if 'memory_mlp' in p),39168000)
    control_names={'mlp_address_down','mlp_address_up','mlp_write_gate','mlp_write_gate_bias'}
    self.assertEqual(sum(size(v) for p,v in flat.items() if set(p)&control_names),3335944)
    for p,v in flat.items():
      if 'memory_mlp' in p and 'wo' in p and p[-1]=='kernel':
        self.assertIn(v.shape,[(768,600),(768,8,600)])  # Scan axis1: eight two-layer blocks.
      if 'mlp_address_up' in p and p[-1]=='kernel':
        self.assertIn(v.shape,[(128,8,32),(128,8,8,32)])
    self.assertFalse(any('final_local_layer_1' in p and ('memory_mlp' in p or set(p)&control_names) for p in flat))
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    self.assertEqual((c.bam_memory_write_heads,c.bam_mlp_write_address_rank),(8,128))
    self.assertEqual(c.tensorboard_gcs_rotate_period,500)

  def test_independent_rank8_content_projection_reference(self):
    c=self.tiny(I8)
    mod=linears.MlpBlock(config=c,intermediate_dim=16,activations=c.mlp_activations,
        output_dim=75,kernel_init=initializers.get_bam_init(c,fallback=initializers.get_model_init(c)),
        intermediate_dropout_rate=0,dtype=jnp.float32,weight_dtype=jnp.float32)
    x=jax.random.normal(jax.random.key(51),(1,4,150))
    p=nn.unbox(mod.init(jax.random.key(52),x,deterministic=True)['params'])
    self.assertEqual(set(p),{'wi_0','wi_1','wo'})
    actual=mod.apply({'params':p},x,deterministic=True)
    ref=(jax.nn.silu(x@p['wi_0']['kernel'])*(x@p['wi_1']['kernel']))@p['wo']['kernel']
    self.assertEqual(actual.shape,(1,4,75));np.testing.assert_allclose(actual,ref,rtol=1e-6,atol=1e-6)
    grad=jax.grad(lambda q:jnp.sum(mod.apply({'params':q},x,deterministic=True)**2))(p)
    for name in p:self.assertGreater(float(jnp.sum(grad[name]['kernel']**2)),0,name)

  def test_independent_rank8_scan_gradients_and_gate_initialization(self):
    c=self.tiny(I8);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quantizations.configure_quantization(c))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(53),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
      def call(q,t):return model.apply({'params':q},t,pos,t,seg,seg,enable_dropout=False)[0]
      y=call(p,tok);yp=call(p,tok.at[:,3].set(8))
      np.testing.assert_allclose(y[:,:3],yp[:,:3],rtol=1e-6,atol=1e-6)
      val,grad=jax.jit(jax.value_and_grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
      self.assertTrue(np.isfinite(float(val)))
      for path,v in flatten_dict(grad).items():
        self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
        if set(path)&{'memory_mlp','mlp_address_down','mlp_address_up','mlp_write_gate'}:
          self.assertGreater(float(jnp.sum(v*v)),0,str(path))
      for path,v in flatten_dict(p).items():
        if path[-1]=='mlp_write_gate_bias':np.testing.assert_allclose(jax.nn.sigmoid(v),.1*np.sqrt(2),rtol=1e-6)
        if path[-1]=='gw_b0':np.testing.assert_allclose(jax.nn.sigmoid(v),.1,rtol=1e-6)

  def test_rank8_reference_and_control_gradient_sources(self):
    for exp in [R8,R8X]:
      c=self.tiny(exp)
      mod=linears.MlpBlock(config=c,intermediate_dim=32,activations=c.mlp_activations,
          kernel_init=initializers.get_model_init(c),bam_memory_output=True,
          bam_memory_write_heads=1,bam_memory_controls_from_hidden=c.bam_memory_controls_from_hidden,
          intermediate_dropout_rate=0,dtype=jnp.float32,weight_dtype=jnp.float32)
      x=jax.random.normal(jax.random.key(41),(1,4,150))
      p=nn.unbox(mod.init(jax.random.key(42),x,deterministic=True)['params'])
      hidden=jax.nn.silu(x@p['wi_0']['kernel'])*(x@p['wi_1']['kernel'])
      y,u,a,g=mod.apply({'params':p},x,deterministic=True)
      np.testing.assert_allclose(y,hidden@p['wo']['kernel'],rtol=1e-6,atol=1e-6)
      if exp==R8:
        factors=hidden@p['memory_factors']['kernel']
        ur,ar,gr=jnp.split(factors,[75,107],axis=-1)
      else:
        ur=hidden@p['memory_wo']['kernel']
        ar,gr=jnp.split(x@p['memory_controls']['kernel'],[32],axis=-1)
      for actual,ref in [(u,ur),(a,ar.reshape(1,4,1,32)),(g,gr)]:
        np.testing.assert_allclose(actual,ref,rtol=1e-6,atol=1e-6)
      # Address/gate losses reach shared nonlinear features only in the joint arm.
      grad=jax.grad(lambda q:sum(jnp.sum(z*z) for z in mod.apply({'params':q},x,deterministic=True)[2:]))(p)
      norm=float(jnp.sum(grad['wi_0']['kernel']**2))
      if exp==R8:self.assertGreater(norm,0)
      else:self.assertEqual(norm,0)

  def test_rank8_scan_unequal_heads_live_gradients_and_causality(self):
    for exp in [R8,R8X]:
      c=self.tiny(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(43),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
        def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False)[0]
        y=call(p,tok);yp=call(p,tok.at[:,3].set(8))
        np.testing.assert_allclose(y[:,:3],yp[:,:3],rtol=1e-6,atol=1e-6)
        val,grad=jax.jit(jax.value_and_grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
        self.assertTrue(np.isfinite(float(val)))
        for path,v in flatten_dict(grad).items():
          self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
          if any(n in path for n in ['memory_factors','memory_wo','memory_controls']):
            self.assertGreater(float(jnp.sum(v*v)),0,str(path))

  def test_half_formal_budget_and_missing_output_projection(self):
    c=self.config(HALF)
    c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(args[0].params)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432113280)
    self.assertEqual(sum(int(np.prod(v.shape)) for p,v in flat.items() if 'mlp' in p),189651600)
    self.assertEqual(sum(int(np.prod(v.shape)) for p,v in flat.items() if 'memory_mlp' in p),48960000)
    self.assertFalse(any('memory_mlp' in p and 'wo' in p for p in flat))
    self.assertFalse(any('final_local_layer_1' in p and 'memory_mlp' in p for p in flat))
    self.assertTrue(c.record_training_health_metrics and c.bam_pallas_core and c.sa_use_fused_bwd_kernel)
    self.assertFalse(c.record_internal_nn_metrics)

  def test_half_hidden_reference_and_both_input_gradients(self):
    c=self.tiny(HALF)
    mod=linears.MlpBlock(config=c,intermediate_dim=150,activations=c.mlp_activations,
        kernel_init=initializers.get_bam_init(c,fallback=initializers.get_model_init(c)),
        bam_memory_direct_hidden=True,intermediate_dropout_rate=0,
        dtype=jnp.float32,weight_dtype=jnp.float32)
    x=jax.random.normal(jax.random.key(31),(1,4,150));p=nn.unbox(mod.init(jax.random.key(32),x,deterministic=True)['params'])
    self.assertEqual(set(p),{'wi_0','wi_1'})
    y=mod.apply({'params':p},x,deterministic=True)
    ref=jax.nn.silu(x@p['wi_0']['kernel'])*(x@p['wi_1']['kernel'])
    np.testing.assert_allclose(y,ref,rtol=1e-6,atol=1e-6)
    grad=jax.grad(lambda q:jnp.sum(mod.apply({'params':q},x,deterministic=True)**2))(p)
    for name in ['wi_0','wi_1']:self.assertGreater(float(jnp.sum(grad[name]['kernel']**2)),0,name)

  def test_half_scan_gradients_and_causality(self):
    c=self.tiny(HALF);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quantizations.configure_quantization(c))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(33),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
      def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False)[0]
      y=call(p,tok);yp=call(p,tok.at[:,3].set(8))
      np.testing.assert_allclose(y[:,:3],yp[:,:3],rtol=1e-6,atol=1e-6)
      val,grad=jax.jit(jax.value_and_grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
      self.assertTrue(np.isfinite(float(val)))
      for path,v in flatten_dict(grad).items():
        self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
        if 'memory_mlp' in path:self.assertGreater(float(jnp.sum(v*v)),0,str(path))

  def test_shared_formal_budget_and_projection_tree(self):
    c=self.config(SH)
    c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(args[0].params)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432121680)
    self.assertFalse(any('memory_mlp' in p for p in flat))
    self.assertEqual(sum(int(np.prod(v.shape)) for p,v in flat.items() if 'memory_wo' in p),57140400)
    paths=['/'.join(p) for p in flat]
    self.assertFalse(any('final_local_layer_1/' in p and 'memory_wo' in p for p in paths))
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.record_training_health_metrics)
    self.assertFalse(c.record_internal_nn_metrics)

  def test_shared_hidden_reference_and_head_gradients(self):
    c=self.tiny(SH)
    mod=linears.MlpBlock(config=c,intermediate_dim=32,activations=c.mlp_activations,
        kernel_init=initializers.get_model_init(c),bam_memory_output=True,
        intermediate_dropout_rate=0,dtype=jnp.float32,weight_dtype=jnp.float32)
    x=jax.random.normal(jax.random.key(21),(1,4,150))
    p=nn.unbox(mod.init(jax.random.key(22),x,deterministic=True)['params'])
    self.assertEqual(set(p),{'wi_0','wi_1','wo','memory_wo'})
    a,b=mod.apply({'params':p},x,deterministic=True)
    hidden=jax.nn.silu(x@p['wi_0']['kernel'])*(x@p['wi_1']['kernel'])
    np.testing.assert_allclose(a,hidden@p['wo']['kernel'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(b,hidden@p['memory_wo']['kernel'],rtol=1e-6,atol=1e-6)
    for index,own,other in [(0,'wo','memory_wo'),(1,'memory_wo','wo')]:
      grad=jax.grad(lambda q:jnp.sum(mod.apply({'params':q},x,deterministic=True)[index]**2))(p)
      for name in ['wi_0','wi_1',own]:self.assertGreater(float(jnp.sum(grad[name]['kernel']**2)),0,name)
      self.assertEqual(float(jnp.sum(grad[other]['kernel']**2)),0)

  def test_shared_scan_gradients_causality_and_terminal_head_omission(self):
    c=self.tiny(SH);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quantizations.configure_quantization(c))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(23),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
      def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False)[0]
      y=call(p,tok);yp=call(p,tok.at[:,3].set(8))
      np.testing.assert_allclose(y[:,:3],yp[:,:3],rtol=1e-6,atol=1e-6)
      val,grad=jax.jit(jax.value_and_grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
      self.assertTrue(np.isfinite(float(val)))
      for path,v in flatten_dict(grad).items():
        self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
        if 'memory_wo' in path:self.assertGreater(float(jnp.sum(v*v)),0,str(path))
      self.assertFalse(any('final_local_layer_1' in path and 'memory_wo' in path for path in flatten_dict(p)))

  def test_branches_are_functionally_separate(self):
    c=self.tiny();mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    mod=fusion.SubDecoderLayer(c,mesh,sliding_window_size=4,layer_inx=0)
    x=jax.random.normal(jax.random.key(1),(1,4,150))
    m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tokens)
    args=(x,seg,pos,tokens,None,True,MODEL_MODE_TRAIN,None)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,is_global=True,layer_index=0)['params'])
      def call(p):return mod.apply({'params':p},*args,M_in=m,is_global=True,layer_index=0)
      y,mm=call(p)
      for branch,index in [('memory_mlp',0),('mlp',1)]:
        pp=copy.deepcopy(p);pp[branch]['wo']['kernel']=pp[branch]['wo']['kernel']*1.2+0.01
        yp,mp=call(pp)
        np.testing.assert_array_equal((yp,mp)[index],(y,mm)[index])
        self.assertGreater(float(jnp.max(jnp.abs((yp,mp)[1-index]-(y,mm)[1-index]))),0)
      for which,zero,live in [(0,'memory_mlp','mlp'),(1,'mlp','memory_mlp')]:
        grad=jax.grad(lambda pp:jnp.sum(call(pp)[which]**2))(p)
        self.assertEqual(sum(float(jnp.sum(g*g)) for g in jax.tree.leaves(grad[zero])),0)
        self.assertGreater(sum(float(jnp.sum(g*g)) for g in jax.tree.leaves(grad[live])),0)

  def test_scan_causality_live_memory_gradients_and_terminal_omission(self):
    for exp in [A,B]:
      c=self.tiny(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False)[0]
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(5),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
        y=call(p,tok);changed=call(p,tok.at[:,3].set(7))
        np.testing.assert_allclose(y[:,:3],changed[:,:3],rtol=1e-6,atol=1e-6)
        val,grad=jax.jit(jax.value_and_grad(lambda pp:jnp.mean(call(pp,tok)**2)))(p)
        self.assertTrue(np.isfinite(float(val)))
        flat=flatten_dict(grad)
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
        for path,v in flat.items():
          if 'memory_mlp' in path:
            self.assertGreater(float(jnp.sum(v*v)),0,str(path))
        paths=['/'.join(path) for path in flatten_dict(p)]
        self.assertTrue(any('final_local_layer_0/block/memory_mlp/' in path for path in paths))
        self.assertFalse(any('final_local_layer_1/block/memory_mlp/' in path for path in paths))
        self.assertFalse(any('final_local_layer_1/block/mlp_address_' in path for path in paths))

  def test_v8t_two_group_forward_and_backward(self):
    keys=iter(jax.random.split(jax.random.key(9),10))
    rnd=lambda shape:jax.random.normal(next(keys),shape)
    m=rnd((1,128,75,32));groups=[(rnd((1,128,n,75)),rnd((1,128,n)),rnd((1,128,n,32))) for n in (2,1)]
    flat=tuple(a for group in groups for a in group);eps=1e-6
    def reference(m,*f):
      for i in range(0,len(f),3):
        c,g,a=f[i:i+3]
        c=normalizations.rms_norm(c,dtype=jnp.float32,epsilon=eps,statistics_dtype=jnp.float32)*jax.nn.sigmoid(g)[...,None]
        a=normalizations.rms_norm(a,dtype=jnp.float32,epsilon=eps,statistics_dtype=jnp.float32)
        m=m+jnp.einsum('btnk,btnv->btkv',c,a)
      return m
    def kernel(m,*f):
      gs=[f[i:i+3] for i in range(0,len(f),3)]
      return jnp.swapaxes(bam_pallas.write(jnp.swapaxes(m,-1,-2),gs,epsilon=eps,body='v8t',interpret=True),-1,-2)
    np.testing.assert_allclose(kernel(m,*flat),reference(m,*flat),rtol=2e-5,atol=2e-5)
    ct=rnd(m.shape);argnums=tuple(range(7))
    a=jax.grad(lambda *v:jnp.sum(kernel(*v)*ct),argnums=argnums)(m,*flat)
    b=jax.grad(lambda *v:jnp.sum(reference(*v)*ct),argnums=argnums)(m,*flat)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-4,atol=1e-4)

if __name__=='__main__':unittest.main()

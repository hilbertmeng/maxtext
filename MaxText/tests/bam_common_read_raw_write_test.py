"""Layer-scalar reads and fixed write denominators: exact sizes, semantics, and all-path gradients."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,max_utils,pyconfig,train,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers import bam_pallas_v8,normalizations
from layers.models import Transformer

PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile'
PREFIX=PARENT.removesuffix('TruePile')+'CommonReadScale'
RUN=PREFIX+'RawWriteTruePile'
ATTENTION=PREFIX+'RawAttentionWriteTruePile'
WRITE_ONLY=PARENT.removesuffix('TruePile')+'AttentionWriteScaleRawWriteTruePile'
MLP=PREFIX+'RawMLPWriteTruePile'

class CommonReadRawWriteTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,period=3,keep=True,length=4):
    c=self.config(getattr(self,'experiment',RUN),bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],
        bam_memory_mlp_dim=16,bam_last_mlp_dim=32,emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_exact_formal_budget_and_actual_scan_initial_values(self):
    for run in [RUN,ATTENTION,MLP]:
      c=self.config(run);c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      selected={p:v for p,v in flat.items() if p[-1]=='common_vo_read_scale'}
      self.assertEqual(sum(math.prod(v.shape) for v in selected.values()),18)
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112210)
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertFalse(c.bam_local_o_dynamic_norm_scale)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.compare_runs,[PARENT]);self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    c=self.tiny();mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):params=nn.unbox(model.init(jax.random.key(7),tok,pos,seg,tok))
    flat=flatten_dict(params);selected={p:v for p,v in flat.items() if p[-1]=='common_vo_read_scale'}
    self.assertEqual(len(selected),2)
    for path,v in selected.items():
      if 'layers' in path:np.testing.assert_allclose(jnp.moveaxis(v,c.param_scan_axis,0).reshape(-1),1/jnp.sqrt(1+2*jnp.arange(2)),rtol=1e-6)
      else:np.testing.assert_allclose(v,jnp.array([1/math.sqrt(5)]),rtol=1e-6)

  def test_common_scale_affects_static_dynamic_v_o_and_residual_write_exactly_once(self):
    from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
    c,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
    flat=flatten_dict(p)
    for leaf,key in [('static_v_key',jax.random.key(71)),('static_o_key',jax.random.key(72))]:
      path=next(path for path in flat if path[-1]==leaf)
      flat[path]=jax.random.normal(key,flat[path].shape)*.1
    p=unflatten_dict(flat);path=next(path for path in flat if path[-1]=='common_vo_read_scale')
    captured=[]
    def intercept(next_fun,args,kwargs,context):
      if context.method_name=='_query_chunk_op':captured.append((args[0],args[1]))
      return next_fun(*args,**kwargs)
    def call(scale):
      changed=dict(flat);changed[path]=jnp.reshape(scale,(1,))
      return module.apply(unflatten_dict(changed),*args,M_in=matrix,deterministic=True)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      with nn.intercept_methods(intercept):baseline,mbase=call(jnp.float32(1.))
      qkbase=captured[-1];delta=mbase-matrix
      for scale in [0.,-.5,2.]:
        with nn.intercept_methods(intercept):actual,mout=call(jnp.float32(scale))
        np.testing.assert_allclose(actual,scale*baseline,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(mout-matrix,scale*delta,rtol=3e-5,atol=2e-6)
        for actual_qk,expected_qk in zip(captured[-1],qkbase):np.testing.assert_array_equal(actual_qk,expected_qk)
      target=jax.random.normal(jax.random.key(73),baseline.shape);mtarget=jax.random.normal(jax.random.key(74),matrix.shape)
      residual_grad=jax.grad(lambda scale:jnp.sum(call(scale)[0]*target))(jnp.float32(.7))
      write_grad=jax.grad(lambda scale:jnp.sum((call(scale)[1]-matrix)*mtarget))(jnp.float32(.7))
      np.testing.assert_allclose(residual_grad,jnp.sum(baseline*target),rtol=2e-5,atol=2e-6)
      np.testing.assert_allclose(write_grad,jnp.sum(delta*mtarget),rtol=2e-5,atol=3e-6)
      self.assertGreater(abs(float(residual_grad)),1e-8);self.assertGreater(abs(float(write_grad)),1e-8)

  def test_bf16_divisors_are_not_rounded_before_division(self):
    from layers.bam_write_content import fixed_content
    x=jnp.linspace(-.01,.01,128).astype(jnp.bfloat16)
    for tau in [.0016052833,.1946289063,.0045444682]:
      expected=(x.astype(jnp.float32)/jnp.float32(tau)).astype(jnp.bfloat16)
      np.testing.assert_array_equal(fixed_content(x,tau),expected)

  def test_all18_layer_indices_and_tail_initialization(self):
    c=self.tiny();c.get_keys().update(num_decoder_layers=18,bam_layer_modes=['local_qk+local_v+local_o']*18,
        bam_attention_write_layers=list(range(17)),bam_mlp_write_layers=list(range(17)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):p=nn.unbox(model.init(jax.random.key(12),tok,pos,seg,tok))
    flat=flatten_dict(p);body=[v for k,v in flat.items() if k[-1]=='common_vo_read_scale' and 'layers' in k][0]
    tail=[v for k,v in flat.items() if k[-1]=='common_vo_read_scale' and 'final_local_layer' in k][0]
    values=jnp.concatenate([jnp.moveaxis(body,c.param_scan_axis,0).reshape(-1),tail.reshape(-1)])
    np.testing.assert_allclose(values,1/jnp.sqrt(1+2*jnp.arange(18)),rtol=1e-6)

  def test_signed_parameter_health_and_weight_decay_exemption(self):
    c=self.config(RUN);values=jnp.linspace(-.5,.5,17).reshape(1,17)
    p={'params':{'decoder':{'layers':{'block':{'self_attention':{'common_vo_read_scale':values}}},'final_local_layer':{'block':{'self_attention':{'common_vo_read_scale':jnp.array([-.2])}}}}}}
    metrics=train.compute_params_norm(p,c);signed={k:v for k,v in metrics.items() if k.startswith('parameter_values/')}
    self.assertEqual(len(signed),18);self.assertTrue(any(float(v)<0 for v in signed.values()))
    self.assertTrue(any(abs(float(v)+.2)<1e-6 for v in signed.values()))
    wd=flatten_dict(train.get_wd_tree(SimpleNamespace(wd_mults=getattr(exp,RUN).wd_mults,adam_weight_decay=.1),p))
    self.assertTrue(all(v==0 for v in wd.values()))

  def kernel_case(self,divisors):
    keys=jax.random.split(jax.random.key(30),8);b,t,n,k,v=2,256,3,16,8
    args=(jax.random.normal(keys[0],(b,t,v,k)),jax.random.normal(keys[1],(b,t,n,k)),jax.random.normal(keys[2],(b,t,n)),jax.random.normal(keys[3],(b,t,n,v)),jax.random.normal(keys[4],(b,t,2,k)),jax.random.normal(keys[5],(b,t,2)),jax.random.normal(keys[6],(b,t,2,v)))
    # Content samples match each writer's physical RMS so cancellation tests the model's scale.
    args=list(args);args[1]=args[1]*(divisors[0] or 1.);args[4]=args[4]*(divisors[1] or 1.);args=tuple(args)
    target=jax.random.normal(keys[7],args[0].shape)
    def reference(m,*xs):
      norm=lambda a:a*jax.lax.rsqrt(jnp.mean(a*a,axis=-1,keepdims=True)+1e-6)
      for (content,z,a),tau in zip([xs[:3],xs[3:]],divisors):
        content=norm(content) if tau is None else content/tau
        m=m+jnp.einsum('btnv,btnk->btvk',norm(a),jax.nn.sigmoid(z)[...,None]*content)
      return m
    def fused(m,*xs):return bam_pallas_v8.write_tm(m,[xs[:3],xs[3:]],epsilon=1e-6,content_divisors=divisors,interpret=True)
    np.testing.assert_allclose(fused(*args),reference(*args),rtol=3e-5,atol=3e-5)
    gn=jax.grad(lambda *x:jnp.sum(reference(*x)*target),argnums=tuple(range(7)))(*args)
    gk=jax.grad(lambda *x:jnp.sum(fused(*x)*target),argnums=tuple(range(7)))(*args)
    worst=max(float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12)) for a,b in zip(gn,gk));print('RAW_CONTENT_KERNEL_GRAD',divisors,worst,flush=True);self.assertLess(worst,1e-5)

  def test_fused_all_raw_writer_gradients(self):self.kernel_case((.1946289063,.0045444682))
  def test_fused_attention_only_raw_writer_gradients(self):self.kernel_case((.1946289063,None))
  def test_fused_mlp_only_raw_writer_gradients(self):self.kernel_case((None,.0045444682))

  def test_full_model_raw_all_gradient_paths(self):StandardValueTest.check_pallas(self,True)
  def test_full_model_raw_attention_all_gradient_paths(self):
    self.experiment=ATTENTION;StandardValueTest.check_pallas(self,True)
  def test_full_model_raw_mlp_all_gradient_paths(self):
    self.experiment=MLP;StandardValueTest.check_pallas(self,True)

  def test_same_old_random_parameters_and_all_layer_scale_gradients(self):
    c=self.tiny();oldc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_common_vo_read_scale=False,bam_embedding_write_content_divisor=None,bam_attention_write_content_divisor=None,bam_mlp_write_content_divisor=None)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quant=None);old=Transformer(oldc,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(7),tok,pos,seg,tok));q=nn.unbox(old.init(jax.random.key(7),tok,pos,seg,tok))
      a,b=flatten_dict(p),flatten_dict(q)
      self.assertEqual(sum(v.size for k,v in a.items() if k not in b),3)
      for path,v in b.items():np.testing.assert_array_equal(a[path],v,err_msg=str(path))
      call=lambda z:model.apply(z,tok,pos,seg,tok,enable_dropout=False)[0]
      grad=jax.grad(lambda z:jnp.mean(call(z)**2))(p)
      selected=[v for k,v in flatten_dict(grad).items() if k[-1]=='common_vo_read_scale']
      self.assertEqual(sum(v.size for v in selected),3)
      for v in selected:self.assertTrue(bool(jnp.all(jnp.isfinite(v)&(v!=0))))

class AttentionWriteOnlyScaleTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,*args,**kwargs):
    self.experiment=WRITE_ONLY
    return CommonReadRawWriteTest.tiny(self,*args,**kwargs)

  def test_formal17_live_scalars_and_no_dead_tail(self):
    c=self.config(WRITE_ONLY);c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    scales={p:v for p,v in flat.items() if p[-1]=='attention_write_scale'}
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112209)
    self.assertEqual(sum(math.prod(v.shape) for v in scales.values()),17)
    self.assertFalse(any('final_local_layer' in p for p in scales))
    self.assertFalse(any(p[-1]=='common_vo_read_scale' for p in flat))

  def test_full_model_write_only_all_gradient_paths(self):StandardValueTest.check_pallas(self,True)

  def test_write_only_does_not_scale_vector_update_at_fixed_m(self):
    from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
    c,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=WRITE_ONLY)
    flat=flatten_dict(p);path=next(path for path in flat if path[-1]=='attention_write_scale')
    def call(scale):
      changed=dict(flat);changed[path]=jnp.reshape(scale,(1,));return module.apply(unflatten_dict(changed),*args,M_in=matrix,deterministic=True)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      baseline,mbase=call(jnp.float32(1.));delta=mbase-matrix
      for scale in [0.,-.5,2.]:
        actual,mout=call(jnp.float32(scale))
        np.testing.assert_array_equal(actual,baseline)
        np.testing.assert_allclose(mout-matrix,scale*delta,rtol=3e-5,atol=2e-6)
      grad=jax.grad(lambda scale:jnp.sum(call(scale)[0]))(jnp.float32(.7));self.assertEqual(float(grad),0.)
      target=jax.random.normal(jax.random.key(74),matrix.shape)
      grad=jax.grad(lambda scale:jnp.sum((call(scale)[1]-matrix)*target))(jnp.float32(.7))
      np.testing.assert_allclose(grad,jnp.sum(delta*target),rtol=2e-5,atol=3e-6)

if __name__=='__main__':unittest.main()

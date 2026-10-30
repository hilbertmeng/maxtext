"""Separate complete V/O scales, per-layer or per-head: scopes, init, budget and gradients."""
import contextlib
import math
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import pyconfig
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from layers.attentions import BamAttention
from layers.models import Transformer

P = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75'
H = P + 'CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile'
LAYER = P + 'IndependentVOReadScaleRawMLPEmbeddingFixedTruePile'
HEAD = P + 'IndependentVOReadScalePerHeadRawMLPEmbeddingFixedTruePile'


class IndependentVOReadScaleTest(MLPWriteTest, unittest.TestCase):
  tiny = CommonAllReadGammaTest.tiny

  def test_budget_scan_initial_values_initial_function_and_health(self):
    initial = []
    for run, heads, count in [(LAYER,1,432112228), (HEAD,16,432112768)]:
      c = self.config(run)
      self.assertEqual(c.compare_runs,[H])
      self.assertFalse(c.bam_common_vo_read_scale)
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertFalse(c.bam_local_o_dynamic_norm_scale or c.bam_local_v_dynamic_norm_scale)
      c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),count)
      self.assertFalse(any(p[-1]=='common_vo_read_scale' for p in flat))
      for arm in ('v','o'):
        self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if p[-1]==arm+'_read_scale'),18*heads)
      cfg=self.tiny(run)
      cfg.get_keys().update(num_decoder_layers=18,bam_layer_modes=['local_qk+local_v+local_o']*18,
          bam_attention_write_layers=list(range(17)),bam_mlp_write_layers=list(range(17)))
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
      tok=jnp.array([[1,2,3,4]],jnp.int32)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=nn.unbox(model.init(jax.random.key(1),tok,jnp.arange(4)[None],tok,jnp.ones_like(tok)))
        logits=model.apply(params,tok,jnp.arange(4)[None],tok,jnp.ones_like(tok))[0]
      for path,value in flatten_dict(params).items():
        if path[-1] not in ('v_read_scale','o_read_scale'):continue
        a=np.asarray(jnp.moveaxis(value,cfg.param_scan_axis,0) if 'layers' in path else value[None])
        expected=1/np.sqrt(1+2*np.arange(17)) if 'layers' in path else np.array([1/np.sqrt(35)])
        np.testing.assert_allclose(a,np.broadcast_to(expected[:,None],a.shape),rtol=1e-6)
      metrics=train.compute_params_norm(params,cfg);signed={k:v for k,v in metrics.items() if k.startswith('parameter_values/')}
      self.assertEqual(len(signed),36)
      self.assertEqual(sum(k.endswith('_head_mean') for k in signed),36 if run==HEAD else 0)
      wd=flatten_dict(train.get_wd_tree(cfg,params));self.assertTrue(all(v==0. for p,v in wd.items() if p[-1] in ('v_read_scale','o_read_scale')))
      initial.append(logits)
      print('INDEPENDENT_VO_BUDGET_INIT_HEALTH_OK',run,flush=True)
    np.testing.assert_array_equal(initial[0],initial[1])
    shared=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),
        bam_independent_vo_read_scale=False,bam_independent_vo_read_scale_per_head=False,
        bam_common_vo_read_scale=True)))
    reference=Transformer(shared,mesh,quant=None)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      q=nn.unbox(reference.init(jax.random.key(1),tok,jnp.arange(4)[None],tok,jnp.ones_like(tok)))
      expected=reference.apply(q,tok,jnp.arange(4)[None],tok,jnp.ones_like(tok))[0]
    np.testing.assert_allclose(initial[0],expected,rtol=1e-6,atol=1e-7)

  def test_dynamic_and_static_complete_scope_all_gradients(self):
    for run in (LAYER,HEAD):
      cfg,mesh,model,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=run)
      refcfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_independent_vo_read_scale=False,
          bam_independent_vo_read_scale_per_head=False)))
      ref=BamAttention(config=refcfg,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,
          read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=0,max_target_length=4,
          max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=cfg.dtype)
      flat=flatten_dict(p)
      for i,name in enumerate(('static_v_key','static_o_key')):
        path=next(k for k in flat if k[-1]==name);flat[path]=.1*jax.random.normal(jax.random.key(20+i),flat[path].shape)
      vp=next(k for k in flat if k[-1]=='v_read_scale');op=next(k for k in flat if k[-1]=='o_read_scale')
      flat[vp]=jnp.linspace(-.4,.7,flat[vp].size);flat[op]=jnp.linspace(.3,1.1,flat[op].size)
      p=unflatten_dict(flat);target=jax.random.normal(jax.random.key(22),(4,4,150));mtarget=jax.random.normal(jax.random.key(23),matrix.shape)
      def loss(reference,p,m,x):
        f=flatten_dict(p);sv=f[vp][0] if run==LAYER else f[vp].reshape(1,1,2,1);so=f[op][0] if run==LAYER else f[op].reshape(1,1,2,1)
        def intercept(next_fun,call_args,kwargs,ctx):
          out=next_fun(*call_args,**kwargs)
          if ctx.method_name=='_independent_local_vo':return out[0]*sv,out[1]*so
          if ctx.method_name=='_static_column' and call_args[1] in ('v','o'):
            return out*(sv if call_args[1]=='v' else so)
          return out
        with nn.intercept_methods(intercept) if reference else contextlib.nullcontext():
          y,mout=(ref if reference else model).apply(p,x,x,args[2],args[3],M_in=m,deterministic=True)
        return jnp.mean(y*target)+jnp.mean(mout*mtarget),(y,mout)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        (_,a),ga=jax.value_and_grad(lambda p,m,x:loss(False,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
        (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(True,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
      errors=[float(jnp.max(jnp.abs(x-y))/(jnp.max(jnp.abs(y))+1e-12)) for x,y in zip(jax.tree.leaves((a,ga)),jax.tree.leaves((b,gb)))]
      self.assertLess(max(errors),1e-3)
      self.assertTrue(bool(jnp.all(flatten_dict(ga[0])[vp]!=0)))
      self.assertTrue(bool(jnp.all(flatten_dict(ga[0])[op]!=0)))
      print('INDEPENDENT_FULL_VO_STATIC_DYNAMIC_GRADS',run,max(errors),flush=True)

  def test_layer_native_pallas_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,LAYER)

  def test_head_native_pallas_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,HEAD)

  def test_head_bf16_native_pallas_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,HEAD,dtype='bfloat16',bf16_atol=1e-7)


if __name__=='__main__':unittest.main()

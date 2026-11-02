"""Write-adapter placement: independent norm-order oracle, identity and fused gradients."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_content_adapter_test import AdapterFusedTest,PREFIX,PARENT as UNIT_H
from bam_standard_value_test import StandardValueTest
from layers.models import Transformer
from layers import fusion
from common_types import MODEL_MODE_TRAIN
from layers.attentions import BamAttention
from layers import initializers
CASES=[('ContentWriteAdapterFull',432112210,(2494,1119,3904)),('ContentWriteOAdapterR16',432113410,(2496,1121,3888))]

class PreRmsTest(MLPWriteTest,unittest.TestCase):
  tiny=AdapterFusedTest.tiny
  def test_parameter_tree_budget_controls(self):
    for variant,total,widths in CASES:
      a=self.config(PREFIX+variant+'PreRmsTruePile');b=self.config(PREFIX+variant+'TruePile')
      for c in (a,b):c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(a),a.mesh_axes);aa,*_=train_compile.get_shaped_inputs(mesh,a);bb,*_=train_compile.get_shaped_inputs(mesh,b);fa=flatten_dict(nn.unbox(aa[0].params));fb=flatten_dict(nn.unbox(bb[0].params));self.assertEqual({p:v.shape for p,v in fa.items()},{p:v.shape for p,v in fb.items()});self.assertEqual(sum(math.prod(v.shape) for v in fa.values()),total)
      self.assertEqual((a.mlp_dim,a.bam_memory_mlp_dim,a.bam_last_mlp_dim),widths);self.assertTrue(a.bam_content_adapter_write_before_rms);self.assertTrue(a.bam_pallas_core and a.bam_pallas_write and a.bam_pallas_fused_inputs);self.assertIsNone(a.bam_attention_write_content_divisor);self.assertEqual(a.bam_mlp_write_content_divisor,1.)
      self.assertEqual(a.compare_runs,[UNIT_H]);self.assertTrue(a.record_training_health_metrics and not a.record_internal_nn_metrics);self.assertEqual(a.tensorboard_gcs_rotate_period,500);self.assertNotIn('bam_mlp_dim_by_layer',a.get_keys())
      print('PRE_RMS_BUDGET',variant,total,widths,flush=True)

  def test_initial_identity_and_independent_norm_order(self):
    for variant,_,_ in CASES:
      self.variant=variant+'PreRms';a=self.tiny(length=8);self.variant=variant;b=self.tiny(length=8)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(a),a.mesh_axes);pre=Transformer(a,mesh,quant=None);post=Transformer(b,mesh,quant=None);tok=jnp.array([[1,2,3,4,5,6,7,8]],jnp.int32);pos=jnp.arange(8)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(a.logical_axis_rules):
        args=(tok,pos,tok,seg,seg);p=nn.unbox(pre.init(jax.random.key(41),*args,enable_dropout=False));q=nn.unbox(post.init(jax.random.key(41),*args,enable_dropout=False))
        self.assertEqual(jax.tree.structure(p),jax.tree.structure(q))
        for x,y in zip(jax.tree.leaves(p),jax.tree.leaves(q)):np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(pre.apply(p,*args,enable_dropout=False)[0],post.apply(q,*args,enable_dropout=False)[0],rtol=1e-6,atol=1e-6)
        # Inspect an unscanned SubDecoderLayer: capturing arrays from a remat/scan
        # would leak tracers rather than expose an eager normalization result.
        layer=fusion.SubDecoderLayer(a,mesh,quant=None,layer_inx=0)
        x=jax.random.normal(jax.random.key(42),(1,8,150))
        matrix=jax.random.normal(jax.random.key(43),(1,8,75,32))
        layer_args=(x,seg,pos,seg,None,True,MODEL_MODE_TRAIN,None)
        p=nn.unbox(layer.init(jax.random.key(44),*layer_args,M_in=matrix,is_global=True))
        flat=flatten_dict(p)
        for path,v in list(flat.items()):
          if 'content_write_adapter' not in path:continue
          if path[-1]=='delta':value=jnp.eye(75,dtype=v.dtype)
          elif path[-1]=='down':value=jnp.eye(75,16,dtype=v.dtype)
          else:value=jnp.eye(16,75,dtype=v.dtype)
          flat[path]=jnp.broadcast_to(value,v.shape)
        p=unflatten_dict(flat);seen=[]
        def capture(next_fun,args,kwargs,ctx):
          out=next_fun(*args,**kwargs)
          if isinstance(ctx.module,BamAttention) and ctx.method_name=='_attention_write_content':seen.append((args[0],out))
          return out
        with nn.intercept_methods(capture):layer.apply(p,*layer_args,M_in=matrix,is_global=True)
        self.assertEqual(len(seen),1)
        eps=initializers.get_bam_write_epsilon(a)
        norm=lambda c:c/jnp.sqrt(jnp.mean(c*c,axis=-1,keepdims=True)+eps)
        adapter=(lambda c:2*c) if variant.endswith('Full') else (lambda c:jnp.concatenate((2*c[...,:16],c[...,16:]),-1))
        for raw,out in seen:
          np.testing.assert_allclose(out,norm(adapter(raw)),rtol=2e-6,atol=2e-6)
          self.assertGreater(float(jnp.max(jnp.abs(out-adapter(norm(raw))))),1e-4)

  @unittest.skipUnless(jax.device_count()==8,'launch separately on8 virtual CPU devices')
  def test_formal_eight_device_sharding(self):
    for variant,_,_ in CASES:
      c=self.config(PREFIX+variant+'PreRmsTruePile');c.get_keys().update(max_target_length=256,max_prefill_predict_length=256);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);a,_,s,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(a[0].params));parts=flatten_dict(nn.unbox(s.params));total=sum(math.prod(v.shape) for v in flat.values());local=sum(math.prod(parts[p].shard_shape(v.shape)) for p,v in flat.items());self.assertLess(local/(total/8)-1,c.sharding_tolerance)
      for p,v in flat.items():
        if any(str(k).startswith('content_') and str(k).endswith('_adapter') for k in p):self.assertEqual(math.prod(parts[p].shard_shape(v.shape)),math.prod(v.shape)//8)
      print('PRE_RMS_SHARDING',variant,local/(total/8)-1,flush=True)

class PreRmsFusedTest(MLPWriteTest,unittest.TestCase):
  tiny=AdapterFusedTest.tiny
  def test_full_all_gradients(self):
    self.variant='ContentWriteAdapterFullPreRms';StandardValueTest.check_pallas(self,True)
  def test_write_o_r16_all_gradients(self):
    self.variant='ContentWriteOAdapterR16PreRms';StandardValueTest.check_pallas(self,True)

if __name__=='__main__':unittest.main()

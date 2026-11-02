"""Fast-kernel ShortTrain adapters: budget, head-independent math, large-FSDP layout and gradients."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils
from layers.models import Transformer
from layers.bam_content_adapter import HeadContentAdapter,relative_update_norm,parameter_health
from bam_fast_kernel_test import FastKernelTest

BASE='BamPythia2p8BHUnitShortTrainFastKernel'
FULL=BASE+'ContentWriteAdapterFull'
WO=BASE+'ContentWriteOAdapterR16'
CASES=[(FULL,0,8704,6348800,2774984832,31),(WO,16,8960,5160960,2775107968,63)]

class PythiaAdapterTest(FastKernelTest):
  def test_formal_parameters_and_sharding(self):
    for name,rank,tail,extra,total,metrics in CASES:
      c=self.config(name);parent=self.config(BASE)
      changed={k for k,v in c.get_keys().items() if v!=parent.get_keys().get(k)}
      allowed={'exp_class','model_name','compare_runs','bam_content_adapter_write','bam_content_adapter_o','bam_content_adapter_v','bam_content_adapter_rank','bam_content_adapter_flatten_params','bam_last_mlp_dim'}
      self.assertLessEqual(changed,allowed)
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(7808,2944,tail));self.assertTrue(all(x%128==0 for x in (c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim)))
      self.assertEqual((c.steps,c.learning_rate_schedule_steps,c.learning_rate,c.warmup_steps_fraction),(28000,28000,.0002,.02));self.assertIsNone(c.bam_qk_logit_scale_dim)
      self.assertTrue(c.bam_pallas_packed_read_keys and c.bam_pallas_fused_qk_norm and c.bam_pallas_core and c.bam_pallas_write)
      self.assertEqual((c.bam_pallas_vmem_mib,c.remat_policy),(63,'minimal_save_splash'));self.assertEqual(c.compare_runs,[BASE]);self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      c.get_keys().update(max_target_length=8,max_prefill_predict_length=8,bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);tokens=jnp.ones((1,8),jnp.int32)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=jax.eval_shape(lambda k:Transformer(c,mesh,quant=None).init(k,tokens,jnp.arange(8)[None],tokens,tokens,enable_dropout=False)['params'],jax.random.key(71))
        spec=flatten_dict(nn.logical_to_mesh(nn.get_partition_spec(p)))
      flat=flatten_dict(nn.unbox(p));self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),total)
      adapters={k:v for k,v in flat.items() if any(str(t).startswith('content_') and str(t).endswith('_adapter') for t in k)};self.assertEqual(sum(math.prod(v.shape) for v in adapters.values()),extra)
      for path,v in adapters.items():
        axes=spec[path];has_fsdp=False
        for dim,axis in zip(v.shape,axes):
          if axis=='fsdp' or isinstance(axis,tuple) and 'fsdp' in axis:
            has_fsdp=True;self.assertEqual(dim%128,0,(path,v.shape,axes))
        self.assertTrue(has_fsdp,(path,axes))
      for devices in (64,128):
        perchip=sum(math.prod(v.shape)//math.prod([devices for dim,axis in zip(v.shape,spec[path]) if axis=='fsdp' or isinstance(axis,tuple) and 'fsdp' in axis]) for path,v in flat.items());overhead=perchip/(total/devices)-1
        if devices==64:self.assertLessEqual(overhead,c.sharding_tolerance)
        print('PYTHIA_ADAPTER_SHARDING',name,devices,total,extra,overhead,flush=True)
      self.assertFalse(any('final_local_layer' in '/'.join(k) and 'content_write_adapter' in k for k in adapters))
      self.assertEqual(sum(v.shape[c.param_scan_axis] if 'layers' in path else 1 for path,v in adapters.items() if path[-1] in ('delta','up')),metrics)

  def test_flat_parameters_identity_gradients_and_health(self):
    x=jax.random.normal(jax.random.key(72),(2,3,32,80));target=jax.random.normal(jax.random.key(73),x.shape)
    for rank in (0,16):
      flat=HeadContentAdapter(32,80,rank,dtype=jnp.float32,flatten_params=True);ref=HeadContentAdapter(32,80,rank,dtype=jnp.float32)
      p=nn.unbox(flat.init(jax.random.key(74),x));np.testing.assert_array_equal(flat.apply(p,x),x)
      p=jax.tree.map(lambda a:a+.03*jax.random.normal(jax.random.key(a.size),a.shape),p)
      def shaped(q):
        return {'params':{k:v.reshape((32,80,80) if k=='delta' else (32,80,16) if k=='down' else (32,16,80)) for k,v in q['params'].items()}}
      np.testing.assert_allclose(flat.apply(p,x),ref.apply(shaped(p),x),rtol=1e-6,atol=1e-6)
      loss=lambda a,q,y:jnp.sum(a.apply(q,y)*target)
      g,dx=jax.grad(lambda q,y:loss(flat,q,y),argnums=(0,1))(p,x);rg,rdx=jax.grad(lambda q,y:loss(ref,shaped(q),y),argnums=(0,1))(p,x)
      for a,b in zip(jax.tree.leaves((g,dx)),jax.tree.leaves((rg,rdx))):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
      value=relative_update_norm(p['params'],num_heads=32);np.testing.assert_allclose(value,relative_update_norm(shaped(p)['params']),rtol=2e-6)
      cfg=SimpleNamespace(bam_content_adapter_flatten_params=True,num_query_heads=32,param_scan_axis=1,num_decoder_layers=32,bam_content_adapter_activation='linear')
      tree={'params':{'decoder':{'layers':{'self_attention':{'content_write_adapter':jax.tree.map(lambda a:jnp.stack([a,a],axis=1),p['params'])}},'final_local_layer':{'self_attention':{'content_o_adapter':p['params']}}}}}
      h=parameter_health(tree,cfg);self.assertEqual(len(h),3)
      for v in h.values():np.testing.assert_allclose(v,value,rtol=2e-6)

  def check_adapter(self,name,dtype='float32'):
    # Exercise all adapter factors away from zero and non-unit QKNorm/read scales.
    original=nn.Module.init
    def init(module,*args,**kwargs):
      flat=flatten_dict(nn.unbox(original(module,*args,**kwargs)))
      for i,path in enumerate(flat):
        if any(str(t).startswith('content_') and str(t).endswith('_adapter') for t in path):flat[path]=flat[path]+.03*jax.random.normal(jax.random.key(400+i),flat[path].shape)
      return unflatten_dict(flat)
    self.bf16_training_objective = dtype == 'bfloat16' and name == WO
    with mock.patch.object(nn.Module,'init',init):self.check_model(name,dtype)
  def test_full_all_gradients(self):self.check_adapter(FULL)
  def test_write_o_r16_all_gradients(self):self.check_adapter(WO)
  def test_full_bf16_against_truth(self):self.check_adapter(FULL,'bfloat16')
  def test_write_o_r16_bf16_training_loss(self):self.check_adapter(WO,'bfloat16')

if __name__=='__main__':unittest.main()

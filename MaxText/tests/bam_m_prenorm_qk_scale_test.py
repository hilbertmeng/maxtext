"""Independent QK strength on the retained-V/O M-pre-norm control."""
from types import SimpleNamespace
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
import pyconfig,max_utils,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_m_prenorm_test import MatrixPreNormTest,KEEP
from layers.models import Transformer
from layers import attentions

RUN=KEEP.removesuffix('TruePile')+'QKScaleTruePile'

class MatrixQKScaleTest(MLPWriteTest,unittest.TestCase):
  tiny=MatrixPreNormTest.tiny
  model=MatrixPreNormTest.model
  inputs=MatrixPreNormTest.inputs

  def test_formal_budget_and_initial_unit_equivalence(self):
    cfg=self.config(RUN);self.assertTrue(cfg.bam_common_vo_read_scale and cfg.bam_common_qk_read_scale)
    self.assertFalse(cfg.qk_norm)
    self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_write and cfg.bam_pallas_packed_read_keys)
    self.assertEqual(cfg.compare_runs,[KEEP])
    cfg.get_keys().update(max_target_length=128,max_prefill_predict_length=128)  # kernel tiles need T>=chunk
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    shaped,*_=train_compile.get_shaped_inputs(mesh,cfg)
    flat=flatten_dict(nn.unbox(shaped[0].params))
    self.assertEqual(sum(np.prod(v.shape,dtype=np.int64) for v in flat.values()),432112228)
    self.assertEqual(sum(v.size for p,v in flat.items() if p[-1]=='common_qk_read_scale'),18)
    a,b=self.tiny(RUN),self.tiny(KEEP);tok,pos,seg,target=self.inputs()
    with self.model(a) as model,self.model(b) as parent:
      params=nn.unbox(model.init(jax.random.key(14),tok,pos,seg,target))
      pf=flatten_dict(params);base=unflatten_dict({p:v for p,v in pf.items() if p[-1]!='common_qk_read_scale'})
      for p,v in pf.items():
        if p[-1]=='common_qk_read_scale':np.testing.assert_array_equal(v,jnp.ones_like(v))
      loss=lambda m,p:jnp.mean(m.apply(p,tok,pos,seg,target)[0].astype(jnp.float32))
      l,g=jax.value_and_grad(lambda p:loss(model,p))(params)
      r,h=jax.value_and_grad(lambda p:loss(parent,p))(base)
      np.testing.assert_allclose(l,r,rtol=1e-6,atol=1e-7)
      for p,v in flatten_dict(h).items():np.testing.assert_allclose(flatten_dict(g)[p],v,rtol=2e-5,atol=1e-6)
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for p,v in flatten_dict(g).items() if p[-1]=='common_qk_read_scale'),0)

  def test_only_local_prefix_scaled_rope_unchanged(self):
    cfg=self.tiny(RUN);tok,pos,seg,target=self.inputs();orig=attentions.BamAttention._add_local_qk
    def measured(module,q,k,lq,lk,*args,**kwargs):
      module.sow('intermediates','qk_scale_probe',jnp.stack((jnp.concatenate((lq,q),-1),jnp.concatenate((lk,k),-1))))
      return orig(module,q,k,lq,lk,*args,**kwargs)
    with self.model(cfg) as model:
      params=nn.unbox(model.init(jax.random.key(11),tok,pos,seg,target))
      doubled=jax.tree_util.tree_map_with_path(lambda p,v:v*2 if p[-1].key=='common_qk_read_scale' else v,params)
      with mock.patch.object(attentions.BamAttention,'_add_local_qk',measured):
        _,x=model.apply(params,tok,pos,seg,target,mutable=['intermediates'])
        _,y=model.apply(doubled,tok,pos,seg,target,mutable=['intermediates'])
      def first(d):
        for p,v in flatten_dict(d['intermediates']).items():
          if p[-1]=='qk_scale_probe' and 'layers' in p:
            while isinstance(v,tuple):v=v[0]
            return np.asarray(v[0])
        raise AssertionError('Missing scanned QK probe')
      before,after=first(x),first(y)
      np.testing.assert_allclose(after[...,:75],before[...,:75]*2,rtol=1e-6,atol=1e-7)
      np.testing.assert_array_equal(after[...,75:],before[...,75:])

  def test_no_qknorm_pallas_scale_bridge_all_gradients(self):
    # Isolate the pre-existing read-kernel bridge; whole-M norm remains delegated.
    cfg=self.tiny(RUN);cfg.get_keys().update(bam_m_read_norm='none',max_target_length=128,
        max_prefill_predict_length=128,query_chunk_size=64)
    pcfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_pallas_core=True,
        bam_pallas_write=True,bam_pallas_interpret=True,bam_splash_attention=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    native,fused=Transformer(cfg,mesh,quant=None),Transformer(pcfg,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(7),(1,128),0,32);pos=(jnp.arange(128)%64)[None]
    seg=jnp.where(jnp.arange(128)<64,1,2)[None]
    make=splash.make_splash_mha
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=lambda *a,**k:make(*a,**dict(k,interpret=True))):
      params=nn.unbox(native.init(jax.random.key(10),tok,pos,seg,tok))
      params=jax.tree_util.tree_map_with_path(lambda p,v:v*.6 if p[-1].key=='common_qk_read_scale' else v,params)
      def loss(m,p):return jnp.mean(m.apply(p,tok,pos,seg,tok)[0].astype(jnp.float32))
      l,g=jax.value_and_grad(lambda p:loss(native,p))(params)
      r,h=jax.value_and_grad(lambda p:loss(fused,p))(params)
      np.testing.assert_allclose(l,r,rtol=1e-5,atol=1e-6)
      for p,v in flatten_dict(g).items():np.testing.assert_allclose(v,flatten_dict(h)[p],rtol=3e-4,atol=2e-6,err_msg=str(p))

if __name__=='__main__':unittest.main()

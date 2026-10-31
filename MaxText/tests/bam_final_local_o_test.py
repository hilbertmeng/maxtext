"""Final M exit: independent reference, write dependency, layout and exact budgets."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.bam_final_local_o import FinalLocalO
from layers.models import Transformer

PREFIX='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixed'
H=PREFIX+'TruePile'
OFF=PREFIX+'FinalLocalOTruePile'
ON=PREFIX+'FinalLocalOLastMLPWriteTruePile'

class FinalLocalOTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, run=OFF, length=4, dtype='float32'):
    c=self.config(run,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
      base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype=dtype,weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,
      bam_memory_mlp_dim=16,bam_last_mlp_dim=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
      bam_attention_write_layers=[0,1],bam_mlp_write_layers=None,emb_bam_num_head=2,
      bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
      bam_local_full_m_read_bottleneck_dim=8,max_target_length=length,max_prefill_predict_length=length,
      query_chunk_size=length)
    return c

  def test_formal_budget_and_scope(self):
    for run,width,expected in ((H,3904,432112210),(OFF,3371,432112339),(ON,2116,432113139)):
      c=self.config(run);c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),expected)
      self.assertEqual(c.bam_last_mlp_dim,width)
      self.assertEqual(c.mlp_dim,2510);self.assertEqual(c.bam_memory_mlp_dim,1128)
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      self.assertNotIn(17,c.bam_attention_write_layers)
      tail_memory=[p for p in flat if 'final_local_layer' in p and 'memory_mlp' in p]
      self.assertEqual(bool(tail_memory),run==ON)
      exits=[v for p,v in flat.items() if 'final_local_o' in p]
      self.assertEqual(sum(math.prod(v.shape) for v in exits),1918929 if run!=H else 0)

  def test_explicit_generalized_read_and_all_gradients(self):
    c=self.tiny();module=FinalLocalO(c);x=jax.random.normal(jax.random.key(1),(1,4,150))
    matrix=jax.random.normal(jax.random.key(2),(1,4,75,32))
    variables=nn.unbox(module.init(jax.random.key(3),x,matrix));p=variables['params']
    # Both channels and both gates must contribute away from semantic zero initializers.
    p['static_o_key']=jax.random.normal(jax.random.key(4),(32,2))*.1
    p['static_gate']['kernel']=jax.random.normal(jax.random.key(5),(150,2))*.01
    p['dynamic_gate']['kernel']=jax.random.normal(jax.random.key(6),(150,2,1))*.01
    def reference(p,x,m):
      norm=x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+c.normalization_layer_epsilon)*(1+p['input_norm']['scale'])
      key=nn.gelu(norm@p['read_down']['kernel'])
      key=jnp.einsum('btr,rnv->btnv',key,p['read_up']['kernel'])+p['pre_rms_bias']
      key=key*jax.lax.rsqrt(jnp.mean(key*key,axis=-1,keepdims=True)+c.bam_read_key_epsilon)
      gate=jax.nn.sigmoid(jnp.einsum('btd,dns->btns',norm,p['dynamic_gate']['kernel'])+p['dynamic_gate_bias'])
      dynamic=jnp.einsum('btkv,btnv->btnk',m,key*gate*c.bam_read_key_scale)
      sg=1.1*jax.nn.sigmoid(norm@p['static_gate']['kernel']+p['static_gate_bias'])/c.bam_static_read_gate_init
      static=jnp.einsum('btkv,vn->btnk',m,p['static_o_key'])*sg[...,None]
      return x+jnp.einsum('btnk,nkd->btd',(dynamic+static)*p['common_vo_read_scale'][0],p['out']['kernel'])
    target=jax.random.normal(jax.random.key(7),x.shape)
    def loss(fn,p,x,m):
      y=fn(p,x,m);return jnp.sum(y*target),y
    actual=lambda p,x,m:module.apply({'params':p},x,m)
    (_,a),ga=jax.value_and_grad(lambda p,x,m:loss(actual,p,x,m),argnums=(0,1,2),has_aux=True)(p,x,matrix)
    (_,b),gb=jax.value_and_grad(lambda p,x,m:loss(reference,p,x,m),argnums=(0,1,2),has_aux=True)(p,x,matrix)
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
    for u,v in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):np.testing.assert_allclose(u,v,rtol=5e-4,atol=3e-6)
    np.testing.assert_allclose(p['common_vo_read_scale'],[1/math.sqrt(7)],rtol=1e-6)
    self.assertGreater(float(jnp.linalg.norm(ga[2])),0)
    # Existing signed health exporter includes this extra position; its scale gets no WD.
    metrics=train.compute_params_norm({'params':{'decoder':{'final_local_o':{'common_vo_read_scale':jnp.array([-.2])}}}},c)
    self.assertTrue(any(float(v)<0 for k,v in metrics.items() if k.startswith('parameter_values/')))
    wd=train.get_wd_tree(c,{'params':{'decoder':{'final_local_o':p}}})
    self.assertEqual(wd['params']['decoder']['final_local_o']['common_vo_read_scale'],0)
    pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True)))
    packed=FinalLocalO(pc).apply({'params':p},x,jnp.swapaxes(matrix,-1,-2))
    np.testing.assert_array_equal(packed,a)

  def test_last_memory_writer_is_live_only_when_restored(self):
    for run in (OFF,ON):
      c=self.tiny(run);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32)
      pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(20),tok,pos,seg,tok))
        def call(p,tok):return model.apply(p,tok,pos,seg,tok)[0]
        y=call(p,tok);y2=call(p,tok.at[:,3].set(5))
        np.testing.assert_allclose(y[:,:3],y2[:,:3],rtol=2e-5,atol=1e-6)
        value,g=jax.value_and_grad(lambda p:jnp.mean(call(p,tok)**2))(p)
      self.assertTrue(np.isfinite(float(value)))
      flat=flatten_dict(g)
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
      last=[(k,v) for k,v in flat.items() if 'final_local_layer' in k and 'memory_mlp' in k]
      self.assertEqual(bool(last),run==ON)
      for path,value in last:self.assertGreater(float(jnp.sum(value**2)),0,str(path))
      self.assertFalse(any('final_local_layer' in k and 'self_attention' in k and ('P_loc_down' in k or 'W_gw' in k) for k in flat))
      self.assertGreater(float(jnp.sum(g['params']['decoder']['final_local_o']['out']['kernel']**2)),0)

  def test_pallas_scan_full_model_logits_and_all_gradients(self):
    for run in (OFF,ON):
      c=self.tiny(run,length=128)
      pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_write=True,bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      native,fused=Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
      tok=jax.random.randint(jax.random.key(30),(1,128),0,32);pos=(jnp.arange(128)%64)[None]
      seg=jnp.where(jnp.arange(128)<64,1,2)[None]
      original=splash.make_splash_mha
      def interpret(*args,**kwargs):return original(*args,**dict(kwargs,interpret=True))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
        p,pp=(nn.unbox(m.init(jax.random.key(31),tok,pos,seg,tok)) for m in (native,fused))
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
        def loss(m,p):
          y=m.apply(p,tok,pos,seg,tok)[0];return jnp.mean(y*y),y
        (_,a),ga=jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
        (_,b),gb=jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(p)
        np.testing.assert_allclose(a,b,rtol=4e-4,atol=3e-6)
        for path,a in flatten_dict(ga).items():
          b=flatten_dict(gb)[path]
          np.testing.assert_allclose(a,b,rtol=3e-3,atol=4e-6,err_msg=str(path))
      print('FINAL_LOCAL_O_NATIVE_PALLAS_ALL_GRADIENTS_OK',run,flush=True)

if __name__=='__main__':unittest.main()

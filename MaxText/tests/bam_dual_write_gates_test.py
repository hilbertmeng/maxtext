"""Split attention writer: shared denominator/address, independent .1 gates."""
import math
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from layers import bam_write, bam_pallas
from layers.models import Transformer

PARENT=('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
        'LogitDiagNegativeQK75LocalODynamicNormScale050TruePile')
RUN=PARENT.replace('050TruePile','050DualWriteGatesTruePile')


class DualWriteGatesTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=3, keep=True, length=4):
    c=self.config(RUN,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32',bam_splash_attention=False)
    c.get_keys().update(bam_pallas_core=False,mlp_dim=32,bam_last_mlp_dim=32,
        bam_memory_mlp_dim=16,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],
        bam_write_v_bottleneck_dim=8,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
        max_target_length=length,max_prefill_predict_length=length,query_chunk_size=length)
    return c

  def test_explicit_shared_denominator_and_all_gradients(self):
    a=jax.random.normal(jax.random.key(1),(1,4,2,5));o=2*jax.random.normal(jax.random.key(2),a.shape)
    za=jax.random.normal(jax.random.key(3),a.shape[:-1]);zo=jax.random.normal(jax.random.key(4),za.shape)
    p=jax.random.normal(jax.random.key(5),(1,4,2,3));m=jnp.ones((1,4,3,5))*.01;target=jax.random.normal(jax.random.key(6),m.shape)
    def actual(a,o,za,zo,p):
      return bam_write.native_write(m,[bam_write.SplitAttentionWrite(a,o,za,zo,p)],epsilon=1e-6)[0]
    def reference(a,o,za,zo,p):
      total=a+o;r=jnp.sqrt(jnp.mean(total**2,axis=-1,keepdims=True)+1e-6)
      content=(jax.nn.sigmoid(za)[...,None]*a+jax.nn.sigmoid(zo)[...,None]*o)/r
      address=p/jnp.sqrt(jnp.mean(p**2,axis=-1,keepdims=True)+1e-6)
      return m+jnp.einsum('btnv,btnk->btvk',address,content)
    args=(a,o,za,zo,p)
    np.testing.assert_allclose(actual(*args),reference(*args),rtol=2e-6,atol=2e-6)
    ga=jax.grad(lambda *x:jnp.sum(actual(*x)*target),argnums=(0,1,2,3,4))(*args)
    gr=jax.grad(lambda *x:jnp.sum(reference(*x)*target),argnums=(0,1,2,3,4))(*args)
    for left,right in zip(ga,gr):np.testing.assert_allclose(left,right,rtol=2e-5,atol=2e-6)
    # Equal zero logits: sum has gate .5, not two separately normalized writes.
    zero=jnp.zeros_like(za);group=bam_write.SplitAttentionWrite(a,o,zero,zero,p)
    content,_=bam_write.split_content(group,1e-6)
    np.testing.assert_allclose(content,.5*(a+o)/jnp.sqrt(jnp.mean((a+o)**2,-1,keepdims=True)+1e-6),rtol=2e-6,atol=2e-6)

  def test_formal_budget_and_no_dead_terminal_gate(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    shaped,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(shaped[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432121408-928)
    extra={p:v for p,v in flat.items() if 'W_gw_feedback' in p or p[-1]=='feedback_gw_b0'}
    self.assertEqual(sum(math.prod(v.shape) for v in extra.values()),17*(1200*16+16))
    self.assertFalse(any('final_local_layer' in p for p in extra))
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2505,1128,3898))
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_body=='v8t')
    self.assertTrue(c.scan_layers and not c.bam_pair_scan)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    self.assertEqual(c.compare_runs,[PARENT])
    print('DUAL_WRITE_FORMAL_BUDGET_OK',432120480,'delta-928 extra326672',flush=True)

  def test_independent_initialization_and_residual_unchanged(self):
    c,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN,initial=.5)
    main=np.asarray(p['params']['W_gw']['kernel']);feedback=np.asarray(p['params']['W_gw_feedback']['kernel'])
    self.assertFalse(np.array_equal(main,feedback));self.assertGreater(float(feedback.std()),0.)
    np.testing.assert_allclose(p['params']['gw_b0'],math.log(.1/.9),rtol=1e-6)
    np.testing.assert_array_equal(p['params']['gw_b0'],p['params']['feedback_gw_b0'])
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      y0,m0=module.apply(p,*args,M_in=matrix,deterministic=True)
      p['params']['feedback_gw_b0']=jnp.ones_like(p['params']['feedback_gw_b0'])*2
      y1,m1=module.apply(p,*args,M_in=matrix,deterministic=True)
    np.testing.assert_array_equal(y0,y1)
    self.assertGreater(float(jnp.max(jnp.abs(m0-m1))),1e-5)

  def test_full_model_pallas_read_native_write_all_gradients(self):
    StandardValueTest.check_pallas(self,True)

  def test_full_model_pallas_read_and_write_all_gradients(self):
    tiny=self.tiny
    def with_write(*a,**k):
      c=tiny(*a,**k);c.get_keys().update(bam_pallas_write=True);return c
    self.tiny=with_write
    StandardValueTest.check_pallas(self,True)

  def test_v8t_split_write_kernel_matches_native_all_gradients(self):
    from layers import bam_pallas_v8
    k=jax.random.split(jax.random.key(30),10);b,t,n,kd,v=2,128,3,16,8
    args=(jax.random.normal(k[0],(b,t,v,kd)),jax.random.normal(k[1],(b,t,n,kd)),
          2*jax.random.normal(k[2],(b,t,n,kd)),jax.random.normal(k[3],(b,t,n)),
          jax.random.normal(k[4],(b,t,n)),jax.random.normal(k[5],(b,t,n,v)),
          jax.random.normal(k[6],(b,t,2,kd)),jax.random.normal(k[7],(b,t,2)),jax.random.normal(k[8],(b,t,2,v)))
    target=jax.random.normal(k[9],args[0].shape)
    groups=lambda x:[bam_write.SplitAttentionWrite(*x[:5]),tuple(x[5:])]
    native=lambda m,*x:bam_write.native_write(m,groups(x),epsilon=1e-6)[0]
    kernel=lambda m,*x:bam_pallas_v8.write_tm(m,groups(x),epsilon=1e-6,interpret=True)
    np.testing.assert_allclose(kernel(*args),native(*args),rtol=2e-5,atol=2e-5)
    gn=jax.grad(lambda *x:jnp.sum(native(*x)*target),argnums=tuple(range(9)))(*args)
    gk=jax.grad(lambda *x:jnp.sum(kernel(*x)*target),argnums=tuple(range(9)))(*args)
    worst=max(float(jnp.max(jnp.abs(l-r))/jnp.max(jnp.abs(l))) for l,r in zip(gn,gk))
    print('DUAL_WRITE_V8T_KERNEL_WORST_GRAD',worst,flush=True);self.assertLess(worst,1e-5)

  def test_actual_scan_health_only_live_writers(self):
    from layers.bam_write import record_health_metrics
    for fused in (False,True):
      c=self.tiny(length=128 if fused else 4)
      c.get_keys().update(bam_pallas_core=fused,bam_pallas_interpret=True)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quant=None);tok=jnp.ones((1,c.max_target_length),jnp.int32)
      pos=jnp.arange(tok.shape[1])[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(7),tok,pos,seg,tok))
        _,health=model.apply(p,tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
      metrics={'scalar':{}};record_health_metrics(metrics,health,c)
      self.assertEqual(len(metrics['scalar']),6)
      self.assertFalse(any('layer_002' in k for k in metrics['scalar']))
      self.assertTrue(all(np.isfinite(v) for v in metrics['scalar'].values()))

  def test_native_three_factor_fallback_matches_existing_reference(self):
    m=jax.random.normal(jax.random.key(8),(1,4,3,5));c=jax.random.normal(jax.random.key(9),(1,4,2,5))
    z=jax.random.normal(jax.random.key(10),(1,4,2));a=jax.random.normal(jax.random.key(11),(1,4,2,3))
    actual,_=bam_write.native_write(m,[(c,z,a)],epsilon=1e-6)
    minor=lambda x:jnp.moveaxis(x,1,-1)
    expected=jnp.moveaxis(bam_pallas.reference_write(minor(m),minor(c),minor(z),minor(a),epsilon=1e-6),-1,1)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)


if __name__=='__main__':unittest.main()

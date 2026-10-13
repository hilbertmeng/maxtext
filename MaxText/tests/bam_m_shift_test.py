"""Causal M shifting, carry semantics, scan gradients and exact budget."""
import copy
import math
import unittest
from unittest import mock
from pathlib import Path
import tempfile
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import train_compile
import train
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention, mix_bam_previous_token, bam_m_shift_health
from layers.models import Transformer

PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
READ='BamMediumPropSharedR256GeneralReadMShiftReadG005TruePile'
PERSIST='BamMediumPropSharedR256GeneralReadMShiftPersistG005TruePile'
SEPARATE='BamMediumPropSharedR256GeneralReadMShiftReadQKVoG050TruePile'

class MShiftTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, mode):
    cfg=self.config(READ if mode=='read' else PERSIST, bam_splash_attention=False,
        base_emb_dim=150, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=3, base_mlp_dim=32, vocab_size=32,
        dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,mlp_dim_by_block=[32,24,32],
        bam_layer_modes=cfg.bam_layer_modes[:3])
    return cfg

  def test_boundaries_column_broadcast_and_nonrecursive_source(self):
    m=jnp.arange(6*3*2,dtype=jnp.float32).reshape(1,6,3,2)
    g=jnp.broadcast_to(jnp.array([.25,.75]),(1,6,2))
    seg=jnp.array([[1,1,2,2,2,0]])
    pos=jnp.array([[0,1,0,1,0,0]])
    y=mix_bam_previous_token(m,g,seg,pos)
    expected=np.asarray(m).copy()
    for t in (1,3): expected[:,t]=np.array([.75,.25])*np.asarray(m[:,t])+np.array([.25,.75])*np.asarray(m[:,t-1])
    np.testing.assert_array_equal(y,expected)
    np.testing.assert_array_equal(mix_bam_previous_token(m,jnp.zeros_like(g)),m)
    all_previous=mix_bam_previous_token(m,jnp.ones_like(g))
    np.testing.assert_array_equal(all_previous[:,1:],m[:,:-1])
    grad=jax.grad(lambda a:jnp.sum(mix_bam_previous_token(a,g,seg,pos)[:,1]))(m)
    np.testing.assert_allclose(grad[:,0],jnp.broadcast_to(jnp.array([.25,.75]),(1,3,2)))
    np.testing.assert_allclose(grad[:,1],jnp.broadcast_to(jnp.array([.75,.25]),(1,3,2)))
    np.testing.assert_array_equal(grad[:,2:],jnp.zeros_like(grad[:,2:]))

  def test_health_masks_and_known_mix(self):
    m=jnp.array([0.,2.,4.,100.],jnp.float32).reshape(1,4,1,1)
    gate=jnp.array([.99,.25,.25,.99],jnp.float32).reshape(1,4,1)
    segments=jnp.array([[1,1,1,2]])
    mixed=mix_bam_previous_token(m,gate,segments)
    stats,columns,state=bam_m_shift_health(m,mixed,gate,segments)
    np.testing.assert_allclose(stats[0],.25)
    np.testing.assert_allclose(columns[0],.25)
    np.testing.assert_allclose(state[4],.5/np.sqrt(10),rtol=1e-6)
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in (stats,columns,state)))
    empty=bam_m_shift_health(m,m,gate,jnp.zeros_like(segments))
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in empty))

  def test_read_and_persist_same_reads_but_distinct_write_bases(self):
    cfg=self.tiny('read')
    mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
    mod=BamAttention(config=cfg,mesh=mesh,num_query_heads=2,num_kv_heads=2,
        head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,
        attention_kernel='dot_product_chunk',dtype=cfg.dtype,weight_dtype=cfg.weight_dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=cfg.attention_type)
    x=jax.random.normal(jax.random.key(1),(1,4,150))
    m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    args=(x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
    mixed=mix_bam_previous_token(m,jnp.full((1,4,32),cfg.bam_m_shift_init))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,deterministic=True)['params'])
      np.testing.assert_array_equal(params['M_shift_gate']['kernel'],jnp.zeros((150,32)))
      np.testing.assert_allclose(jax.nn.sigmoid(params['M_shift_gate_b0']),.05,rtol=1e-6)
      a=mod.apply({'params':params},*args,M_in=m,deterministic=True,defer_write=True)
      cfg.get_keys()['bam_m_shift_mode']='persist'
      b=mod.apply({'params':params},*args,M_in=m,deterministic=True,defer_write=True)
      np.testing.assert_array_equal(a[0],b[0])
      np.testing.assert_array_equal(a[1],m)
      np.testing.assert_array_equal(b[1],mixed)
      for v,w in zip(a[2],b[2]):np.testing.assert_array_equal(v,w)
      cfg.get_keys()['bam_m_shift_mode']='read'
      a=mod.apply({'params':params},*args,M_in=m,deterministic=True)
      cfg.get_keys()['bam_m_shift_mode']='persist'
      b=mod.apply({'params':params},*args,M_in=m,deterministic=True)
      np.testing.assert_array_equal(a[0],b[0])
      np.testing.assert_allclose(b[1]-a[1],mixed-m,rtol=1e-5,atol=1e-6)
      grad=jax.grad(lambda p:jnp.mean(mod.apply({'params':p},*args,M_in=m,deterministic=True)[0]**2))(params)
      self.assertGreater(float(jnp.linalg.norm(grad['M_shift_gate']['kernel'])),0)

  def test_scan_forward_gradients_and_future_causality(self):
    for mode, opening, separate in [('read',.05,False),('persist',.05,False),
                                     ('read',.5,False),('persist',.5,False),('read',.5,True)]:
      cfg=self.tiny(mode)
      cfg.get_keys()['bam_m_shift_init']=opening
      cfg.get_keys()['bam_m_shift_separate_reads']=separate
      mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None)
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(4),tok,pos,seg,tok)['params'])
        def f(params,tokens=tok):
          out=model.apply({'params':params},tokens,pos,seg,tokens,enable_dropout=False)
          return out[0] if isinstance(out,tuple) else out
        loss,grad=jax.jit(jax.value_and_grad(lambda params:jnp.mean(f(params)**2)))(p)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
        gates=[v for path,v in flatten_dict(grad).items() if any(part.startswith('M_shift_') for part in path)]
        self.assertEqual(len(gates),18 if separate else 6)
        self.assertTrue(all(float(jnp.linalg.norm(v))>0 for v in gates))
        np.testing.assert_allclose(f(p)[:,:3],f(p,tok.at[:,3].set(9))[:,:3],rtol=1e-6,atol=1e-6)
        _,collections=model.apply({'params':p},tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
        metrics={'scalar':{}}
        train.record_bam_concat_health_metrics(metrics,collections,cfg)
        health={k:v for k,v in metrics['scalar'].items() if k.startswith('bam/m_shift/')}
        self.assertEqual(len(health),3*(18 if separate else 64+14))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in health.values()))
        for layer in range(3):
          for arm in ('q/','k/','vo/') if separate else ('',):
            np.testing.assert_allclose(health[f'bam/m_shift/layer_{layer:03d}/{arm}gate_mean'],opening,rtol=1e-5)
        if separate:
          self.assertFalse(any('/column_' in tag for tag in health))
          for pair in ('q_k','q_vo','k_vo'):
            np.testing.assert_allclose(health[f'bam/m_shift/layer_000/{pair}_gate_mean_abs_diff'],0,atol=1e-7)

  def test_separate_routes_equal_shared_at_init_and_keep_original_carry(self):
    cfg=self.tiny('read')
    cfg.get_keys().update(bam_m_shift_init=.5,bam_m_shift_separate_reads=True)
    mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
    mod=BamAttention(config=cfg,mesh=mesh,num_query_heads=2,num_kv_heads=2,
        head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,
        attention_kernel='dot_product_chunk',dtype=cfg.dtype,weight_dtype=cfg.weight_dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=cfg.attention_type)
    x=jax.random.normal(jax.random.key(11),(1,4,150))
    m=jax.random.normal(jax.random.key(12),(1,4,75,32))
    args=(x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params=nn.unbox(mod.init(jax.random.key(13),*args,M_in=m,deterministic=True)['params'])
      separated=mod.apply({'params':params},*args,M_in=m,deterministic=True,defer_write=True)
      cfg.get_keys()['bam_m_shift_separate_reads']=False
      shared=mod.apply({'params':params},*args,M_in=m,deterministic=True,defer_write=True)
      np.testing.assert_array_equal(separated[0],shared[0])
      np.testing.assert_array_equal(separated[1],m)
      for a,b in zip(separated[2],shared[2]):np.testing.assert_array_equal(a,b)
      cfg.get_keys()['bam_m_shift_separate_reads']=True
      views={}
      read_qk=BamAttention._read_direct_qk_c8
      read_vo=BamAttention._independent_local_vo
      def capture_qk(self,arm,state,*rest):
        views[arm]=state
        return read_qk(self,arm,state,*rest)
      def capture_vo(self,state,*rest):
        views['vo']=state
        return read_vo(self,state,*rest)
      for arm,opening in [('q',.2),('k',.7),('vo',.9)]:
        name='M_shift_gate_b0' if arm=='q' else f'M_shift_{arm}_gate_b0'
        params[name]=jnp.full((32,),math.log(opening/(1-opening)),jnp.float32)
      with mock.patch.object(BamAttention,'_read_direct_qk_c8',capture_qk), \
           mock.patch.object(BamAttention,'_independent_local_vo',capture_vo):
        actual=mod.apply({'params':params},*args,M_in=m,deterministic=True,defer_write=True)
      np.testing.assert_array_equal(actual[1],m)
      for arm,opening in [('q',.2),('k',.7),('vo',.9)]:
        expected=mix_bam_previous_token(m,jnp.full((1,4,32),opening),args[3],args[2])
        expected=mod.apply({'params':params},expected,method=mod._matrix_for_read)
        np.testing.assert_allclose(views[arm],expected,rtol=1e-6,atol=1e-6)


  def test_full_parameter_budget_and_gate_sharding(self):
    for name in (PARENT,READ,PERSIST,READ.replace('G005','G050'),PERSIST.replace('G005','G050'),SEPARATE):
      cfg=self.config(name,bam_splash_attention=False)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      args,_,spec,_=train_compile.get_shaped_inputs(mesh,cfg)
      params=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in params.values()),
                       432111104 if name==PARENT else 432112832 if name==SEPARATE else 432111680)
      gates={path:v for path,v in params.items() if any(part.startswith('M_shift_') for part in path)}
      self.assertEqual(sum(math.prod(v.shape) for v in gates.values()),
                       0 if name==PARENT else 2075328 if name==SEPARATE else 691776)
      if name!=PARENT:
        self.assertEqual(len(gates),18 if name==SEPARATE else 6)
        specs=flatten_dict(nn.unbox(spec.params))
        for path,v in gates.items():
          self.assertEqual(v.shape,(1200,6,32) if path[-1]=='kernel' else (32,6))
          if path[-1]=='kernel':
            self.assertLess(math.prod(specs[path].shard_shape(v.shape)),math.prod(v.shape))

if __name__=='__main__':unittest.main()

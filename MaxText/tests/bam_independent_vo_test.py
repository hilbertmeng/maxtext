"""Independent V/O dynamic keys: budget, tied initialization, and fused reverse."""
import copy
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train_compile
from bam_hybrid_value_test import HybridValueTest
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from common_types import MODEL_MODE_TRAIN
from layers import bam_pallas_v8, fusion

RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile'


class IndependentVOTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=2, keep=True, length=4):
    c = HybridValueTest.tiny(self, period, True, length)
    c.get_keys().update(bam_standard_value_heads=0, bam_local_vo_independent_keys=True)
    return c

  def test_formal_tree_and_budget(self):
    c = self.config(RUN)
    c.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    args, *_ = train_compile.get_shaped_inputs(mesh, c)
    flat = flatten_dict(nn.unbox(args[0].params))
    actual = sum(math.prod(v.shape) for v in flat.values())
    print('INDEPENDENT_VO_PARAMETERS', actual, flush=True)
    self.assertEqual(actual, 432112192)
    self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (2510, 1128, 3904))
    self.assertLess(abs(3600*c.mlp_dim/(3600*c.bam_memory_mlp_dim+458000)-2), .001)
    self.assertNotIn('bam_mlp_dim_by_layer', c.get_keys())
    self.assertIsNone(c.mlp_dim_by_block)
    self.assertFalse(c.bam_pair_scan)
    self.assertEqual(c.bam_final_local_layer_count, 1)
    self.assertEqual(c.bam_attention_write_layers, list(range(17)))
    self.assertTrue(c.bam_skip_final_mlp_write)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
    body = {p[p.index('self_attention')+1:]:v for p,v in flat.items()
            if 'layers' in p and 'self_attention' in p}
    self.assertEqual(body['W_lv_c8_up', 'kernel'].shape, (256, 17, 16, 32))
    self.assertEqual(body['general_v_pre_bias',].shape, (16, 17, 32))
    self.assertEqual(body['full_m_read_down', 'kernel'].shape, (1200, 17, 256))
    self.assertFalse(any('W_lv_c8_down' in p for p in flat))
    for p in flat:
      if 'final_local_layer' in p:
        self.assertFalse(set(p) & {'memory_mlp', 'P_loc_down', 'P_loc_up', 'W_gw', 'mlp_address_up'})

  def test_cloned_initial_key_and_independent_learning_paths(self):
    c = self.tiny()
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    mod = fusion.SubDecoderLayer(c, mesh, quant=None, layer_inx=4)
    x = jax.random.normal(jax.random.key(1), (1,4,150))
    m = jax.random.normal(jax.random.key(2), (1,4,75,32))
    pos = jnp.arange(4)[None]; seg = jnp.ones((1,4),jnp.int32)
    args = (x,seg,pos,seg,None,True,MODEL_MODE_TRAIN,None)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      p = nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,is_global=True)['params'])
      a = p['self_attention']
      v_heads = a['W_lv_c8_up']['kernel'].shape[1]
      np.testing.assert_array_equal(a['W_lv_c8_up']['kernel'], a['W_R_up']['kernel'][:,-v_heads:,0,:])
      seen = []
      def capture(next_fun,args,kwargs,ctx):
        out = next_fun(*args,**kwargs)
        if ctx.method_name == '_independent_local_vo': seen.append(out)
        return out
      with nn.intercept_methods(capture):
        mod.apply({'params':p},*args,M_in=m,is_global=True)
        changed = copy.deepcopy(p)
        changed['self_attention']['general_v_pre_bias'] += .2
        mod.apply({'params':changed},*args,M_in=m,is_global=True)
      self.assertGreater(float(jnp.max(jnp.abs(seen[0][0]-seen[1][0]))),1e-6)
      np.testing.assert_array_equal(seen[0][1], seen[1][1])
      if c.bam_standard_value_heads:
        np.testing.assert_array_equal(seen[0][0][...,:c.bam_standard_value_heads,:],0)
        if not c.bam_standard_value_keep_local_o:
          np.testing.assert_array_equal(seen[0][1][...,:c.bam_standard_value_heads,:],0)
      grad = jax.grad(lambda q:jnp.mean(mod.apply({'params':q},*args,M_in=m,is_global=True)[0]**2))(p)
      for name in ('W_R_up','W_lv_c8_up'):
        self.assertGreater(float(jnp.sum(grad['self_attention'][name]['kernel']**2)),0)

  check_pallas = StandardValueTest.check_pallas
  def test_model_pallas_all_gradients(self):
    self.check_pallas(True)

  def test_independent_kernel_matches_direct_math(self):
    b,t,n,v,k,rs = 1,128,2,8,16,6
    keys = iter(jax.random.split(jax.random.key(42),30))
    random = lambda shape:jax.random.normal(next(keys),shape,jnp.float32)
    m = random((b,t,v,k)); sw = random((4*n,v))*.2
    xs = (random((b,t,n,v)),random((b,t,n)),random((b,t,n,v)),random((b,t,n)),
          random((b,t,2*n,v)),random((b,t,n)),random((b,t,n)),
          random((b,t,n,rs)),random((b,t,n,rs)))
    sg = tuple(jax.nn.sigmoid(random((b,t,n))) for _ in range(4))
    sh = jax.nn.sigmoid(random((b,t,v))).at[:,0].set(0)
    eps,scale = 1e-4,.2
    def kernel(m,sw,xs,sg,sh):
      return bam_pallas_v8.read_tm(m,sw,*xs,qk_cols=k-rs,read_epsilon=eps,key_scale=scale,
          forward_tile=64,reverse_tile=64,interpret=True,q_scale=.125,
          static_gates=sg,shift=sh,pad_heads=True)
    def direct(m,sw,xs,sg,sh):
      rq,lq,rk,lk,rr,lo,lv,qs,ks = xs
      rv,ro = jnp.split(rr,2,axis=2)
      mr = m + sh[...,None]*(jnp.concatenate((m[:,:1],m[:,:-1]),axis=1)-m)
      out = []
      for i,(r,l) in enumerate(zip((rq,rk,rv,ro),(lq,lk,lv,lo))):
        norm = r*jax.lax.rsqrt(jnp.mean(r*r,axis=-1,keepdims=True)+eps)
        weight = norm*scale*jax.nn.sigmoid(l)[...,None] + sw[i*n:(i+1)*n]*sg[i][...,None]
        out.append(jnp.einsum('btnv,btvk->btnk',weight,mr))
      out[0] = jnp.concatenate((out[0][...,:k-rs],qs),axis=-1)*.125
      out[1] = jnp.concatenate((out[1][...,:k-rs],ks),axis=-1)
      return tuple(out)+(m,)
    cts = tuple(random(y.shape) for y in direct(m,sw,xs,sg,sh))
    loss = lambda fn:lambda *args:sum(jnp.sum(y*ct) for y,ct in zip(fn(*args),cts))
    for a,bv in zip(kernel(m,sw,xs,sg,sh),direct(m,sw,xs,sg,sh)):
      np.testing.assert_allclose(a,bv,rtol=1e-4,atol=1e-5)
    ga = jax.grad(loss(kernel),argnums=(0,1,2,3,4))(m,sw,xs,sg,sh)
    gb = jax.grad(loss(direct),argnums=(0,1,2,3,4))(m,sw,xs,sg,sh)
    for a,bv in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
      self.assertLess(float(jnp.max(jnp.abs(a-bv))/(jnp.max(jnp.abs(bv))+1e-12)),1e-4)


class HybridIndependentVOTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=2, keep=None, length=4):
    keep = getattr(self,'keep',True) if keep is None else keep
    c = HybridValueTest.tiny(self,period,keep,length)
    c.get_keys()['bam_local_vo_independent_keys'] = True
    return c

  def test_formal_hybrid_trees_and_budgets(self):
    for keep,h,hm,last,total in [(True,2451,1098,3831,432115216),(False,2459,1102,3840,432114640)]:
      suffix=('Keep' if keep else 'Drop')+'LocalO'
      parent='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4'+suffix+'TruePile'
      run=parent.replace('TruePile','SeparateVOKeysTruePile')
      c=self.config(run);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      actual=sum(math.prod(v.shape) for v in flat.values());print('HYBRID_INDEPENDENT_VO_PARAMETERS',keep,actual,flush=True)
      self.assertEqual(actual,total)
      self.assertEqual(c.compare_runs,[parent,RUN])
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(h,hm,last))
      self.assertLess(abs(3600*h/(3600*hm+458000)-2),.001)
      self.assertNotIn('bam_mlp_dim_by_layer',c.get_keys());self.assertIsNone(c.mlp_dim_by_block)
      self.assertIsNone(c.bam_final_local_mlp_dim);self.assertFalse(c.bam_pair_scan)
      self.assertEqual(c.bam_final_local_layer_count,1);self.assertEqual(c.bam_attention_write_layers,list(range(17)))
      self.assertTrue(c.bam_skip_final_mlp_write and c.bam_pallas_core and c.bam_pallas_fused_inputs)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics and c.sa_use_fused_bwd_kernel)
      self.assertFalse(any(v for k,v in c.get_keys().items() if k.startswith(('bam_record','record_bam'))))
      body={p[p.index('self_attention')+1:]:v for p,v in flat.items() if 'layers' in p and 'self_attention' in p}
      self.assertEqual(body['W_lv_c8_up','kernel'].shape,(256,17,12,32))
      self.assertEqual(body['general_v_pre_bias',].shape,(12,17,32))
      self.assertEqual(body['W_R_up','kernel'].shape,(256,17,16 if keep else 12,1,32))
      self.assertEqual(body['value','kernel'].shape,(1200,17,4,75))
      self.assertFalse(any('W_lv_c8_down' in p for p in flat))
      for p in flat:
        if 'final_local_layer' in p:
          self.assertFalse(set(p)&{'memory_mlp','P_loc_down','P_loc_up','W_gw','mlp_address_up'})

  def test_hybrid_cloned_keys_masks_and_independence(self):
    for self.keep in (True,False):
      IndependentVOTest.test_cloned_initial_key_and_independent_learning_paths(self)

  check_pallas=StandardValueTest.check_pallas
  def test_hybrid_keep_pallas_all_gradients(self):self.check_pallas(True)
  def test_hybrid_drop_pallas_all_gradients(self):self.check_pallas(False)


if __name__ == '__main__': unittest.main()

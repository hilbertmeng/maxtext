"""Integrated V/O gamma: independent kernel truth, actual scan initial values, model gradients."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest
from bam_pallas_mshift_test import inputs, mixed, B, T, V, K, N, QKC, RS, EPS, SCALE
from layers import bam_pallas_v8 as v8
from layers.models import Transformer

P = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75'
H = P + 'CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile'
D2 = P + 'DynamicVONormScaleDepth2RawMLPEmbeddingFixedTruePile'
D1 = P + 'DynamicVONormScaleDepth1RawMLPEmbeddingFixedTruePile'


class DynamicVONormScaleTest(MLPWriteTest, unittest.TestCase):
  tiny = CommonAllReadGammaTest.tiny

  def test_formal_configs_budget_and_scan_initial_values(self):
    for run, slope in ((D2, 2.), (D1, 1.)):
      c = self.config(run)
      self.assertEqual(c.compare_runs, [H])
      self.assertFalse(c.bam_common_vo_read_scale)
      self.assertEqual(c.bam_local_vo_dynamic_norm_scale_depth_slope, slope)
      self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (2510, 1128, 3904))
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432130624)
      self.assertFalse(any(p[-1] == 'common_vo_read_scale' for p in flat))
      for arm in ('v', 'o'):
        self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items()
            if p[-1] == 'local_' + arm + '_dynamic_norm_scale'), 9216)
      c = self.tiny(run)
      c.get_keys().update(num_decoder_layers=18, bam_layer_modes=['local_qk+local_v+local_o']*18,
          bam_attention_write_layers=list(range(17)), bam_mlp_write_layers=list(range(17)))
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quant=None)
      tok = jnp.array([[1,2,3,4]], jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = nn.unbox(model.init(jax.random.key(1), tok, jnp.arange(4)[None], tok, jnp.ones_like(tok)))
      for path, value in flatten_dict(p).items():
        if path[-1] not in ('local_v_dynamic_norm_scale', 'local_o_dynamic_norm_scale'):
          continue
        expected = 1/np.sqrt(1+slope*np.arange(17)) if 'layers' in path else [1/np.sqrt(1+slope*17)]
        actual = np.asarray(jnp.moveaxis(value, c.param_scan_axis, 0) if 'layers' in path else value[None])
        np.testing.assert_allclose(actual, np.broadcast_to(np.array(expected)[:,None,None], actual.shape), rtol=1e-6)
      self.assertTrue(all(v == 0. for path, v in flatten_dict(train.get_wd_tree(c, p)).items()
          if path[-1] in ('local_v_dynamic_norm_scale', 'local_o_dynamic_norm_scale')))
      print('VO_GAMMA_BUDGET_SCAN_OK', run, flush=True)

  def test_kernel_independent_reference_all_gradients(self):
    m, sw, xs, sgs, shift = inputs(82)
    xs = xs[:4] + (jax.random.normal(jax.random.key(83), (B,T,2*N,V)),) + xs[5:]
    gv = .2 + .4*jax.random.normal(jax.random.key(84), (N,V))
    go = .3 + .5*jax.random.normal(jax.random.key(85), (N,V))
    mask = jax.random.bernoulli(jax.random.key(86), .8, (B,T,N,V)).astype(jnp.float32)/.8
    cts = tuple(jax.random.normal(jax.random.key(90+i), (B,T,N,K)) for i in range(4)) + (
        jax.random.normal(jax.random.key(94), m.shape),)

    def reference(m, sw, xs, sgs, shift, gv, go):
      rq,lq,rk,lk,rr,lo,lv,qs,ks = xs
      norm = lambda x: x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+EPS)
      rv,ro = jnp.split(rr, 2, axis=2)
      rows = (rq,rk,rv,ro);logits = (lq,lk,lv,lo);scales = (1.,1.,gv,go)
      keys = [(SCALE*jax.nn.sigmoid(z)[...,None]*norm(r)*g
          + sw[i*N:(i+1)*N]*sgs[i][...,None]) for i,(r,z,g) in enumerate(zip(rows,logits,scales))]
      keys[3] *= mask
      y = [jnp.einsum('btvk,btnv->btnk', mixed(m,shift), a) for a in keys]
      return jnp.concatenate((y[0][...,:QKC],qs),-1)*.125, jnp.concatenate((y[1][...,:QKC],ks),-1), y[2], y[3], m

    for tile in (64, 256):
      def kernel(m,sw,xs,sgs,shift,gv,go):
        return v8.read_tm(m,sw,*xs,qk_cols=QKC,read_epsilon=EPS,key_scale=SCALE,
            forward_tile=tile,reverse_tile=tile,interpret=True,q_scale=.125,
            static_gates=sgs,shift=shift,o_weight_multiplier=mask,
            v_dynamic_scale=gv,o_dynamic_scale=go,pad_heads=True)
      def loss(fn, *args):
        outs = fn(*args)
        return sum(jnp.sum(o*c) for o,c in zip(outs,cts)), outs
      args=(m,sw,xs,sgs,shift,gv,go)
      (_,a),ga=jax.value_and_grad(lambda *a:loss(kernel,*a),argnums=tuple(range(7)),has_aux=True)(*args)
      (_,b),gb=jax.value_and_grad(lambda *a:loss(reference,*a),argnums=tuple(range(7)),has_aux=True)(*args)
      errors=[]
      for x,y in zip(jax.tree.leaves((a,ga)),jax.tree.leaves((b,gb))):
        errors.append(float(jnp.max(jnp.abs(x-y))/(jnp.max(jnp.abs(y))+1e-12)))
      self.assertLess(max(errors),1e-5)
      self.assertGreater(float(jnp.max(jnp.abs(ga[-2]))),0.)
      self.assertGreater(float(jnp.max(jnp.abs(ga[-1]))),0.)
      print('VO_GAMMA_KERNEL_ALL_GRADS',tile,max(errors),flush=True)

  def test_depth2_native_pallas_model_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,D2)

  def test_depth1_native_pallas_model_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,D1)

  def test_bf16_native_pallas_model_all_gradients(self):
    CommonAllReadGammaTest.check_model(self,D2,dtype='bfloat16')


if __name__ == '__main__':
  unittest.main()

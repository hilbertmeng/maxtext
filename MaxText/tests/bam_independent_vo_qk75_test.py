"""QK93/V75 on the completed private2:1, independent-VO SOTA."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train_compile
from bam_independent_vo_test import IndependentVOTest
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers import bam_pallas_v8

PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile'
RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysQK75TruePile'
PYTHIA_PARENT = 'BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeys'
PYTHIA_RUN = PYTHIA_PARENT + 'QK80'


class IndependentVOQK75Test(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=3, keep=True, length=4):
    c = IndependentVOTest.tiny(self, period, keep, length)
    c.get_keys().update(bam_local_qk_col_output_dim=75, bam_partial_rope_nope_dim=75)
    return c

  def test_equal_full_parameter_trees_and_runtime(self):
    trees = []
    for exp in (PARENT, RUN):
      c = self.config(exp)
      c.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432112192)
      trees.append({p: (v.shape, str(v.dtype)) for p, v in flat.items()})
      self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (2510, 1128, 3904))
      self.assertEqual((c.head_dim, c.bam_k, c.bam_v, c.bam_standard_qk_dim), (75, 75, 32, 18))
      self.assertFalse(c.qk_norm)
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertEqual(c.bam_attention_write_layers, list(range(17)))
      self.assertTrue(c.bam_skip_final_mlp_write)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.bam_pallas_body, 'v8t')
      self.assertEqual(c.tensorboard_gcs_rotate_period, 500)
    self.assertEqual(trees[0], trees[1])
    print('QK93_V75_EQUAL_PARAMETER_TREES_OK 432112192', flush=True)

  def test_model_native_pallas_all_gradients(self):
    StandardValueTest.check_pallas(self, True)

  def test_kernel_qk93_v75_all_gradients_and_tile_boundaries(self):
    b, t, n, v, k, rs = 1, 128, 2, 32, 75, 18
    keys = iter(jax.random.split(jax.random.key(52), 30))
    random = lambda shape: jax.random.normal(next(keys), shape, jnp.float32)
    m = random((b,t,v,k)); sw = random((4*n,v))*.2
    xs = (random((b,t,n,v)),random((b,t,n)),random((b,t,n,v)),random((b,t,n)),
          random((b,t,2*n,v)),random((b,t,n)),random((b,t,n)),
          random((b,t,n,rs)),random((b,t,n,rs)))
    sg = tuple(jax.nn.sigmoid(random((b,t,n))) for _ in range(4))
    sh = jax.nn.sigmoid(random((b,t,v))).at[:,0].set(0).at[:,64].set(0)
    scale = 1 / math.sqrt(75)
    def kernel(m,sw,xs,sg,sh):
      return bam_pallas_v8.read_tm(m,sw,*xs,qk_cols=75,read_epsilon=1e-4,key_scale=.2,
          forward_tile=64,reverse_tile=64,interpret=True,q_scale=scale,
          static_gates=sg,shift=sh,pad_heads=True)
    def direct(m,sw,xs,sg,sh):
      rq,lq,rk,lk,rr,lo,lv,qs,ks = xs
      rv,ro = jnp.split(rr,2,axis=2)
      mr = m + sh[...,None]*(jnp.concatenate((m[:,:1],m[:,:-1]),axis=1)-m)
      out = []
      for i,(r,l) in enumerate(zip((rq,rk,rv,ro),(lq,lk,lv,lo))):
        weight = r*jax.lax.rsqrt(jnp.mean(r*r,axis=-1,keepdims=True)+1e-4)*.2*jax.nn.sigmoid(l)[...,None]
        weight += sw[i*n:(i+1)*n]*sg[i][...,None]
        out.append(jnp.einsum('btnv,btvk->btnk',weight,mr))
      out[0] = jnp.concatenate((out[0],qs),axis=-1)*scale
      out[1] = jnp.concatenate((out[1],ks),axis=-1)
      return tuple(out)+(m,)
    outs = kernel(m,sw,xs,sg,sh)
    self.assertEqual([o.shape[-1] for o in outs[:4]], [93,93,75,75])
    for a,bv in zip(outs,direct(m,sw,xs,sg,sh)):
      np.testing.assert_allclose(a,bv,rtol=1e-4,atol=1e-5)
    cts = tuple(random(y.shape) for y in outs)
    loss = lambda fn: lambda *args: sum(jnp.sum(y*ct) for y,ct in zip(fn(*args),cts))
    ga = jax.grad(loss(kernel),argnums=(0,1,2,3,4))(m,sw,xs,sg,sh)
    gb = jax.grad(loss(direct),argnums=(0,1,2,3,4))(m,sw,xs,sg,sh)
    for a,bv in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
      self.assertLess(float(jnp.max(jnp.abs(a-bv))/(jnp.max(jnp.abs(bv))+1e-12)),1e-4)
    print('QK93_V75_PALLAS_FORWARD_ALL_GRADIENTS_OK', flush=True)


class IndependentVOFullNoPEPythiaTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=3, keep=True, length=4):
    c = self.config(PYTHIA_RUN, target_train_batch_size=None,
        base_emb_dim=160, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=3, base_mlp_dim=32, vocab_size=32,
        dtype='float32', weight_dtype='float32', attention='dot_product',
        bam_splash_attention=False)
    c.get_keys().update(bam_pallas_core=False, max_target_length=length,
        max_prefill_predict_length=length, query_chunk_size=length,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1], bam_mlp_write_layers=[0,1],
        bam_memory_mlp_dim=16, bam_last_mlp_dim=32, emb_bam_num_head=2,
        bam_local_full_m_read_bottleneck_dim=8, bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8)
    return c

  def test_formal_tree_only_adds_100d_norm_scales(self):
    trees = []
    for exp, total, width in ((PYTHIA_PARENT, 2775189600, 80),
                              (PYTHIA_RUN, 2775190880, 100)):
      c = self.config(exp, target_train_batch_size=None, attention='dot_product',
                      bam_splash_attention=False)
      c.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
      trees.append({p: (v.shape, str(v.dtype)) for p, v in flat.items()})
      norms = [v for p,v in flat.items() if 'qk_norm' in p]
      self.assertEqual(sum(math.prod(v.shape) for v in norms), 32*2*width)
      self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat)
      self.assertEqual((c.head_dim,c.bam_k,c.bam_v,c.bam_standard_qk_dim),(80,80,64,20))
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(7808,2944,9984))
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertIsNone(c.mlp_dim_by_block)
      self.assertEqual(c.bam_attention_write_layers,list(range(31)))
      self.assertTrue(c.bam_skip_final_mlp_write)
    self.assertEqual(trees[0].keys(), trees[1].keys())
    for path in trees[0]:
      if trees[0][path] != trees[1][path]:
        self.assertIn('qk_norm', path)
    print('PYTHIA_QK100_V80_BUDGET_OK parent2775189600 new2775190880 delta1280', flush=True)

  def test_model_native_pallas_full_qknorm_all_gradients(self):
    StandardValueTest.check_pallas(self, True)


if __name__ == '__main__':
  unittest.main()

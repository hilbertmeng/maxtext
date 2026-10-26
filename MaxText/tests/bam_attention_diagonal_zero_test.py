"""Post-softmax diagonal zero, including the retained denominator's gradient."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as masks

import max_utils, pyconfig, train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions
from layers.bam_attention_diagonal_zero import make_zero_diagonal_splash, make_negative_diagonal_splash
from layers.models import Transformer

PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile'
RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysAlphaDiagZeroTruePile'
PRE_RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeTruePile'
RATIO1_RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio1to1SeparateVOKeysLogitDiagNegativeQK75TruePile'
SHARED_PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile'
SHARED_PRE_RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1LogitDiagNegativeTruePile'
COMBO_RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile'
V48_RUN = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75V48TruePile'


class AttentionDiagonalZeroTest(MLPWriteTest, unittest.TestCase):
  def test_full_parameter_tree_and_runtime(self):
    trees = []
    for name in (PARENT, RUN, PRE_RUN, COMBO_RUN):
      c = self.config(name)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_ = train_compile.get_shaped_inputs(mesh,c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112192)
      trees.append({p:(v.shape,str(v.dtype)) for p,v in flat.items()})
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertEqual(c.bam_attention_zero_diagonal,name==RUN)
      self.assertEqual(c.bam_attention_negative_diagonal,name in (PRE_RUN,COMBO_RUN))
      self.assertEqual((c.bam_local_qk_col_output_dim,c.bam_standard_qk_dim),(75 if name==COMBO_RUN else 57,18))
      self.assertTrue(c.bam_local_vo_independent_keys and c.bam_pallas_core)
      self.assertTrue(c.sa_use_fused_bwd_kernel and c.record_training_health_metrics)
      self.assertFalse(c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertEqual(trees[0],trees[1])
    self.assertEqual(trees[0],trees[2])
    self.assertEqual(trees[0],trees[3])
    print('DIAGONAL_ZERO_PARAMETER_TREE_IDENTICAL 432112192',flush=True)

  def test_shared_vo_pre_parameter_tree_and_runtime(self):
    trees=[]
    for run in (SHARED_PARENT,SHARED_PRE_RUN):
      c=self.config(run)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432113280)
      trees.append({p:(v.shape,str(v.dtype)) for p,v in flat.items()})
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2535,1140,3806))
      self.assertEqual(c.bam_attention_negative_diagonal,run==SHARED_PRE_RUN)
      self.assertFalse(c.bam_local_vo_independent_keys)
      self.assertTrue(c.bam_pair_scan and c.scan_layers)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertEqual(trees[0],trees[1])
    print('SHARED_VO_PRE_TREE_IDENTICAL 432113280',flush=True)

  def test_shared_vo_pre_full_model_pallas_splash_gradients(self):
    self._model_case(SHARED_PRE_RUN)

  def test_ratio1_full_nope_parameter_budget(self):
    trees=[]
    for run in (COMBO_RUN,RATIO1_RUN):
      c=self.config(run)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params));trees.append(flat)
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112192)
      dims=(1883,1755,3904) if run==RATIO1_RUN else (2510,1128,3904)
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),dims)
      self.assertEqual((c.bam_local_qk_col_output_dim,c.bam_standard_qk_dim),(75,18))
      self.assertTrue(c.bam_attention_negative_diagonal and c.bam_local_vo_independent_keys)
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertEqual(c.bam_attention_write_layers,list(range(17)))
      self.assertTrue(c.bam_skip_final_mlp_write)
    self.assertEqual(trees[0].keys(),trees[1].keys())
    for path in trees[0]:
      if trees[0][path].shape!=trees[1][path].shape:self.assertIn('mlp','/'.join(path).lower())
    self.assertAlmostEqual((1883*3600)/(1755*3600+458000),1.0004132231404959)
    print('RATIO1_PRE_FULL_NOPE_BUDGET_OK 432112192 ratio1.000413',flush=True)

  def test_ratio1_full_nope_pallas_splash_all_gradients(self):
    self._model_case(RATIO1_RUN)

  def test_v48_full_nope_parameter_budget_and_runtime(self):
    c=self.config(V48_RUN)
    c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432111056)
    self.assertEqual((c.bam_k,c.bam_v,c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),
                     (75,48,2446,1077,3802))
    for key in ('bam_local_full_m_read_bottleneck_dim','bam_write_v_bottleneck_dim',
                'emb_bam_v_bottleneck_dim','bam_mlp_write_address_rank'):
      self.assertEqual(getattr(c,key),256)
    self.assertEqual((c.bam_local_qk_col_output_dim,c.bam_standard_qk_dim),(75,18))
    self.assertTrue(c.bam_attention_negative_diagonal and c.bam_local_vo_independent_keys)
    self.assertTrue(c.scan_layers and not c.bam_pair_scan)
    self.assertEqual(c.bam_attention_write_layers,list(range(17)))
    self.assertTrue(c.bam_skip_final_mlp_write and c.bam_pallas_core and c.bam_pallas_fused_inputs)
    self.assertTrue(c.record_training_health_metrics and c.sa_use_fused_bwd_kernel)
    self.assertFalse(c.record_internal_nn_metrics or c.qk_norm)
    self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertAlmostEqual(2446*3600/(1077*3600+523792),2.0008216329409367)
    print('V48_PRE_FULL_NOPE_BUDGET_OK 432111056 parent_delta-1136 ratio2.00082163',flush=True)

  def test_v48_full_nope_pallas_splash_all_gradients(self):
    self._model_case(V48_RUN)

  def test_native_removes_diagonal_without_renormalization(self):
    q = jnp.zeros((1,4,1,2));v = jnp.arange(1,5,dtype=jnp.float32)[None,:,None,None]
    valid = jnp.tril(jnp.ones((4,4),jnp.bool_))[None]
    out,alpha = attentions._attention_op(q,q,v,valid,zero_diagonal_mask=jnp.eye(4,dtype=jnp.bool_))
    np.testing.assert_allclose(alpha.sum(-1),jnp.array([[[0.,.5,2/3,3/4]]]),atol=1e-7)
    np.testing.assert_allclose(out.flatten(),jnp.array([0.,.5,1.,1.5]),atol=1e-7)
    # Native query chunks use GLOBAL token indices to identify the diagonal.
    partial,_ = attentions._attention_op(q[:,2:],q,v,valid[:,2:],
        zero_diagonal_mask=jnp.arange(2,4)[:,None]==jnp.arange(4)[None])
    np.testing.assert_array_equal(partial,out[:,2:])
    before,pre_alpha = attentions._attention_op(q,q,v,valid,
        negative_diagonal_mask=jnp.eye(4,dtype=jnp.bool_))
    np.testing.assert_allclose(pre_alpha.sum(-1),jnp.array([[[0.,1.,1.,1.]]]),atol=1e-7)
    np.testing.assert_allclose(before.flatten(),jnp.array([0.,1.,1.5,2.]),atol=1e-7)

  def test_splash_outputs_and_all_gradients_match_dense_reference(self):
    h,t,d = 2,128,75
    keys = iter(jax.random.split(jax.random.key(7),20))
    random = lambda shape:jax.random.normal(next(keys),shape,jnp.float32)
    q,k,v = random((h,t,d))*.3,random((h,t,d))*.3,random((h,t,d))
    seg = jnp.where(jnp.arange(t)<64,1,2)
    ids = splash.SegmentIds(seg,seg)
    for seq_minor,cap in ((False,None),(True,2.)):
      layout = splash.QKVLayout.SEQ_MINOR if seq_minor else splash.QKVLayout.HEAD_DIM_MINOR
      blocks = splash.BlockSizes(block_q=128,block_kv=128,block_kv_compute=128,
          block_q_dkv=128,block_kv_dkv=128,block_kv_dkv_compute=128,
          use_fused_bwd_kernel=True,q_layout=layout,k_layout=layout,v_layout=layout)
      mask = masks.CausalMask((t,t)) & masks.LocalMask((t,t),window_size=(31,0),offset=0)
      kernel = splash.make_splash_mha(mask=masks.MultiHeadMask((mask,)*h),
          head_shards=1,q_seq_shards=1,block_sizes=blocks,interpret=True,attn_logits_soft_cap=cap)
      op = make_zero_diagonal_splash(kernel)
      def reference(q,k,v):
        logits = jnp.einsum('htd,hsd->hts',q,k)
        if cap:logits = jnp.tanh(logits/cap)*cap
        i,j = jnp.arange(t)[:,None],jnp.arange(t)[None]
        valid = (j<=i)&(j>i-32)&(seg[:,None]==seg[None])
        a = jax.nn.softmax(jnp.where(valid,logits,splash.DEFAULT_MASK_VALUE),-1)
        a = jnp.where(i==j,0.,a)
        return jnp.einsum('hts,hsd->htd',a,v)
      a,b = op(q,k,v,ids),reference(q,k,v)
      np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-5)
      np.testing.assert_array_equal(a[:,[0,64]],0)
      bf16_out = op(*(x.astype(jnp.bfloat16) for x in (q,k,v)),ids)
      np.testing.assert_array_equal(bf16_out[:,[0,64]],0)
      target = random(a.shape)
      ga = jax.grad(lambda q,k,v:jnp.sum(op(q,k,v,ids)*target),argnums=(0,1,2))(q,k,v)
      gb = jax.grad(lambda q,k,v:jnp.sum(reference(q,k,v)*target),argnums=(0,1,2))(q,k,v)
      for x,y in zip(ga,gb):np.testing.assert_allclose(x,y,rtol=8e-4,atol=5e-5)
      # The last self key still affects the denominator: it must receive a gradient.
      last_grad = jax.grad(lambda kk:op(q,kk,v,ids)[:,-1].sum())(k)
      self.assertGreater(float(jnp.linalg.norm(last_grad[:,-1])),1e-5)
    print('DIAGONAL_ZERO_SPLASH_FORWARD_AND_ALL_GRADIENTS_OK',flush=True)

  def test_negative_logits_splash_and_all_gradients(self):
    h,t,d = 2,128,75
    keys = jax.random.split(jax.random.key(71),4)
    q,k,v,target = [jax.random.normal(key,(h,t,d),jnp.float32) for key in keys]
    q,k = q*.3,k*.3
    seg = jnp.where(jnp.arange(t)<64,1,2);ids = splash.SegmentIds(seg,seg)
    blocks = splash.BlockSizes(block_q=128,block_kv=128,block_kv_compute=128,
        block_q_dkv=128,block_kv_dkv=128,block_kv_dkv_compute=128,use_fused_bwd_kernel=True)
    mask = masks.CausalMask((t,t),offset=-1) & masks.LocalMask((t,t),window_size=(31,0),offset=0)
    op = make_negative_diagonal_splash(splash.make_splash_mha(
        mask=masks.MultiHeadMask((mask,)*h),head_shards=1,q_seq_shards=1,
        block_sizes=blocks,interpret=True))
    def reference(q,k,v):
      i,j = jnp.arange(t)[:,None],jnp.arange(t)[None]
      valid = (j<=i)&(j>i-32)&(seg[:,None]==seg[None])
      out,_ = attentions._attention_op(q.transpose(1,0,2)[None],k.transpose(1,0,2)[None],
          v.transpose(1,0,2)[None],valid[None],negative_diagonal_mask=i==j)
      return out[0].transpose(1,0,2)
    np.testing.assert_allclose(op(q,k,v,ids),reference(q,k,v),rtol=3e-4,atol=3e-5)
    np.testing.assert_allclose(op(q,k,v,ids)[:,[0,64]],0,atol=1e-7)
    ga = jax.grad(lambda q,k,v:jnp.sum(op(q,k,v,ids)*target),argnums=(0,1,2))(q,k,v)
    gb = jax.grad(lambda q,k,v:jnp.sum(reference(q,k,v)*target),argnums=(0,1,2))(q,k,v)
    for a,b in zip(ga,gb):np.testing.assert_allclose(a,b,rtol=8e-4,atol=5e-5)
    last = jax.grad(lambda kk:op(q,kk,v,ids)[:,-1].sum())(k)
    np.testing.assert_allclose(last[:,-1],0,atol=3e-6)
    print('NEGATIVE_DIAGONAL_LOGITS_SPLASH_ALL_GRADIENTS_OK',flush=True)

  def test_full_model_pallas_splash_logits_and_all_gradients(self):
    self._model_case(RUN)

  def test_full_model_negative_logits_pallas_splash_all_gradients(self):
    self._model_case(PRE_RUN)

  def test_full_nope_negative_logits_pallas_all_gradients(self):
    self._model_case(COMBO_RUN)

  def _model_case(self,run):
    shared_control=run==SHARED_PRE_RUN
    layers=4 if shared_control else 3
    c = self.config(run,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=layers,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_splash_attention=False,bam_memory_mlp_dim=32 if run==RATIO1_RUN else 16,
        bam_last_mlp_dim=32,bam_layer_modes=['local_qk+local_v+local_o']*layers,
        bam_attention_write_layers=list(range(layers)) if shared_control else [0,1],
        bam_mlp_write_layers=list(range(layers-1)),emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,max_target_length=128,max_prefill_predict_length=128,
        query_chunk_size=64)
    if shared_control:c.get_keys().update(bam_final_local_mlp_dim=32,mlp_dim_by_block=[32,32])
    pc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    native,fused = Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
    tok = jax.random.randint(jax.random.key(20),(1,128),0,32)
    seg = jnp.where(jnp.arange(128)<64,1,2)[None];pos = (jnp.arange(128)%64)[None]
    original = splash.make_splash_mha
    def interpret(*args,**kwargs):return original(*args,**dict(kwargs,interpret=True))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
      p,pp = (nn.unbox(m.init(jax.random.key(21),tok,pos,seg,tok)) for m in (native,fused))
      self.assertEqual(jax.tree.structure(p),jax.tree.structure(pp))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
      leaves,tree = jax.tree.flatten(p);keys = jax.random.split(jax.random.key(22),len(leaves))
      p = jax.tree.unflatten(tree,[a+.01*jax.random.normal(k,a.shape,a.dtype) for a,k in zip(leaves,keys)])
      call = lambda m,p:m.apply(p,tok,pos,seg,tok,enable_dropout=False)[0]
      target = jax.random.normal(jax.random.key(23),call(native,p).shape)
      def loss(m,p):
        y=call(m,p);return jnp.mean(y*target),y
      (_,a),ga = jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
      (_,b),gb = jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(p)
    np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    errors = [float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12))
              for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb))]
    self.assertLess(max(errors),1e-3)
    print('DIAGONAL_FULL_MODEL_ALL_GRADIENTS_OK',run,max(errors),flush=True)


if __name__=='__main__':unittest.main()

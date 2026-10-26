"""Strict-history BAM attention: native/Splash/Pallas outputs, gradients and budgets."""
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
from layers.bam_attention_diagonal_zero import make_negative_diagonal_splash
from layers.models import Transformer

PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile'
PRE_RUN = PARENT.removesuffix('TruePile') + 'LogitDiagNegativeTruePile'
COMBO_RUN = PARENT.removesuffix('TruePile') + 'LogitDiagNegativeQK75TruePile'
PYTHIA_PARENT = 'BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeys'
PYTHIA_PRE = PYTHIA_PARENT + 'LogitDiagNegative'
PYTHIA_COMBO = PYTHIA_PRE + 'QK80'


class NegativeDiagonalTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budgets_and_parameter_trees(self):
    pairs = ((PARENT,PRE_RUN,432112192,75),(PARENT,COMBO_RUN,432112192,75),
             (PYTHIA_PARENT,PYTHIA_PRE,2775189600,80),
             (PYTHIA_PARENT+'QK80',PYTHIA_COMBO,2775190880,100))
    for parent,run,total,width in pairs:
      trees=[]
      for name in (parent,run):
        c=self.config(name,**({'target_train_batch_size':None} if name.startswith('BamPythia') else {}))
        c.get_keys().update(max_target_length=256,max_prefill_predict_length=256,
                            attention='dot_product',bam_splash_attention=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        args,*_=train_compile.get_shaped_inputs(mesh,c)
        flat=flatten_dict(nn.unbox(args[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),total)
        trees.append({p:(v.shape,str(v.dtype)) for p,v in flat.items()})
        self.assertEqual(c.bam_attention_negative_diagonal,name==run or parent==PYTHIA_COMBO)
        self.assertTrue(c.scan_layers and not c.bam_pair_scan)
        self.assertTrue(c.bam_skip_final_mlp_write)
        if name.startswith('BamPythia'):
          self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(7808,2944,9984))
          self.assertEqual(c.bam_attention_write_layers,list(range(31)))
          self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat)
          norms=[v for p,v in flat.items() if 'qk_norm' in p]
          self.assertEqual(sum(math.prod(v.shape) for v in norms),32*2*width)
          if name==run:self.assertEqual(c.bam_pallas_vmem_mib,60)
        else:
          self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
          self.assertEqual(c.bam_attention_write_layers,list(range(17)))
        self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
        self.assertTrue(c.sa_use_fused_bwd_kernel)
      self.assertEqual(trees[0],trees[1])
    print('NEGATIVE_DIAGONAL_FORMAL_TREES_IDENTICAL; PYTHIA2775189600/2775190880',flush=True)

  def test_native_history_normalization_and_global_chunk_positions(self):
    q=jnp.zeros((1,4,1,2));v=jnp.arange(1,5,dtype=jnp.float32)[None,:,None,None]
    valid=jnp.tril(jnp.ones((4,4),jnp.bool_))[None]
    out,alpha=attentions._attention_op(q,q,v,valid,negative_diagonal_mask=jnp.eye(4,dtype=jnp.bool_))
    np.testing.assert_allclose(alpha.sum(-1),jnp.array([[[0.,1.,1.,1.]]]),atol=1e-7)
    np.testing.assert_allclose(out.flatten(),jnp.array([0.,1.,1.5,2.]),atol=1e-7)
    partial,_=attentions._attention_op(q[:,2:],q,v,valid[:,2:],
        negative_diagonal_mask=jnp.arange(2,4)[:,None]==jnp.arange(4)[None])
    np.testing.assert_array_equal(partial,out[:,2:])

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

  def test_full_model_negative_logits_pallas_splash_all_gradients(self):
    self._model_case(PRE_RUN)

  def test_full_nope_negative_logits_pallas_all_gradients(self):
    self._model_case(COMBO_RUN)

  def test_pythia_negative_logits_full_head_qknorm_all_gradients(self):
    self._model_case(PYTHIA_PRE)

  def test_pythia_full_nope_negative_logits_all_gradients(self):
    self._model_case(PYTHIA_COMBO)

  def _model_case(self,run):
    pythia=run.startswith('BamPythia')
    c = self.config(run,**({'target_train_batch_size':None} if pythia else {}),base_emb_dim=160 if pythia else 150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_splash_attention=False,bam_memory_mlp_dim=16,
        bam_last_mlp_dim=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,max_target_length=128,max_prefill_predict_length=128,
        query_chunk_size=64)
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

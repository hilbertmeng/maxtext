"""Sparse XL writer contracts, native/Pallas equivalence and residual isolation."""
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
from types import SimpleNamespace
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
import max_utils, pyconfig
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer
from layers.fusion import SubDecoderLayer

NORM='BamXLPropV5EveryThirdMLPWriteNorm'
RAW='BamXLPropV5EveryThirdMLPWriteRawMemoryOnly'

class SparseWriterTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, cls):
    cfg=self.config(cls, base_emb_dim=192, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=4, base_mlp_dim=32,
        vocab_size=32, dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(bam_layer_modes=['local_qk+local_v+local_o']*4,
        bam_attention_write_layers=[0,1,2], bam_mlp_write_layers=[1],
        mlp_dim_by_block=[32,24,32], bam_final_local_mlp_dim=36,
        emb_bam_num_head=2, bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, bam_pallas_core=False,
        bam_pallas_write=False, bam_splash_attention=False,
        max_target_length=128, max_prefill_predict_length=128, query_chunk_size=64)
    return cfg

  def test_formal_contract(self):
    a,b=self.config(NORM),self.config(RAW)
    for cfg in (a,b):
      self.assertEqual(cfg.bam_mlp_write_layers,list(range(1,27,3)))
      self.assertEqual(cfg.bam_attention_write_layers,list(range(27)))
      self.assertEqual(cfg.bam_memory_mlp_dim,0)
      self.assertEqual(cfg.mlp_dim_by_block,[6071,5875,6071])
      self.assertEqual(cfg.bam_final_local_mlp_dim,6272)
      self.assertTrue(cfg.bam_pair_scan and cfg.bam_pallas_core and cfg.bam_pallas_write)
      self.assertEqual(cfg.remat_policy,'full')
      self.assertEqual(cfg.steps,50000)
      self.assertEqual(cfg.bam_attention_write_content_divisor,None)
    self.assertIsNone(a.bam_embedding_write_content_divisor)
    self.assertIsNone(a.bam_mlp_write_content_divisor)
    self.assertAlmostEqual(b.bam_embedding_write_content_divisor,0.00203054060713255)
    self.assertEqual(b.bam_mlp_write_content_divisor,1.)

  def check_all_gradients(self, cls):
    cfg=self.tiny(cls)
    pcfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),
        bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True,
        bam_pallas_packed_read_keys=True,bam_splash_attention=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    models=[Transformer(c,mesh,quant=None) for c in (cfg,pcfg)]
    tok=jax.random.randint(jax.random.key(3),(1,128),0,32)
    pos=(jnp.arange(128)%64)[None];seg=jnp.where(jnp.arange(128)<64,1,2)[None]
    target=(tok+1)%32;mask=jnp.ones_like(tok);make=splash.make_splash_mha
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(
        splash,'make_splash_mha',side_effect=lambda *a,**kw:make(*a,**dict(kw,interpret=True))):
      params=nn.unbox(models[0].init(jax.random.key(4),tok,pos,target,mask,seg,enable_dropout=False))
      assert not any('memory_mlp' in '/'.join(p) for p in flatten_dict(params))
      control=jax.random.normal(jax.random.key(5),(1,128))
      def forward(m,p):return m.apply(p,tok,pos,target,mask,seg,enable_dropout=False)[0].astype(jnp.float32)
      result=[]
      for m in models:
        fn=jax.jit(jax.value_and_grad(lambda p:jnp.mean(forward(m,p)*control)))
        value,grads=fn(params);result.append((value,flatten_dict(grads)))
      (x,g),(y,h)=result
      np.testing.assert_allclose(x,y,rtol=3e-4,atol=2e-6)
      self.assertEqual(g.keys(),h.keys())
      for p,v in g.items():np.testing.assert_allclose(v,h[p],rtol=4e-4,atol=3e-6,err_msg='/'.join(p))
      self.assertTrue(any('mlp_address' in '/'.join(p) and float(jnp.linalg.norm(v))>0 for p,v in g.items()))
      print('ALL_GRADIENTS_OK',cls,len(g),flush=True)

  def test_norm_scanned_native_pallas_all_gradients(self):self.check_all_gradients(NORM)
  def test_raw_scanned_native_pallas_all_gradients(self):self.check_all_gradients(RAW)

  def test_sparse_writer_has_only_matrix_gradient_path(self):
    cfg=self.tiny(RAW);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    x=jax.random.normal(jax.random.key(21),(1,128,192))*.1
    M=jax.random.normal(jax.random.key(22),(1,128,96,40))*.1
    tok=jnp.ones((1,128),jnp.int32);pos=jnp.arange(128)[None]
    layer=SubDecoderLayer(cfg,mesh,sliding_window_size=128,layer_inx=1)
    args=(x,tok,pos,tok,None,True,'train',0)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      p=nn.unbox(layer.init(jax.random.key(23),*args,M_in=M,is_global=True,layer_index=jnp.asarray(1)))
      def apply(v):return layer.apply(v,*args,M_in=M,is_global=True,layer_index=jnp.asarray(1))
      vector_grad=flatten_dict(jax.grad(lambda v:jnp.sum(apply(v)[0]))(p))
      matrix_grad=flatten_dict(jax.grad(lambda v:jnp.sum(apply(v)[1]**2))(p))
      keys=[k for k in vector_grad if 'mlp' in k and k[-1]=='kernel']
      self.assertTrue(keys)
      self.assertTrue(all(float(jnp.linalg.norm(vector_grad[k]))==0 for k in keys))
      self.assertGreater(sum(float(jnp.linalg.norm(matrix_grad[k])) for k in keys),0)

if __name__=='__main__':unittest.main()

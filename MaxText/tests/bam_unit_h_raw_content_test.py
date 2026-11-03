"""Raw attention-write experiment: fixed embedding doses, budgets, and full-model gradients."""
from types import SimpleNamespace
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_m_prenorm_test import MatrixPreNormTest, PARENT
from layers.models import Transformer

PREFIX=PARENT.removesuffix('TruePile')
RUNS=[PREFIX+s+'TruePile' for s in ('RawWrites','RawWritesEmbeddingFixed','RawWritesEmbeddingSmall20')]
DIVISORS=[1.,.0016052833,.032105666]

class RawContentTest(MLPWriteTest, unittest.TestCase):
  tiny=MatrixPreNormTest.tiny
  model=MatrixPreNormTest.model

  def test_formal_configs_parameter_budget_and_only_intended_changes(self):
    parent=self.config(PARENT)
    pc=parent.get_keys()
    for run,tau in zip(RUNS,DIVISORS):
      cfg=self.config(run); c=cfg.get_keys()
      actual={k for k,v in c.items() if v!=pc[k]}
      expected={'exp_class','model_name','compare_runs','bam_attention_write_content_divisor'}
      if tau!=pc['bam_embedding_write_content_divisor']:expected.add('bam_embedding_write_content_divisor')
      self.assertEqual(actual,expected)
      self.assertEqual((cfg.bam_embedding_write_content_divisor,cfg.bam_attention_write_content_divisor,cfg.bam_mlp_write_content_divisor),(tau,1.,1.))
      self.assertEqual(cfg.bam_common_vo_read_scale_init,1.)
      self.assertEqual(cfg.bam_m_read_norm,'none')
      self.assertEqual(cfg.compare_runs,[PARENT])
      self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_write and cfg.bam_pallas_fused_inputs)
      self.assertTrue(cfg.scan_layers and not cfg.bam_pair_scan)
      self.assertEqual(cfg.bam_attention_write_layers,list(range(17)))
      self.assertTrue(cfg.bam_skip_final_mlp_write)
      self.assertEqual((cfg.mlp_dim,cfg.bam_memory_mlp_dim,cfg.bam_last_mlp_dim),(2510,1128,3904))
      self.assertTrue(cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics or cfg.bam_record_m_shift_health)
      self.assertEqual(cfg.tensorboard_gcs_rotate_period,500)
      self.assertEqual((cfg.sa_block_q,cfg.sa_block_kv,cfg.sa_block_kv_compute),(1024,1024,512))
      self.assertTrue(cfg.sa_use_fused_bwd_kernel)
      cfg.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      shaped,*_=train_compile.get_shaped_inputs(mesh,cfg)
      total=sum(v.size for v in jax.tree.leaves(nn.unbox(shaped[0].params)))
      self.assertEqual(total,432112210)
      print('FORMAL_OK',run,tau,total,flush=True)

  def test_embedding_seed_matrix_doses_and_parameter_initializations(self):
    from layers.models import EmbeddingBamWrite
    # Direct writer probe avoids confusing embedding scale with subsequent feedback.
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];labels=jnp.roll(tok,-1,axis=1)
    matrices=[];trees=[]
    for run in RUNS:
      cfg=self.tiny(run)
      seen=[]
      def capture(next_fun,args,kwargs,context):
        out=next_fun(*args,**kwargs)
        if isinstance(context.module,EmbeddingBamWrite) and context.method_name=='__call__':seen.append(out)
        return out
      with self.model(cfg) as m,nn.intercept_methods(capture):
        p=nn.unbox(m.init(jax.random.key(8),tok,pos,labels,jnp.ones_like(tok),enable_dropout=False))
      self.assertEqual(len(seen),1)
      matrices.append(np.asarray(seen[0]));trees.append(p)
    for p in trees[1:]:
      self.assertEqual(jax.tree.structure(p),jax.tree.structure(trees[0]))
      for a,b in zip(jax.tree.leaves(trees[0]),jax.tree.leaves(p)):np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(matrices[0],matrices[1]*DIVISORS[1],rtol=2e-5,atol=1e-10)
    np.testing.assert_allclose(matrices[2],matrices[1]/20,rtol=2e-5,atol=1e-9)

  def check_pallas(self,index):
    cfg=self.tiny(RUNS[index]);cfg.get_keys().update(max_target_length=128,max_prefill_predict_length=128,query_chunk_size=64)
    pcfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True,bam_splash_attention=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    native,fused=Transformer(cfg,mesh,quant=None),Transformer(pcfg,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(7),(1,128),0,32);pos=jnp.arange(128)[None]
    labels=jnp.roll(tok,-1,axis=1);mask=(pos<127).astype(jnp.float32)
    make=splash.make_splash_mha
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=lambda *a,**k:make(*a,**dict(k,interpret=True))):
      args=(tok,pos,labels,mask)
      p=nn.unbox(native.init(jax.random.key(10),*args,enable_dropout=False))
      pp=nn.unbox(fused.init(jax.random.key(10),*args,enable_dropout=False))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
      # Check away from initialization so scalar gradients cannot hide behind identity.
      p=jax.tree_util.tree_map_with_path(lambda path,v:v*.8 if path[-1].key=='common_vo_read_scale' else v,p)
      def loss(m,p):
        ce=m.apply(p,*args,enable_dropout=False)[0].astype(jnp.float32)
        return jnp.sum(ce*mask)/jnp.sum(mask)
      a,g=jax.value_and_grad(lambda p:loss(native,p))(p)
      b,h=jax.value_and_grad(lambda p:loss(fused,p))(p)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves((a,b,g,h))))
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    fg,fh=flatten_dict(g),flatten_dict(h);self.assertEqual(fg.keys(),fh.keys())
    errors={k:float(jnp.max(jnp.abs(v-fh[k]))/(jnp.max(jnp.abs(v))+1e-12)) for k,v in fg.items()}
    worst=max(errors,key=errors.get)
    self.assertLess(errors[worst],1e-3,str(worst))
    self.assertGreater(sum(float(jnp.linalg.norm(v)) for k,v in fg.items() if k[-1]=='common_vo_read_scale'),0.)
    print('NATIVE_PALLAS_CE_ALL_GRADS_OK',index,float(a),errors[worst],worst,flush=True)

  def test_raw_embedding_native_pallas_all_gradients(self):self.check_pallas(0)
  def test_fixed_embedding_native_pallas_all_gradients(self):self.check_pallas(1)
  def test_small20_embedding_native_pallas_all_gradients(self):self.check_pallas(2)

if __name__=='__main__':unittest.main()

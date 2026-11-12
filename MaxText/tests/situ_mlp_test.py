"""SiTU ordinary-MLP formulas, all gradients, budgets and private-writer isolation."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import linears
from layers.models import Transformer

PARENT = 'BamMediumPropV5'
RUN = PARENT+'SiTU47'
BASE = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneMPreNormQKScaleTruePile'


class SiTUMLPTest(MLPWriteTest, unittest.TestCase):
  def test_scalar_formula_and_both_partial_derivatives(self):
    u = jnp.asarray([-30.,-7.,-1.,0.,.5,1.,4.,7.,10.,30.])
    v = jnp.asarray([7.,-7.,1.,2.,2.,1.,7.,2.,-10.,30.])
    f = lambda u,v: linears._situ_activation(u,0,4.,7.)*linears._situ_activation(v,1,4.,7.)
    h = np.asarray(jax.jit(f)(u,v));du,dv = jax.jit(jax.grad(lambda a,b:jnp.sum(f(a,b)),argnums=(0,1)))(u,v)
    a,b=np.asarray(u,dtype=np.float64),np.asarray(v,dtype=np.float64)
    s=1/(1+np.exp(-a));t=np.tanh(a/4);q=np.tanh(b/7)
    gate,up=4*t*s,7*q
    np.testing.assert_allclose(h,gate*up,rtol=3e-6,atol=1e-6)
    np.testing.assert_allclose(du,((1-t*t)*s+4*t*s*(1-s))*up,rtol=2e-5,atol=1e-6)
    np.testing.assert_allclose(dv,gate*(1-q*q),rtol=2e-5,atol=1e-6)
    self.assertTrue(np.isfinite(h).all());self.assertLessEqual(np.max(np.abs(h)),28.)

  def test_dense_mlp_matches_independent_forward_and_every_gradient(self):
    for fused in (False,True):
      cfg=SimpleNamespace(deep_embed_type='',fused_mlp=fused,activations_in_float32=True,
                          matmul_precision='highest',mgate_dim=1)
      module=linears.MlpBlock(cfg,intermediate_dim=6,activations=('silu','linear'),
          intermediate_dropout_rate=0.,dtype=jnp.float32,weight_dtype=jnp.float32,
          situ_gate_limit=4.,situ_up_limit=7.)
      x=jnp.arange(12,dtype=jnp.float32).reshape(1,3,4)/3-2
      p=nn.unbox(module.init(jax.random.key(9),x,deterministic=True))
      def oracle(p,x):
        w=p['params']
        if fused:uv=x@w['wi']['kernel'].reshape(4,12);u,v=uv.reshape(1,3,2,6).transpose(2,0,1,3)
        else:u,v=x@w['wi_0']['kernel'],x@w['wi_1']['kernel']
        gate=4*jnp.tanh(u/4)*jax.nn.sigmoid(u)
        return (gate*7*jnp.tanh(v/7))@w['wo']['kernel']
      out=module.apply(p,x,deterministic=True)
      np.testing.assert_allclose(out,oracle(p,x),rtol=2e-6,atol=2e-6)
      f=lambda p,x:jnp.sum(module.apply(p,x,deterministic=True)**2)
      g=lambda p,x:jnp.sum(oracle(p,x)**2)
      actual=jax.grad(f,argnums=(0,1))(p,x);expected=jax.grad(g,argnums=(0,1))(p,x)
      for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)

  def test_full_shape_budget_and_formal_contract(self):
    totals=[]
    for name in (PARENT,RUN):
      c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',
          enable_checkpointing=False,base_output_directory=self.tmp.name+'/',jax_cache_dir='',
          log_config=False,dataset_type='synthetic',max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      shaped,*_=train_compile.get_shaped_inputs(mesh,c)
      totals.append(sum(np.prod(v.shape,dtype=np.int64) for v in flatten_dict(nn.unbox(shaped[0].params)).values()))
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertEqual((c.steps,c.learning_rate_schedule_steps,c.per_device_batch_size),(13500,13500,16.))
      self.assertTrue(c.scan_layers and c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_packed_read_keys)
      self.assertEqual(c.remat_policy,'full');self.assertFalse(getattr(c,'bam_splash_residual_name',''))
      self.assertTrue(c.record_training_health_metrics)
      self.assertFalse(c.record_internal_nn_metrics or c.bam_record_m_shift_health)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      if name==RUN:
        self.assertEqual((c.ordinary_mlp_situ_gate_limit,c.ordinary_mlp_situ_up_limit),(4.,7.))
        self.assertEqual(c.compare_runs,[BASE])
    self.assertEqual(totals,[432112228,432112228]);print('FORMAL_PARAMETER_COUNTS',totals,flush=True)

  def test_scanned_ordinary_and_terminal_only_private_writer_unchanged(self):
    configs=[]
    for name in (PARENT,RUN):
      c=self.config(name,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,head_dim=75,num_decoder_layers=3,
          mlp_dim=24,vocab_size=32,bam_k=75,bam_v=32,bam_abs_v_compression_dim=2,
          bam_local_qk_c8_rank=2,bam_partial_rope_nope_dim=75,bam_local_qk_col_output_dim=75,
          bam_standard_qk_dim=18,bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],bam_memory_mlp_dim=12,
          bam_last_mlp_dim=24,emb_bam_num_head=2,bam_write_v_bottleneck_dim=4,
          emb_bam_v_bottleneck_dim=4,bam_mlp_write_address_rank=4,bam_local_full_m_read_bottleneck_dim=4,
          bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False)
      configs.append(c)
    tokens=jnp.array([[1,2,3,4]],jnp.int32);positions=jnp.arange(4)[None];seg=jnp.ones_like(tokens);target=tokens+1
    parameter_sets=[]
    for c in configs:
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        model=Transformer(c,mesh,quant=None)
        p=nn.unbox(model.init(jax.random.key(5),tokens,positions,seg,target));parameter_sets.append(p)
        seen=[]
        def intercept(next_fun,args,kwargs,ctx):
          m=ctx.module
          if ctx.method_name=='__call__' and isinstance(m,linears.MlpBlock):
            seen.append((m.name,m.situ_gate_limit,m.situ_up_limit))
          return next_fun(*args,**kwargs)
        with nn.intercept_methods(intercept):
          loss,grad=jax.value_and_grad(lambda p:jnp.mean(model.apply(p,tokens,positions,seg,target)[0]))(p)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
        ordinary=[x for x in seen if x[0]=='mlp'];private=[x for x in seen if x[0]=='memory_mlp']
        self.assertGreaterEqual(len(ordinary),2);self.assertTrue(private)
        expected=(c.ordinary_mlp_situ_gate_limit,c.ordinary_mlp_situ_up_limit)
        self.assertTrue(all(x[1:]==expected for x in ordinary))
        self.assertTrue(all(x[1:]==(0.,0.) for x in private))
    a,b=[flatten_dict(p) for p in parameter_sets];self.assertEqual(a.keys(),b.keys())
    for name in a:np.testing.assert_array_equal(a[name],b[name],err_msg=str(name))


if __name__=='__main__':unittest.main()

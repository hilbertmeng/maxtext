"""2.8B dynamic-O scale: exact budget, 32-layer health, full100-D QKNorm fused parity."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention

PARENT = 'BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1AllR640SeparateVOKeysLogitDiagNegativeQK80'
RUN = PARENT + 'LocalODynamicNormScale020'


class PythiaDynamicOScaleTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budget_configuration_and_gamma_shapes(self):
    c = self.config(RUN)
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_ = train_compile.get_shaped_inputs(mesh,c)
    flat = flatten_dict(nn.unbox(args[0].params))
    extra = {p:v for p,v in flat.items() if p[-1]=='local_o_dynamic_norm_scale'}
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),2775190880+65536)
    self.assertEqual(sum(math.prod(v.shape) for v in extra.values()),65536)
    self.assertEqual((c.num_decoder_layers,c.num_query_heads,c.bam_v,c.bam_k),(32,32,64,80))
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(7808,2944,9984))
    self.assertEqual((c.bam_local_full_m_read_bottleneck_dim,c.bam_write_v_bottleneck_dim,
                      c.bam_mlp_write_address_rank,c.emb_bam_v_bottleneck_dim),(640,)*4)
    self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat and c.bam_attention_negative_diagonal)
    self.assertTrue(c.bam_local_vo_independent_keys and c.bam_local_vo_independent_gates)
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_fused_inputs)
    self.assertEqual((c.bam_pallas_body,c.bam_pallas_vmem_mib,c.remat_policy),('v8t',60,'minimal'))
    self.assertTrue(c.scan_layers and not c.bam_pair_scan)
    self.assertEqual(c.bam_final_local_layer_count,1)
    self.assertEqual(c.bam_local_o_dynamic_norm_scale_init,.2)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    self.assertEqual(c.compare_runs,[PARENT])
    self.assertEqual((c.steps,c.learning_rate_schedule_steps),(28000,28000))
    self.assertFalse(c.load_parameters_path or c.load_full_state_path)
    print('PYTHIA_GAMMA020_BUDGET_OK 2775256416 extra65536 MLP7808/2944/9984',flush=True)

  def test_32_layer_generic_parameter_and_raw_gradient_health(self):
    c = self.config(RUN)
    shape=[32,64];shape.insert(c.param_scan_axis,31)
    def tree(value):
      body={'block':{'self_attention':{'local_o_dynamic_norm_scale':jnp.full(shape,value)}}}
      tail={'block':{'self_attention':{'local_o_dynamic_norm_scale':jnp.full((32,64),value)}}}
      return {'params':{'decoder':{'layers':body,'final_local_layer':tail}}}
    p = train.compute_params_norm(tree(.2),c,prefix='total_params')
    g = train.compute_params_norm(tree(.001),c,prefix='raw_grads')
    self.assertEqual(len(p),32);self.assertEqual(len(g),32)
    for i in range(31):
      self.assertTrue(any(f'/layers_{i}/' in key for key in p))
    self.assertTrue(any('/final_local_layer/' in key for key in p))
    np.testing.assert_allclose(list(p.values()),.2*math.sqrt(2048),rtol=1e-5)
    np.testing.assert_allclose(list(g.values()),.001*math.sqrt(2048),rtol=1e-5)
    wd=train.get_wd_tree(c,tree(.2))
    for value in flatten_dict(wd).values():self.assertEqual(value,0.)
    print('PYTHIA_GAMMA_HEALTH_OK 32 parameter +32 rawgradient; RMS /sqrt2048',flush=True)

  def test_32_head_qk100_native_pallas_outputs_and_all_gradients(self):
    c = self.config(RUN,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,
        bam_local_full_m_read_bottleneck_dim=8,bam_write_v_bottleneck_dim=8,
        max_target_length=128,max_prefill_predict_length=128,query_chunk_size=128)
    pc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    def module(cfg):
      return BamAttention(config=cfg,num_query_heads=32,num_kv_heads=32,head_dim=80,
          bam_k=80,bam_v=64,read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=31,
          max_target_length=128,max_prefill_predict_length=128,mesh=mesh,
          attention_kernel='dot_product_chunk',dtype=cfg.dtype)
    native,fused=module(c),module(pc)
    x=jax.random.normal(jax.random.key(1),(1,128,2560))
    matrix=jax.random.normal(jax.random.key(2),(1,128,80,64))
    seg=jnp.where(jnp.arange(128)<64,1,2)[None];pos=(jnp.arange(128)%64)[None]
    args=(x,x,pos,seg)
    original=splash.make_splash_mha
    def interpret(*a,**kw):return original(*a,**dict(kw,interpret=True))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
      p=nn.unbox(native.init(jax.random.key(3),*args,M_in=matrix))
      q=nn.unbox(fused.init(jax.random.key(3),*args,M_in=jnp.swapaxes(matrix,-1,-2)))
      self.assertEqual(jax.tree.structure(p),jax.tree.structure(q))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(q)):np.testing.assert_array_equal(a,b)
      flat=flatten_dict(p)
      gamma=next(k for k in flat if k[-1]=='local_o_dynamic_norm_scale')
      self.assertEqual(flat[gamma].shape,(32,64))
      np.testing.assert_array_equal(flat[gamma],jnp.full((32,64),.2))
      self.assertEqual(flat[('params','qk_norm','q_norm','scale')].shape,(100,))
      static=next(k for k in flat if k[-1]=='static_o_key')
      flat[gamma]=flat[gamma]+.05*jax.random.normal(jax.random.key(4),flat[gamma].shape)
      flat[static]=.02*jax.random.normal(jax.random.key(5),flat[static].shape)
      p=unflatten_dict(flat)
      target=jax.random.normal(jax.random.key(6),(1,128,2560))
      def loss(mod,p,m,x):
        matrix_input=jnp.swapaxes(m,-1,-2) if mod is fused else m
        y,carry=mod.apply(p,x,x,pos,seg,M_in=matrix_input,deterministic=True)[:2]
        return jnp.mean(y*target)+.01*jnp.mean(carry**2),y
      (_,a),ga=jax.value_and_grad(lambda p,m,x:loss(native,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,x)
      (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(fused,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,x)
    np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    errors=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(ga)[0],jax.tree.leaves(gb)):
      err=float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12));errors.append((err,jax.tree_util.keystr(path)))
    self.assertGreater(float(jnp.max(jnp.abs(flatten_dict(ga[0])[gamma]))),0.)
    worst=max(errors)
    self.assertLess(worst[0],1e-3,worst)
    print('PYTHIA_QK100_N32_V64_GAMMA020_PALLAS_ALL_GRAD_OK',worst,flush=True)


if __name__=='__main__':unittest.main()

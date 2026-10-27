"""Learned local-O gain: parent identity, write semantics, scan gradients and health."""
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

import exp
import max_utils
import pyconfig
import train
import train_compile
from bam_config import validate_bam_config
from bam_local_o_regularization_test import LocalORegularizationTest, PARENT
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention
from layers.models import Transformer

HALF = PARENT.replace('TruePile', 'LocalOScale050TruePile')
ONE = PARENT.replace('TruePile', 'LocalOScale100TruePile')


class LocalOScaleTest(MLPWriteTest, unittest.TestCase):
  fixture = LocalORegularizationTest.fixture

  def test_formal_budget_and_settings(self):
    for run, initial in ((HALF, .5), (ONE, 1.)):
      cfg = self.config(run)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(args[0].params))
      gains = [v for k, v in flat.items() if k[-1] == 'local_o_scale']
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432112192 + 18)
      self.assertEqual(sum(math.prod(v.shape) for v in gains), 18)
      self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (2510,1128,3904))
      self.assertEqual(cfg.bam_local_o_scale_init, initial)
      self.assertTrue(cfg.bam_pallas_core and cfg.scan_layers and cfg.sa_use_fused_bwd_kernel)
      self.assertTrue(cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics or cfg.bam_record_concat_health)
      self.assertEqual(cfg.bam_local_o_drop_head_rate, 0.)
      self.assertEqual(cfg.bam_local_o_read_weight_dropout_rate, 0.)
      self.assertEqual(cfg.train_no_dropout_loss_interval, 0)
      self.assertEqual(cfg.compare_runs, [PARENT])

  def test_gain_one_parent_identity_and_unchanged_init_stream(self):
    fixtures = [self.fixture(run) for run in (PARENT, HALF, ONE)]
    reference = flatten_dict(fixtures[0][3])
    for run, fixture in zip((PARENT, HALF, ONE), fixtures):
      cfg, mesh, module, variables, args, matrix = fixture
      flat = flatten_dict(variables)
      for key, value in reference.items():
        np.testing.assert_array_equal(flat[key], value)
      if run != PARENT:
        np.testing.assert_array_equal(variables['params']['local_o_scale'],
                                      [.5 if run == HALF else 1.])
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        actual = module.apply(variables, *args, M_in=matrix, deterministic=True)
      if run == PARENT:
        expected = actual
      elif run == ONE:
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
          np.testing.assert_array_equal(a,b)

  def test_scale_covers_static_dynamic_output_and_write_input(self):
    cfg, mesh, module, variables, args, matrix = self.fixture(HALF)
    flat = flatten_dict(variables)
    static = next(k for k in flat if k[-1] == 'static_o_key')
    flat[static] = jax.random.normal(jax.random.key(40), flat[static].shape) * .1
    gain = next(k for k in flat if k[-1] == 'local_o_scale')
    captured = []
    original = BamAttention._deferred_write_factors
    def capture(self, head, *a, **kw):
      captured.append(head)
      return original(self, head, *a, **kw)
    outputs = []
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        BamAttention, '_deferred_write_factors', capture):
      for value in (0., 1., .5):
        p = unflatten_dict(dict(flat))
        p['params']['local_o_scale'] = jnp.asarray([value])
        outputs.append(module.apply(p, *args, M_in=matrix, deterministic=True, defer_write=True))
    self.assertEqual(len(captured), 3)
    np.testing.assert_allclose(captured[2], (captured[0]+captured[1])*.5, rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(outputs[2][0], (outputs[0][0]+outputs[1][0])*.5, rtol=1e-5,atol=1e-6)
    for actual, head in zip(outputs, captured):
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        expected = module.apply(p, head, args[0], method=module._deferred_write_factors)
      for a,b in zip(jax.tree.leaves(actual[2]),jax.tree.leaves(expected)):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)

  def test_scan_native_pallas_logits_and_all_gradients(self):
    cfg = self.config(HALF, base_emb_dim=150, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    cfg.get_keys().update(bam_pallas_core=False,bam_splash_attention=False,bam_memory_mlp_dim=16,
        bam_last_mlp_dim=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,max_target_length=128,max_prefill_predict_length=128,
        query_chunk_size=64)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_pallas_core=True,
        bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    native,fused = Transformer(cfg,mesh,quant=None),Transformer(pcfg,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(20),(1,128),0,32)
    seg=jnp.where(jnp.arange(128)<64,1,2)[None];pos=(jnp.arange(128)%64)[None]
    original=splash.make_splash_mha
    def interpret(*a,**kw):
      return original(*a,**dict(kw,interpret=True))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
      p,pp=(nn.unbox(m.init(jax.random.key(21),tok,pos,seg,tok)) for m in (native,fused))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):
        np.testing.assert_array_equal(a,b)
      def call(m,p):
        return m.apply(p,tok,pos,seg,tok,enable_dropout=False)[0]
      target=jax.random.normal(jax.random.key(23),call(native,p).shape)
      def loss(m,p):
        y=call(m,p)
        return jnp.mean(y*target),y
      (_,a),ga=jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
      (_,b),gb=jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(pp)
    np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    errors=[float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12))
        for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb))]
    self.assertLess(max(errors),1e-3)
    gain_grads=[v for k,v in flatten_dict(gb).items() if k[-1]=='local_o_scale']
    self.assertEqual(sum(v.size for v in gain_grads),3)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in gain_grads))
    self.assertGreater(sum(float(jnp.sum(jnp.abs(v))) for v in gain_grads),0.)
    print('LOCAL_O_SCALE_SCAN_NATIVE_PALLAS_ALL_GRADIENTS_OK',max(errors),flush=True)

  def test_signed_preupdate_health_and_no_decay(self):
    params={'params':{'decoder':{'layers':{'block':{'self_attention':{
        'local_o_scale':jnp.linspace(-.5,1.5,17)[None], 'kernel':jnp.ones((1,17))}}},
        'final_local_layer':{'block':{'self_attention':{'local_o_scale':jnp.asarray([-.25])}}}}}}
    cfg=SimpleNamespace(scan_layers=True,param_scan_axis=1,wd_mults=getattr(exp,HALF).wd_mults,adam_weight_decay=.1)
    values=train.compute_params_norm(params,cfg)
    gains={k:v for k,v in values.items() if k.startswith('bam/local_o_scale/')}
    self.assertEqual(len(gains),18)
    self.assertTrue(any(float(v)==-.5 for v in gains.values()))
    self.assertTrue(any(float(v)==-.25 for v in gains.values()))
    self.assertFalse(any(k.startswith('bam/local_o_scale/') for k in train.compute_params_norm(params,cfg,'raw_grads')))
    decay=flatten_dict(train.get_wd_tree(cfg,params))
    self.assertTrue(all(float(v)==0. for k,v in decay.items() if k[-1]=='local_o_scale'))
    self.assertEqual(decay[('params','decoder','layers','block','self_attention','kernel')],.1)

  def test_bf16_output_and_gain_gradient(self):
    cfg,mesh,module,variables,args,matrix=self.fixture(HALF,dtype='bfloat16')
    def loss(p):
      y,m=module.apply(p,*args,M_in=matrix,deterministic=True)
      self.assertEqual(y.dtype,jnp.bfloat16)
      self.assertEqual(m.dtype,jnp.bfloat16)
      return jnp.mean(y.astype(jnp.float32)**2)+jnp.mean(m.astype(jnp.float32)**2)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      g=jax.grad(loss)(variables)
    self.assertTrue(bool(jnp.all(jnp.isfinite(g['params']['local_o_scale']))))
    self.assertGreater(float(jnp.sum(jnp.abs(g['params']['local_o_scale']))),0.)

  def test_validation(self):
    for invalid in (float('nan'),float('inf'),True,'0.5'):
      with self.assertRaisesRegex(ValueError,'finite scalar'):
        validate_bam_config(dict(bam_enabled=True,bam_local_o_learned_scale=True,bam_local_o_scale_init=invalid))


if __name__ == '__main__':
  unittest.main()

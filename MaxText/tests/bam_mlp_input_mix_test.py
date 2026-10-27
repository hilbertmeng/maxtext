"""Two input-gate parameterizations crossed with private write-control source."""
import math
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
from bam_standard_value_test import StandardValueTest
from common_types import MODEL_MODE_TRAIN
from layers import fusion, quantizations
from layers.models import Transformer

PARENT = ('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
          'LogitDiagNegativeQK75TruePile')
PREFIX = PARENT.removesuffix('TruePile')
RUNS = [PREFIX + suffix + 'TruePile' for suffix in (
    'MLPInputMixG099WriteControlH', 'MLPInputMixG099WriteControlMix',
    'MLPInputMixG100Scale2WriteControlH', 'MLPInputMixG100Scale2WriteControlMix')]


class MLPInputMixTest(MLPWriteTest, unittest.TestCase):
  variant = 0

  def tiny(self, period=3, keep=True, length=4):
    c = self.config(RUNS[self.variant], base_emb_dim=150, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=3, base_mlp_dim=32,
        vocab_size=32, dtype='float32', weight_dtype='float32', bam_splash_attention=False)
    c.get_keys().update(bam_pallas_core=False, mlp_dim=32, bam_last_mlp_dim=32,
        bam_memory_mlp_dim=16, bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1],
        bam_write_v_bottleneck_dim=8, emb_bam_num_head=2, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8,
        max_target_length=length, max_prefill_predict_length=length, query_chunk_size=length)
    return c

  def test_formal_four_budgets_and_no_unused_terminal_gates(self):
    trees = []
    for run in RUNS:
      c = self.config(run)
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432154227)
      gates = {p:v for p,v in flat.items() if any('mlp_input_mix_gate' in x for x in p)}
      self.assertEqual(sum(math.prod(v.shape) for v in gates.values()), 42035)
      terminal = {p:v for p,v in gates.items() if 'final_local_layer' in p}
      self.assertEqual(sum(math.prod(v.shape) for v in terminal.values()), 1201)
      self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (2510,1128,3904))
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs)
      self.assertEqual((c.bam_pallas_body, c.bam_pallas_vmem_mib), ('v8t',48))
      self.assertFalse(c.bam_record_concat_health or c.bam_record_m_shift_health)
      trees.append({p:v.shape for p,v in flat.items()})
    self.assertTrue(all(t == trees[0] for t in trees))
    print('MLP_INPUT_MIX_FOUR_BUDGETS_OK 432154227 extra42035', flush=True)

  def test_real_inputs_norm_sharing_write_control_and_outer_residual(self):
    for variant in range(4):
      self.variant = variant
      c = self.tiny()
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      block = fusion.SubDecoderLayer(c, mesh, quantizations.configure_quantization(c), layer_inx=0)
      x = jax.random.normal(jax.random.key(10), (1,4,150))
      matrix = jax.random.normal(jax.random.key(11), (1,4,75,32))
      seg = jnp.ones((1,4), jnp.int32)
      args = (x,seg,jnp.arange(4)[None],seg,None,True,MODEL_MODE_TRAIN,None)
      seen = {}
      def capture(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if context.method_name == '__call__' and context.module.name in (
            'self_attention','post_self_attention_layer_norm','mlp_input_mix_gate',
            'mlp','memory_mlp','mlp_address_down','mlp_write_gate'):
          seen.setdefault(context.module.name, []).append((args, kwargs, out))
        return out
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        params = nn.unbox(block.init(jax.random.key(12), *args, M_in=matrix))
        self.assertEqual(set(params['params']['post_self_attention_layer_norm']), {'scale'})
        gate = params['params']['mlp_input_mix_gate']['kernel']
        np.testing.assert_array_equal(gate, jnp.zeros_like(gate))
        expected_bias = math.log(.99/.01) if variant < 2 else 0.
        np.testing.assert_allclose(params['params']['mlp_input_mix_gate_bias'], expected_bias, rtol=1e-6)
        params['params']['mlp_input_mix_gate']['kernel'] = .05*jax.random.normal(jax.random.key(13), gate.shape)
        params['params']['mlp_input_mix_gate_bias'] = jnp.array([-1.,.4])
        with nn.intercept_methods(capture):
          output = block.apply(params, *args, M_in=matrix)
      norm = seen['post_self_attention_layer_norm']
      self.assertEqual(len(norm), 3)
      a = seen['self_attention'][0][2][0]
      np.testing.assert_allclose(norm[0][0][0], x+a, rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(seen['mlp_input_mix_gate'][0][0][0], norm[0][2])
      z = norm[0][2] @ params['params']['mlp_input_mix_gate']['kernel']
      g = c.bam_mlp_input_mix_gate_scale*jax.nn.sigmoid(z+params['params']['mlp_input_mix_gate_bias'])
      np.testing.assert_allclose(norm[1][0][0], g[...,0:1]*x+a, atol=1e-6)
      np.testing.assert_allclose(norm[2][0][0], g[...,1:2]*x+a, atol=1e-6)
      np.testing.assert_array_equal(seen['mlp'][0][0][0], norm[1][2])
      np.testing.assert_array_equal(seen['memory_mlp'][0][0][0], norm[2][2])
      control = norm[2][2] if variant % 2 else norm[0][2]
      np.testing.assert_array_equal(seen['mlp_address_down'][0][0][0], control)
      np.testing.assert_array_equal(seen['mlp_write_gate'][0][0][0], control)
      np.testing.assert_allclose(output[0], x+a+seen['mlp'][0][2], rtol=1e-6, atol=1e-6)

  def test_exact_unit_opening_parent_initial_function_and_parameters(self):
    self.variant = 2
    c = self.tiny()
    old = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        bam_mlp_input_mix=False, bam_record_mlp_input_mix_health=False)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    parent = Transformer(old,mesh,quant=None)
    model = Transformer(c,mesh,quant=None)
    tok = jnp.array([[1,2,3,4]],jnp.int32)
    pos = jnp.arange(4)[None]
    seg = jnp.ones_like(tok)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      original = nn.unbox(parent.init(jax.random.key(21),tok,pos,seg,tok))
      params = nn.unbox(model.init(jax.random.key(21),tok,pos,seg,tok))
      left = flatten_dict(original['params'])
      right = flatten_dict(params['params'])
      for path in left:
        np.testing.assert_array_equal(left[path],right[path],err_msg=str(path))
      expected = parent.apply(original,tok,pos,seg,tok,enable_dropout=False)[0]
      for controls in (False,True):
        cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
            bam_mlp_input_mix_write_controls=controls)))
        actual = Transformer(cfg,mesh,quant=None).apply(params,tok,pos,seg,tok,enable_dropout=False)[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)

  def test_actual_health_collection_and_export(self):
    from train import record_bam_mlp_input_mix_health_metrics
    c = self.tiny()
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    model = Transformer(c,mesh,quant=None)
    tok = jnp.array([[1,2,3,4]],jnp.int32)
    pos = jnp.arange(4)[None]
    seg = jnp.ones_like(tok)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      params = nn.unbox(model.init(jax.random.key(22),tok,pos,seg,tok))
      _, health = model.apply(params,tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
    metrics = {'scalar':{}}
    record_bam_mlp_input_mix_health_metrics(metrics,health,c)
    self.assertEqual(len(metrics['scalar']),10)
    self.assertFalse(any('memory/layer_002' in key for key in metrics['scalar']))
    for key,value in metrics['scalar'].items():
      self.assertAlmostEqual(float(value), 1./1.99 if key.endswith('mean') else 0., places=6)
    print('MLP_INPUT_MIX_HEALTH_REAL_SCAN_EXPORT_OK', flush=True)

  def test_pallas_sigmoid_original_controls(self):
    self.variant = 0
    StandardValueTest.check_pallas(self, True)

  def test_pallas_sigmoid_mixed_controls(self):
    self.variant = 1
    StandardValueTest.check_pallas(self, True)

  def test_pallas_scale2_original_controls(self):
    self.variant = 2
    StandardValueTest.check_pallas(self, True)

  def test_pallas_scale2_mixed_controls(self):
    self.variant = 3
    StandardValueTest.check_pallas(self, True)


if __name__ == '__main__':
  unittest.main()

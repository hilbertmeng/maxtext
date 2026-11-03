"""H transfer: target budget/scan/health and GELU+QKNorm native/Pallas gradients (self-contained)."""
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
from layers.models import Transformer

BASE = ('BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1'
        'AllR640SeparateVOKeysLogitDiagNegativeQK80')
REFERENCE = BASE + 'LocalODynamicNormScale020'
H = ('BamPythia2p8BShortTrainSeparateVOKeysLogitDiagNegativeQK80'
     'CommonReadScaleUnscaledMLPContentEmbeddingFixed')
GAMMA = H + 'LocalODynamicNormScale020'
UNIT = H + 'ReadScaleInitOneLocalODynamicNormScale020'


class PythiaCommonReadRawWriteTest(MLPWriteTest, unittest.TestCase):
  def test_target_config_shapes_budget_and_no_unintended_changes(self):
    parent = self.config(REFERENCE)
    for run, gamma, init in [(H, False, None), (GAMMA, True, None), (UNIT, True, 1.)]:
      c = self.config(run)
      changed = {k for k, v in c.get_keys().items() if v != parent.get_keys()[k]}
      expected = {'exp_class', 'model_name', 'bam_common_vo_read_scale',
                  'bam_embedding_write_content_divisor', 'bam_mlp_write_content_divisor', 'compare_runs'}
      if not gamma:
        expected.add('bam_local_o_dynamic_norm_scale')
      if init is not None:
        expected.add('bam_common_vo_read_scale_init')
      self.assertEqual(changed, expected)
      self.assertEqual(c.compare_runs, [REFERENCE])
      self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (7808, 2944, 9984))
      self.assertEqual(c.bam_attention_write_layers, list(range(31)))
      self.assertTrue(c.bam_skip_final_mlp_write and c.scan_layers and not c.bam_pair_scan)
      self.assertEqual(c.bam_final_local_layer_count, 1)
      self.assertEqual(c.bam_common_vo_read_scale_init, init)
      self.assertEqual(c.bam_embedding_write_content_divisor, .0020825863)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertEqual(c.bam_mlp_write_content_divisor, 1.)
      self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat)
      self.assertEqual(c.mlp_activations, ['gelu_exact'])
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_fused_inputs)
      self.assertEqual((c.bam_pallas_body, c.bam_pallas_vmem_mib, c.remat_policy), ('v8t', 60, 'minimal'))
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 2775190880 + 32 + 65536 * gamma)
      self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items() if p[-1] == 'common_vo_read_scale'), 32)
      print('TARGET_CONFIG_BUDGET_OK', run, flush=True)

  def tiny(self, run, length=128, dtype='float32'):
    c = self.config(run, dtype=dtype, weight_dtype='float32')
    c.get_keys().update(emb_dim=160, num_query_heads=2, num_kv_heads=2,
        num_decoder_layers=3, mlp_dim=32, vocab_size=32,
        bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False,
        max_target_length=length, max_prefill_predict_length=length,
        query_chunk_size=64, bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1],
        bam_memory_mlp_dim=16, bam_last_mlp_dim=32, emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_unit_init_scan_values_and_signed_health(self):
    for run in (H, UNIT):
      c = self.tiny(run, length=4)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quant=None)
      tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = nn.unbox(model.init(jax.random.key(1), tok, jnp.arange(4)[None], tok, jnp.ones_like(tok)))
      for path, value in flatten_dict(p).items():
        if path[-1] != 'common_vo_read_scale':
          continue
        expected = np.ones(value.size) if run == UNIT else (
            1 / np.sqrt(1 + 2 * np.arange(2)) if 'layers' in path else [1 / np.sqrt(5)])
        np.testing.assert_allclose(np.asarray(value).reshape(-1), expected, rtol=1e-6)
    c = self.config(UNIT)
    shape = [1]
    shape.insert(c.param_scan_axis, 31)
    tree = {'params': {'decoder': {
        'layers': {'block': {'self_attention': {'common_vo_read_scale': -jnp.ones(shape)}}},
        'final_local_layer': {'block': {'self_attention': {'common_vo_read_scale': -jnp.ones(1)}}}}}}
    signed = {k: v for k, v in train.compute_params_norm(tree, c).items() if k.startswith('parameter_values/')}
    self.assertEqual(len(signed), 32)
    self.assertTrue(all(float(v) == -1. for v in signed.values()))
    self.assertTrue(all(v == 0. for v in flatten_dict(train.get_wd_tree(c, tree)).values()))

  def check_model(self, run, dtype='float32'):
    """Compare per-token losses and every gradient; BF16 against same-parameter F32 truth."""
    cfg = self.tiny(run, length=128, dtype=dtype)
    cfg.get_keys().update(query_chunk_size=64, qk_norm=True)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_pallas_core=True, bam_pallas_write=True,
        bam_pallas_interpret=True, bam_splash_attention=True, bam_pallas_fused_inputs=True)))
    tcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(), dtype=jnp.float32)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    native, fused, truth = (Transformer(c, mesh, quant=None) for c in (cfg, pcfg, tcfg))
    tok = jax.random.randint(jax.random.key(20), (1, 128), 0, 32)
    pos = (jnp.arange(128) % 64)[None]
    seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
    original = splash.make_splash_mha
    def interpret(*args, **kwargs):
      return original(*args, **dict(kwargs, interpret=True))
    training_objective = getattr(self, 'bf16_training_objective', False)
    if training_objective:
      # Use one unpacked training sequence for the precision probe; packed
      # integer segment cotangents in CPU Splash interpret are a separate issue.
      pos = jnp.arange(128)[None]
      labels = jnp.roll(tok, -1, axis=1)
      loss_mask = (pos < 127).astype(jnp.float32)
      call_args, call_kwargs = (tok, pos, labels, loss_mask), {}
    else:
      call_args, call_kwargs = (tok, pos, seg, tok), {}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        splash, 'make_splash_mha', side_effect=interpret):
      params, pp = (nn.unbox(m.init(jax.random.key(21), *call_args, **call_kwargs)) for m in (native, fused))
      for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(pp)):
        np.testing.assert_array_equal(a, b)
      flat = flatten_dict(params)
      for i, path in enumerate(flat):   # break the unit/layer-constant initial scales
        if path[-1] in ('local_o_dynamic_norm_scale', 'common_vo_read_scale', 'common_qk_read_scale'):
          flat[path] = flat[path] * (1 + .3 * jax.random.normal(jax.random.key(70 + i), flat[path].shape))
          if path[-1] == 'common_qk_read_scale':
            # Exercise the pre-norm denominator at zero, fractional and negative scales.
            flat[path] = jnp.asarray([0., .6] if flat[path].size == 2 else [-1.4], flat[path].dtype).reshape(flat[path].shape)
      params = unflatten_dict(flat)
      if dtype == 'bfloat16' and training_objective:
        # Compare arithmetic at exactly the same representable parameter point;
        # otherwise the F32 oracle also changes every weight before the first dot.
        params = jax.tree.map(lambda x: x.astype(jnp.bfloat16).astype(x.dtype), params)
      def call(model, p):
        return model.apply(p, *call_args, enable_dropout=False, **call_kwargs)[0].astype(jnp.float32)
      target = jax.random.normal(jax.random.key(23), call(native, params).shape)
      if training_objective:
        # Transformer returns per-token cross entropies, not logits. Apply the
        # next-token mask and average once, matching the actual training loss.
        def objective(y):
          assert y.shape == tok.shape
          return jnp.sum(y * loss_mask) / jnp.sum(loss_mask)
      else:
        objective = lambda y: jnp.mean(y * target)
      def grads(model):
        return jax.value_and_grad(lambda p: (lambda y: (objective(y), y))(call(model, p)),
                                  has_aux=True)(params)
      (_, a), ga = grads(native)
      (_, b), gb = grads(fused)
      gt = grads(truth)[1] if dtype == 'bfloat16' else None
    fa, fb = flatten_dict(ga), flatten_dict(gb)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves((a, b, ga, gb))))
    for path, value in fa.items():
      if path[-1] in ('common_vo_read_scale', 'local_o_dynamic_norm_scale', 'common_qk_read_scale'):
        self.assertGreater(float(jnp.max(jnp.abs(value))), 0., str(path))
    if dtype == 'float32':
      errors = {k: float(jnp.max(jnp.abs(v - fb[k])) / (jnp.max(jnp.abs(v)) + 1e-12)) for k, v in fa.items()}
      worst = max(errors, key=errors.get)
      print('PYTHIA_H_MODEL_GRAD', run, errors[worst], worst, flush=True)
      self.assertLess(errors[worst], 1e-3, worst)
      np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
      return
    # BF16 rounds at different points on the two paths (the fused read folds the scalar into
    # F32 weights); bound each against same-parameter F32 truth by complete relative L2.
    ft = flatten_dict(gt)
    def rel_l2(g):
      num = sum(float(jnp.sum((g[p].astype(jnp.float32) - ft[p].astype(jnp.float32))**2)) for p in ft)
      return (num / sum(float(jnp.sum(v.astype(jnp.float32)**2)) for v in ft.values()))**.5
    native_error, fused_error = rel_l2(fa), rel_l2(fb)
    print('PYTHIA_H_BF16_F32_GLOBAL_GRAD_L2', run, native_error, fused_error, flush=True)
    control_error = native_error
    if getattr(cfg, 'bam_common_qk_read_scale', False):
      # Preserve every common parameter and the sample/target when measuring the
      # parent norm's own BF16 error. A native-only ratio can confound the new
      # scalar with the fused norm epilogue's existing rounding order.
      control_params = unflatten_dict({p: v for p, v in flatten_dict(params).items()
                                      if p[-1] != 'common_qk_read_scale'})
      def control_grad(c):
        control_cfg = pyconfig.HyperParameters(SimpleNamespace(
            keys=dict(c.get_keys(), bam_common_qk_read_scale=False)))
        control_model = Transformer(control_cfg, mesh, quant=None)
        with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
            splash, 'make_splash_mha', side_effect=interpret):
          return flatten_dict(jax.grad(lambda p: objective(call(control_model, p)))(control_params))
      parent_truth, parent_native, parent_fused = (control_grad(c) for c in (tcfg, cfg, pcfg))
      def parent_rel_l2(g):
        num = sum(float(jnp.sum((g[p].astype(jnp.float32) - parent_truth[p].astype(jnp.float32))**2))
                  for p in parent_truth)
        return (num / sum(float(jnp.sum(v.astype(jnp.float32)**2)) for v in parent_truth.values()))**.5
      control_native_error, control_error = parent_rel_l2(parent_native), parent_rel_l2(parent_fused)
      print('PYTHIA_H_MATCHED_PARENT_BF16_GRAD_L2', control_native_error, control_error, flush=True)
    if training_objective:
      # The full-model precision floor depends on the model and objective. In
      # particular, read pre-norm with a zero local-QK scale has a >5% matched
      # BF16 control error. Retain the 5% ordinary-model allowance, or at most
      # 25% over the measured same-kernel control, rather than rejecting a
      # kernel that is more accurate than its native BF16 reference.
      budget = max(.05, control_error * 1.25)
      print('PYTHIA_TRAIN_CE_GRAD_BUDGET', budget, flush=True)
      self.assertLess(fused_error, budget)
      # Judge BF16 rounding by complete losses and gradients against F32, not
      # elementwise relative error on individual, quantized token losses.
      loss_delta = float(jnp.abs(objective(a) - objective(b)))
      output_rel_l2 = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
      print('PYTHIA_TRAIN_CE_PARITY', loss_delta, output_rel_l2, flush=True)
      self.assertLess(loss_delta, .01)
      self.assertLess(output_rel_l2, .025)
    else:
      # Preserve the existing random-VJP regression's stricter fixed limits.
      self.assertLess(fused_error, .05)
      self.assertLess(fused_error, max(native_error, control_error) * 1.25)
      np.testing.assert_allclose(a, b, rtol=.025, atol=2e-4)

  def test_h_native_pallas_logits_and_all_gradients(self):
    self.check_model(H)

  def test_gamma_native_pallas_logits_and_all_gradients(self):
    self.check_model(GAMMA)

  def test_unit_gamma_native_pallas_logits_and_all_gradients(self):
    self.check_model(UNIT)

  def test_unit_gamma_bf16_gradients_against_f32_truth(self):
    self.check_model(UNIT, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

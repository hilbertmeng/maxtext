"""H descendants: all-read matrix views, dynamic-O gamma, and complete model gradients."""
import contextlib
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
from bam_config import validate_bam_config
from bam_common_read_raw_write_test import CommonReadRawWriteTest
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention
from layers.models import Transformer

H = ('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
     'LogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile')
PREFIX = H.removesuffix('TruePile')
GAMMA = PREFIX + 'LocalODynamicNormScale050TruePile'
ALL = PREFIX + 'AllMReadScaleTruePile'
BOTH = PREFIX + 'AllMReadScaleLocalODynamicNormScale050TruePile'


class CommonAllReadGammaTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, run=ALL, length=4, dtype='float32'):
    self.experiment = run
    cfg = CommonReadRawWriteTest.tiny(self, length=length)
    cfg.get_keys().update(dtype=dtype, bam_pallas_write=False)
    return cfg

  def test_formal_budget_and_isolated_switches(self):
    parent = self.config(H)
    for run, all_read, gamma in ((GAMMA, False, True), (ALL, True, False), (BOTH, True, True)):
      cfg = self.config(run)
      changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys()[k]}
      expected = {'model_name', 'exp_class', 'compare_runs'}
      if all_read:
        expected.add('bam_common_read_scale_qk')
      if gamma:
        expected.update(('bam_local_o_dynamic_norm_scale', 'bam_local_o_dynamic_norm_scale_init'))
      self.assertEqual(changed, expected)
      self.assertEqual(cfg.compare_runs, [H])
      self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (2510, 1128, 3904))
      self.assertEqual(cfg.bam_embedding_write_content_divisor, .0016052833)
      self.assertIsNone(cfg.bam_attention_write_content_divisor)
      self.assertEqual(cfg.bam_mlp_write_content_divisor, 1.)
      self.assertTrue(cfg.scan_layers and cfg.bam_pallas_core and cfg.bam_pallas_write)
      self.assertTrue(cfg.record_training_health_metrics and not cfg.record_internal_nn_metrics)
      self.assertTrue(cfg.sa_use_fused_bwd_kernel)
      self.assertFalse(cfg.qk_norm)
      self.assertEqual(cfg.tensorboard_gcs_rotate_period, 500)
      cfg.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432112210 + 9216 * gamma)
      self.assertEqual(sum(math.prod(v.shape) for k, v in flat.items() if k[-1] == 'common_vo_read_scale'), 18)
      self.assertEqual(sum(math.prod(v.shape) for k, v in flat.items() if k[-1] == 'local_o_dynamic_norm_scale'), 9216 * gamma)

  def test_validation_and_scale_weight_decay(self):
    cfg = self.config(ALL)
    for flags in ({'bam_common_vo_read_scale': False}, {'bam_concat_qk': False}):
      with self.assertRaisesRegex(ValueError, 'concatenated M-derived Q/K'):
        validate_bam_config(dict(cfg.get_keys(), **flags))
    cfg = self.config(BOTH)
    tree = {'params': {'common_vo_read_scale': jnp.ones(1),
                       'local_o_dynamic_norm_scale': jnp.ones((2, 32))}}
    wd = train.get_wd_tree(cfg, tree)
    for value in jax.tree.leaves(wd):
      self.assertEqual(value, 0.)

  def test_matrix_read_view_reference_and_all_gradients(self):
    cfg, mesh, model, params, args, matrix = LocalODynamicNormScaleTest.fixture(self, run=BOTH, initial=.5)
    ref_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_common_vo_read_scale=False, bam_common_read_scale_qk=False)))
    reference = BamAttention(config=ref_cfg, num_query_heads=2, num_kv_heads=2,
        head_dim=75, bam_k=75, bam_v=32, read_side='col', layer_mode='local_qk+local_v+local_o',
        layer_inx=0, max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
        attention_kernel='dot_product_chunk', dtype=cfg.dtype)
    flat = flatten_dict(params)
    scale_path = next(k for k in flat if k[-1] == 'common_vo_read_scale')
    gamma_path = next(k for k in flat if k[-1] == 'local_o_dynamic_norm_scale')
    flat[gamma_path] = jnp.linspace(.2, .8, 64).reshape(2, 32)
    target = jax.random.normal(jax.random.key(91), (4, 4, 150))
    mtarget = jax.random.normal(jax.random.key(92), matrix.shape)

    def loss(ref, p, m, x):
      scale = flatten_dict(p)[scale_path][0]
      def intercept(next_fun, call_args, kwargs, context):
        result = next_fun(*call_args, **kwargs)
        return result * scale if context.method_name == '_matrix_for_read' else result
      with nn.intercept_methods(intercept) if ref else contextlib.nullcontext():
        y, mout = (reference if ref else model).apply(
            p, x, x, args[2], args[3], M_in=m, deterministic=True)
      return jnp.mean(y * target) + jnp.mean(mout * mtarget), (y, mout)

    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      for scale in (1., .37, -.4, 0.):
        flat[scale_path] = jnp.array([scale], jnp.float32)
        p = unflatten_dict(flat)
        (_, actual), ga = jax.value_and_grad(lambda p, m, x: loss(False, p, m, x),
            argnums=(0, 1, 2), has_aux=True)(p, matrix, args[0])
        (_, expected), ge = jax.value_and_grad(lambda p, m, x: loss(True, p, m, x),
            argnums=(0, 1, 2), has_aux=True)(p, matrix, args[0])
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
          np.testing.assert_allclose(a, b, rtol=2e-4, atol=3e-6)
        for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(ge)):
          np.testing.assert_allclose(a, b, rtol=1e-3, atol=3e-6)

  def check_model(self, run, dtype='float32', qknorm=False):
    cfg = self.tiny(run, length=128, dtype=dtype)
    cfg.get_keys().update(query_chunk_size=64, qk_norm=qknorm)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_pallas_core=True, bam_pallas_write=True,
        bam_pallas_interpret=True, bam_splash_attention=True, bam_pallas_fused_inputs=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    native, fused = Transformer(cfg, mesh, quant=None), Transformer(pcfg, mesh, quant=None)
    tok = jax.random.randint(jax.random.key(20), (1, 128), 0, 32)
    pos = (jnp.arange(128) % 64)[None]
    seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
    original = splash.make_splash_mha
    def interpret(*args, **kwargs):
      return original(*args, **dict(kwargs, interpret=True))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        splash, 'make_splash_mha', side_effect=interpret):
      params, pp = (nn.unbox(m.init(jax.random.key(21), tok, pos, seg, tok)) for m in (native, fused))
      for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(pp)):
        np.testing.assert_array_equal(a, b)
      flat = flatten_dict(params)
      for i, path in enumerate(flat):
        if path[-1] == 'local_o_dynamic_norm_scale':
          flat[path] = flat[path] * (1 + .3 * jax.random.normal(jax.random.key(70 + i), flat[path].shape))
      params = unflatten_dict(flat)
      def call(model, p):
        return model.apply(p, tok, pos, seg, tok, enable_dropout=False)[0].astype(jnp.float32)
      target = jax.random.normal(jax.random.key(23), call(native, params).shape)
      def loss(model, p):
        y = call(model, p)
        return jnp.mean(y * target), y
      (_, a), ga = jax.value_and_grad(lambda p: loss(native, p), has_aux=True)(params)
      (_, b), gb = jax.value_and_grad(lambda p: loss(fused, p), has_aux=True)(params)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves((a, b, ga, gb))))
    errors = {k: float(jnp.max(jnp.abs(v - flatten_dict(gb)[k])) / (jnp.max(jnp.abs(v)) + 1e-12))
              for k, v in flatten_dict(ga).items()}
    worst = max(errors, key=errors.get)
    print('ALL_READ_GAMMA_MODEL_GRAD', run, dtype, errors[worst], worst, flush=True)
    tolerance = 1e-3 if dtype == 'float32' else .06
    self.assertLess(errors[worst], tolerance, worst)
    np.testing.assert_allclose(a, b, rtol=2e-4 if dtype == 'float32' else .025,
                               atol=2e-6 if dtype == 'float32' else 2e-4)
    for path, value in flatten_dict(ga).items():
      if path[-1] in ('common_vo_read_scale', 'local_o_dynamic_norm_scale', 'local_q_read_scale', 'local_k_read_scale'):
        self.assertGreater(float(jnp.max(jnp.abs(value))), 0., str(path))

  def test_gamma_model_all_gradients(self): self.check_model(GAMMA)
  def test_all_read_model_all_gradients(self): self.check_model(ALL)
  def test_combined_model_all_gradients(self): self.check_model(BOTH)
  def test_combined_bf16_model_all_gradients(self): self.check_model(BOTH, 'bfloat16')
  def test_optional_full_head_qknorm_model_all_gradients(self): self.check_model(BOTH, qknorm=True)


if __name__ == '__main__':
  unittest.main()

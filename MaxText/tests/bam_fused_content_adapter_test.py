"""The write content adapter inside the v8t write kernel matches the XLA adapter path (both orders)."""
import unittest
from unittest import mock

import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
from layers.models import Transformer
import bam_h_unit_300b_test as base

FULL = 'BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFull'
PRE_RMS = 'BamPythia2p8BHUnitShortTrainFastKernelContentWriteAdapterFullPreRms'


class FusedContentAdapterTest(base.HUnit300BTest):
  flags = dict(bam_pallas_fused_content_adapter=True)

  def tiny(self, run, length=128, dtype='float32'):
    c = base.HUnit300BTest.tiny(self, run, length=length, dtype=dtype)
    c.get_keys().update(self.flags)
    return c

  def check_model(self, run, dtype='float32'):
    # Non-zero adapter deltas and non-unit Q/K scales, so the adapted map and every factor are exercised.
    original = nn.Module.init

    def init(module, *args, **kwargs):
      flat = flatten_dict(nn.unbox(original(module, *args, **kwargs)))
      for i, path in enumerate(flat):
        if path[-1] == 'delta' or ('qk_norm' in path and path[-1] == 'scale'):
          flat[path] = flat[path] + .3 * jax.random.normal(jax.random.key(90 + i), flat[path].shape)
      return unflatten_dict(flat)

    with mock.patch.object(nn.Module, 'init', init):
      base.HUnit300BTest.check_model(self, run, dtype)

  def test_control_unfused_post_rms(self):
    self.flags = dict(bam_pallas_fused_content_adapter=False)
    self.check_model(FULL)

  def test_control_unfused_pre_rms(self):
    self.flags = dict(bam_pallas_fused_content_adapter=False)
    self.check_model(PRE_RMS)

  def test_full_post_rms(self):
    self.check_model(FULL)

  def test_full_pre_rms(self):
    self.check_model(PRE_RMS)

  def test_full_post_rms_bf16(self):
    self.check_model(FULL, 'bfloat16')

  def test_full_pre_rms_bf16(self):
    self.check_model(PRE_RMS, 'bfloat16')


  def test_production_classes_enable_fusion_and_keep_parameters(self):
    def tree(exp, fused):
      cfg = self.config(exp)
      cfg.get_keys().update(max_target_length=8, max_prefill_predict_length=8, bam_pallas_fused_content_adapter=fused,
                            bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      tok = jnp.ones((1, 8), jnp.int32)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        p = jax.eval_shape(lambda k: Transformer(cfg, mesh, quant=None).init(
            k, tok, jnp.arange(8)[None], tok, tok, enable_dropout=False)['params'], jax.random.key(0))
      return {k: v.shape for k, v in flatten_dict(nn.unbox(p)).items()}
    for exp in (FULL, PRE_RMS):
      self.assertTrue(self.config(exp).bam_pallas_fused_content_adapter, exp)
      fused = tree(exp, True)
      self.assertEqual(fused, tree(exp, False), exp)
      self.assertEqual(sum(math.prod(s) for s in fused.values()), 2774984832, exp)


if __name__ == '__main__':
  unittest.main()

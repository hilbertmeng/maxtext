"""The write content adapter inside the v8t write kernel matches the XLA adapter path (both orders)."""
import unittest
from unittest import mock

import jax
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

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
    self.flags = {}
    self.check_model(FULL)

  def test_control_unfused_pre_rms(self):
    self.flags = {}
    self.check_model(PRE_RMS)

  def test_full_post_rms(self):
    self.check_model(FULL)

  def test_full_pre_rms(self):
    self.check_model(PRE_RMS)

  def test_full_post_rms_bf16(self):
    self.check_model(FULL, 'bfloat16')

  def test_full_pre_rms_bf16(self):
    self.check_model(PRE_RMS, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

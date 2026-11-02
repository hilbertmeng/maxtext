"""Full-head QKNorm inside the v8t read epilogue matches the XLA norm (H unit 300B, QK100)."""
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from flax import linen as nn

import bam_h_unit_300b_test as base

RUN = base.RUN

R768 = 'BamPythia2p8BSotaIndependentR768PallasShortTrain'


class FusedQKNormTest(base.HUnit300BTest):
  flags = dict(bam_pallas_fused_qk_norm=True)

  def tiny(self, run, length=128, dtype='float32'):
    c = base.HUnit300BTest.tiny(self, run, length=length, dtype=dtype)
    c.get_keys().update(self.flags)
    return c

  def check_model(self, run, dtype='float32'):
    # Non-unit Q/K norm scales so the fused multiplier and its gradient are exercised.
    original = nn.Module.init

    def init(module, *args, **kwargs):
      from flax.traverse_util import flatten_dict, unflatten_dict
      params = original(module, *args, **kwargs)
      flat = flatten_dict(nn.unbox(params))
      for i, path in enumerate(flat):
        if 'qk_norm' in path and path[-1] == 'scale':
          flat[path] = flat[path] + .4 * jax.random.normal(jax.random.key(90 + i), flat[path].shape)
      return unflatten_dict(flat)

    with mock.patch.object(nn.Module, 'init', init):
      base.HUnit300BTest.check_model(self, run, dtype)

  def test_native_pallas_logits_and_every_gradient(self):
    self.check_model(RUN)

  def test_mxu_statistics(self):
    self.flags = dict(bam_pallas_fused_qk_norm=True, bam_pallas_fused_qk_norm_mxu=True)
    self.check_model(RUN)

  def test_mxu_statistics_padded_qk(self):
    self.flags = dict(bam_pallas_fused_qk_norm=True, bam_pallas_fused_qk_norm_mxu=True, bam_pallas_qk_out_pad=True)
    self.check_model(RUN)

  def test_mxu_statistics_bf16(self):
    self.flags = dict(bam_pallas_fused_qk_norm=True, bam_pallas_fused_qk_norm_mxu=True)
    self.check_model(RUN, 'bfloat16')

  def test_bf16_gradients_against_f32_truth(self):
    self.check_model(RUN, 'bfloat16')

  def test_partial_nope_r768(self):
    self.check_model(R768)

  def test_packed_keys(self):
    self.flags = dict(bam_pallas_packed_read_keys=True)
    self.check_model(RUN)

  def test_packed_keys_shared_vo(self):
    # Shared VO key (old-architecture read): three [B,C,3N,T] slabs through the packed einsum path.
    self.flags = dict(bam_pallas_packed_read_keys=True, bam_local_vo_independent_keys=False)
    self.check_model(RUN)

  def test_packed_keys_fused_norm(self):
    self.flags = dict(bam_pallas_packed_read_keys=True, bam_pallas_fused_qk_norm=True)
    self.check_model(RUN)

  def test_packed_keys_fused_norm_bf16(self):
    self.flags = dict(bam_pallas_packed_read_keys=True, bam_pallas_fused_qk_norm=True)
    self.check_model(RUN, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

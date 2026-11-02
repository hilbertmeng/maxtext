"""Packed read keys, fused full-head QKNorm and Splash residuals match the H unit 300B reference."""
import math
import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from unittest import mock

import max_utils
from layers.models import Transformer
import bam_h_unit_300b_test as base

FAST = 'BamPythia2p8BHUnit300BFastKernel'


class FastKernelTest(base.HUnit300BTest):
  flags = dict(bam_pallas_packed_read_keys=True, bam_pallas_fused_qk_norm=True)

  def tiny(self, run, length=128, dtype='float32'):
    c = base.HUnit300BTest.tiny(self, run, length=length, dtype=dtype)
    c.get_keys().update(self.flags)
    return c

  def check_model(self, run, dtype='float32'):
    # Non-unit Q/K norm scales so the fused multipliers and their gradients are exercised.
    original = nn.Module.init

    def init(module, *args, **kwargs):
      flat = flatten_dict(nn.unbox(original(module, *args, **kwargs)))
      for i, path in enumerate(flat):
        if 'qk_norm' in path and path[-1] == 'scale':
          flat[path] = flat[path] + .4 * jax.random.normal(jax.random.key(90 + i), flat[path].shape)
      return unflatten_dict(flat)

    with mock.patch.object(nn.Module, 'init', init):
      base.HUnit300BTest.check_model(self, run, dtype)

  def test_native_pallas_logits_and_every_gradient(self):
    self.check_model(base.RUN)

  def test_bf16_gradients_against_f32_truth(self):
    self.check_model(base.RUN, 'bfloat16')

  def test_shared_vo_packed_keys(self):
    # Shared VO key (old-architecture read): [B,C,3N,T] through the same packed path.
    self.flags = dict(bam_pallas_packed_read_keys=True, bam_local_vo_independent_keys=False,
                      bam_common_vo_read_scale=False)
    self.check_model(base.RUN)

  def test_production_config_keeps_the_parameter_tree(self):
    def tree(exp):
      cfg = self.config(exp)
      cfg.get_keys().update(max_target_length=8, max_prefill_predict_length=8,
                            bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      tok = jnp.ones((1, 8), jnp.int32)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        p = jax.eval_shape(lambda k: Transformer(cfg, mesh, quant=None).init(
            k, tok, jnp.arange(8)[None], tok, tok, enable_dropout=False)['params'], jax.random.key(0))
      return {k: v.shape for k, v in flatten_dict(nn.unbox(p)).items()}
    fast, ref = tree(FAST), tree(base.RUN)
    self.assertEqual(fast, ref)
    self.assertEqual(sum(math.prod(s) for s in fast.values()), 2775190912)
    cfg = self.config(FAST)
    self.assertEqual((cfg.remat_policy, cfg.bam_splash_residual_name, cfg.bam_pallas_vmem_mib),
                     ('minimal_save_splash', 'bam_splash_residual', 63))


if __name__ == '__main__':
  unittest.main()

"""Full-head pre-RoPE QKNorm must preserve XLA/fused BAM forward and gradients."""
import unittest
from unittest import mock

import bam_pallas_model_test as model_test


class PallasQKNormTest(model_test.ModelEquivalenceTest):
  # Select only these tests; the inherited tests remain in their original file.
  test_logits_and_gradients = None
  test_full_m_general_read = None
  test_v8t_token_major = None

  def compare(self, full, exps, shift=False):
    original_build = model_test.build

    def build(exp, out):
      cfg, _ = original_build(exp, out)
      cfg.get_keys().update(qk_norm=True, bam_qk_norm_after_concat=full)
      if shift:
        cfg.get_keys().update(bam_m_shift_mode='read', bam_m_shift_init=.5)
      return cfg, model_test.Transformer(cfg, model_test.jax.sharding.Mesh(
          model_test.max_utils.create_device_mesh(cfg), cfg.mesh_axes), quant=None)

    with mock.patch.object(model_test, 'build', build):
      self._run('BamPallasFullMTinyXlaTest', exps)

  def test_full_head_v7(self):
    self.compare(True, ('BamPallasFullMTinyPallasV7FusedInputsTest',))

  def test_full_head_v8t_mshift(self):
    self.compare(True, ('BamPallasFullMTinyPallasV8TTest',), shift=True)

  def test_standard_arm_compatibility_v8t(self):
    self.compare(False, ('BamPallasFullMTinyPallasV8TTest',))


if __name__ == '__main__':
  unittest.main()

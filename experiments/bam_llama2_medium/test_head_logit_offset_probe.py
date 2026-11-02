"""Check the FP32 shift-invariant control and the BF16-rounding intervention."""
from pathlib import Path
import sys
import unittest

import jax.numpy as jnp
import numpy as np
import max_utils

sys.path.insert(0,str(Path(__file__).parent))
import head_logit_offset_probe as probe


class LogitOffsetTest(unittest.TestCase):
  def test_common_offset_changes_bf16_logits_but_not_fp32_probabilities(self):
    h=jnp.array([[[1.,1.]]])
    # The first input feature produces an uninformative common +256 offset.
    w=jnp.array([[256.,256.,256.],[.1,.2,.3]])
    result=probe.offset_metrics(h,w,jnp.array([[2]]),max_utils)
    losses=result['losses']
    np.testing.assert_allclose(losses['accumulator_fp32'],losses['center_fp32_control'],atol=2e-5,rtol=0)
    self.assertGreater(abs(float(losses['bf16_logits'][0,0]-losses['accumulator_fp32'][0,0])),.01)
    self.assertLess(abs(float(losses['center_then_bf16'][0,0]-losses['accumulator_fp32'][0,0])),.001)


if __name__=='__main__':
  unittest.main()

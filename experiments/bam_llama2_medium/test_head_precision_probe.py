"""Check precision replay against the real training CE forward."""
import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import max_utils

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('head_precision_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class HeadPrecisionTest(unittest.TestCase):
  def setUp(self):
    self.h = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 6))
    self.w = jax.random.normal(jax.random.PRNGKey(2), (6, 11))
    self.y = jnp.arange(14).reshape(2, 7) % 11

  def test_fp32_matches_log_softmax_reference(self):
    actual = probe.head_losses(self.h, self.w, self.y, jnp.float32, jnp.float32,
                               max_utils._cross_entropy_with_logits_fwd)
    expected = -jnp.take_along_axis(jax.nn.log_softmax(self.h@self.w), self.y[..., None], -1).mean((1, 2))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

  def test_bf16_matches_autodiff_forward(self):
    def training(w):
      logits = self.h.astype(jnp.bfloat16)@w.astype(jnp.bfloat16)
      return max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(self.y, 11), 0.)[0].mean()
    expected, _ = jax.value_and_grad(training)(self.w)
    actual = probe.head_losses(self.h, self.w, self.y, jnp.bfloat16, jnp.bfloat16,
                               max_utils._cross_entropy_with_logits_fwd).mean()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


if __name__ == '__main__':
  unittest.main()

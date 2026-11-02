"""Check the conditional head-gradient formula and amplitude factorization."""
import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('head_gradient_amplitude_probe.py'))
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class HeadGradientTest(unittest.TestCase):
  def test_closed_form_matches_autodiff(self):
    h=jax.random.normal(jax.random.PRNGKey(1),(2,7,6))
    w=jax.random.normal(jax.random.PRNGKey(2),(6,11)).astype(jnp.bfloat16).astype(jnp.float32)
    y=jnp.arange(14).reshape(2,7)%11
    def loss(weights):
      logits=jnp.matmul(h,weights,precision=jax.lax.Precision.HIGHEST)
      return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1).mean()
    actual,stats=probe.head_gradient(h,w,y)
    np.testing.assert_allclose(actual,jax.grad(loss)(w),rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(stats[0],loss(w),rtol=2e-5,atol=2e-6)

  def test_frozen_error_amplitude_identity(self):
    rng=np.random.default_rng(7)
    h=rng.normal(size=(14,6));error=rng.normal(size=(14,11))
    def factors(features):
      g=features.T@error/len(features)
      return probe.factors(np.linalg.norm(g),np.sqrt(np.mean(features**2)),
                           np.sqrt(np.mean(np.sum(error**2,axis=-1))))
    a,b=factors(h),factors(1.7*h)
    self.assertAlmostEqual(b['gradient_norm']/a['gradient_norm'],1.7)
    self.assertAlmostEqual(b['normalized_geometry_factor'],a['normalized_geometry_factor'])


if __name__=='__main__':
  unittest.main()

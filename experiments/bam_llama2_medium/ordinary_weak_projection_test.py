import unittest
import numpy as np
import jax,jax.numpy as jnp
from ordinary_weak_projection import calibrate,patch_write
class OrdinaryWeakTest(unittest.TestCase):
  def test_nested_energy_bases_and_held_out_shift(self):
    a=np.diag([96.,3.,1.]);v=np.diag([10.,10.,80.]);p,m=calibrate(a,v)
    np.testing.assert_allclose(p[0],np.diag([0,0,1]),atol=1e-7);np.testing.assert_allclose(p[1],np.diag([0,1,1]),atol=1e-7)
    self.assertAlmostEqual(m['layers'][0]['thresholds'][0]['weak_calibration_fraction'],.01);self.assertAlmostEqual(m['layers'][0]['thresholds'][0]['weak_held_out_fraction'],.8)
  def test_zero_energy_is_undefined_and_invalid_covariance_rejected(self):
    p,m=calibrate(np.zeros((3,3)),np.zeros((3,3)));self.assertIsNone(m['layers'][0]['thresholds'][0]['weak_calibration_fraction']);self.assertIsNone(m['layers'][0]['thresholds'][0]['weak_held_out_fraction'])
    with self.assertRaises(AssertionError):calibrate(np.diag([1.,-.1]),np.eye(2))
  def test_weak_restore_holds_frozen_strong_and_controls_match_size(self):
    x=jnp.array([[1.,2.,3.,4.],[2.,3.,5.,7.]]);old=-x;p=jnp.diag(jnp.array([0.,0.,1.,1.]));f=jax.jit(patch_write)
    y=f(x,old,p,2);np.testing.assert_allclose(y@(jnp.eye(4)-p),x@(jnp.eye(4)-p));np.testing.assert_allclose(y@p,old@p)
    ref=np.linalg.norm(f(x,old,p,1)-x,axis=-1)
    for mode in (3,4):np.testing.assert_allclose(np.linalg.norm(f(x,old,p,mode)-x,axis=-1),ref,rtol=2e-6)
  def test_bf16_identity_and_zero_are_finite(self):
    x=jnp.asarray([[1.,2.,3.,4.]],jnp.bfloat16);p=jnp.eye(4)
    np.testing.assert_array_equal(patch_write(x,x,p,0),x)
    for mode in range(6):self.assertTrue(np.isfinite(patch_write(jnp.zeros_like(x),x,p,mode).astype(jnp.float32)).all())
if __name__=='__main__':unittest.main()

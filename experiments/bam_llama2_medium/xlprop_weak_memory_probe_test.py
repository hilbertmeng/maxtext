import unittest
import numpy as np
import jax
import jax.numpy as jnp
from xlprop_weak_memory_probe import weak_projectors,correction

class WeakTest(unittest.TestCase):
  def setUp(self):
    gen=np.random.default_rng(12);self.m=gen.normal(size=(2,40,96));self.p,self.rank=weak_projectors(self.m);self.a=jnp.asarray(gen.normal(size=(2,40,96)),jnp.float32);self.o=jnp.asarray(gen.normal(size=(2,40,96)),jnp.float32)
  def test_basis_orthogonal_and_energy_threshold(self):
    np.testing.assert_allclose(self.p,np.swapaxes(self.p,-1,-2),atol=1e-7);np.testing.assert_allclose(self.p.astype(float)@self.p.astype(float),self.p,atol=2e-7)
    frac=np.sum((self.m@self.p)**2,axis=(-2,-1))/np.sum(self.m**2,axis=(-2,-1));self.assertTrue(np.all(frac<=.010001))
    p,r=weak_projectors(np.zeros((40,96)));np.testing.assert_array_equal(p,np.eye(96));self.assertEqual(int(r),0)
  def test_weak_removal_restore_keep_strong_component(self):
    f=jax.jit(correction);p=jnp.asarray(self.p);strong=jnp.eye(96)-p
    removed=self.a+f(self.a,self.a,self.o,self.o,p,1);restored=self.a+f(self.a,self.a,self.o,self.o,p,2)
    np.testing.assert_allclose(removed@p,0,atol=2e-6)
    np.testing.assert_allclose(restored@p,self.o@p,atol=3e-6)
    np.testing.assert_allclose(removed@strong,self.a@strong,atol=2e-6);np.testing.assert_allclose(restored@strong,self.a@strong,atol=3e-6)
  def test_controls_match_weak_correction_norm(self):
    f=jax.jit(correction);norm=lambda x:np.linalg.norm(x,axis=(-2,-1));ref=norm(f(self.a,self.a,self.o,self.o,self.p,1))
    for mode in (3,4):np.testing.assert_allclose(norm(f(self.a,self.a,self.o,self.o,self.p,mode)),ref,rtol=2e-6)
  def test_identity_zero_and_total_weak_are_finite(self):
    f=jax.jit(correction);z=jnp.zeros_like(self.a)
    for mode in range(9):self.assertTrue(np.isfinite(f(z,z,z,z,self.p,mode)).all())
    np.testing.assert_array_equal(f(self.a,self.a,self.o,self.o,self.p,0),z)
    np.testing.assert_allclose(self.a+f(self.a,self.a,self.o,self.o,self.p,8),self.a+(self.o-self.a)@self.p,atol=1e-6)

if __name__=='__main__':unittest.main()

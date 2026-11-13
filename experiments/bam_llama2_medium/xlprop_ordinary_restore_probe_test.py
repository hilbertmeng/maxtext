import runpy,unittest
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
P=runpy.run_path(str(Path(__file__).with_name('xlprop_ordinary_restore_probe.py')))
class ProbeTest(unittest.TestCase):
 def test_masked_global_second_moment(self):
  x=np.arange(2*8*4,dtype=np.float32).reshape(2,8,4);valid=np.ones((2,8),bool);valid[1,3]=False
  positions=np.array([1,3,5]);a=x[:,positions][valid[:,positions]]
  got=jax.jit(P['residual_moment'])(jnp.array(x),jnp.array(valid),jnp.array(positions))
  np.testing.assert_allclose(got,a.T@a/len(a),rtol=1e-6)
  np.testing.assert_array_equal(P['residual_moment'](jnp.array(x),jnp.zeros_like(jnp.array(valid)),jnp.array(positions)),np.zeros((4,4)))
 def test_threshold_dispatch_and_frozen_identity(self):
  peak=jnp.array([[1.,2.,3.,4.]],jnp.bfloat16);old=jnp.array([[4.,3.,2.,1.]],jnp.bfloat16)
  bases=jnp.array([np.diag([0.,0.,1.,1.]),np.diag([0.,1.,1.,1.])],jnp.float32)
  patch=jax.jit(P['patch_write'])
  np.testing.assert_array_equal(patch(peak,old,bases,jnp.int32(0)),peak)
  np.testing.assert_array_equal(patch(peak,old,bases,jnp.int32(9)),old)
  for mode,expected in [(1,[[1,2,0,0]]),(2,[[1,2,2,1]]),(5,[[1,0,0,0]]),(6,[[1,3,2,1]])]:
   np.testing.assert_array_equal(patch(peak,old,bases,jnp.int32(mode)),expected)
 def test_restore_controls_match_and_stay_in_subspaces(self):
  rng=np.random.default_rng(42);q=np.linalg.qr(rng.normal(size=(8,8)))[0];basis=q[:,:4]@q[:,:4].T
  bases=jnp.array([basis,basis]);peak=jnp.array(rng.normal(size=(5,8)),jnp.float32);old=jnp.array(rng.normal(size=(5,8)),jnp.float32)
  delta=lambda mode:np.asarray(P['patch_write'](peak,old,bases,jnp.int32(mode))-peak)
  weak=delta(2);strong=delta(3);random=delta(4)
  for x in (strong,random):np.testing.assert_allclose(np.linalg.norm(x,axis=-1),np.linalg.norm(weak,axis=-1),rtol=2e-6)
  np.testing.assert_allclose(strong@basis,0,atol=1e-6)
  np.testing.assert_allclose(random@(np.eye(8)-basis),0,atol=1e-6)
  self.assertGreater(np.linalg.norm(random-weak),.1)
if __name__=='__main__':unittest.main()

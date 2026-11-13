import unittest
import jax
import jax.numpy as jnp
import numpy as np
from xlprop_write_direction_probe import patch_write


class DirectionTest(unittest.TestCase):
  def test_historical_mlp_hook_contract(self):
    import inspect
    from xlprop_pattern_write_probe import runtime_modules
    runtime_modules()
    from layers import linears
    self.assertIn('inputs',inspect.signature(linears.MlpBlock.__call__).parameters)

  def test_direction_and_amplitude_are_separated(self):
    c=jnp.array([[3.,4.,0.]]);old=jnp.array([[0.,0.,2.]])
    f=jax.jit(patch_write)
    np.testing.assert_array_equal(f(c,old,0),c)
    np.testing.assert_allclose(f(c,old,1),[[0.,0.,5.]],atol=1e-6)
    np.testing.assert_allclose(f(c,old,2),[[1.2,1.6,0.]],atol=1e-6)
    np.testing.assert_array_equal(f(c,old,3),old)
    np.testing.assert_allclose(jnp.linalg.norm(f(c,old,4),axis=-1),[5.],atol=1e-6)

  def test_bfloat16_identity_and_finite_zero_fallback(self):
    x=jnp.array([[1.,2.,3.]],jnp.bfloat16)
    for mode in range(5):
      value=jax.jit(patch_write)(x,jnp.zeros_like(x),mode)
      self.assertEqual(value.dtype,x.dtype);self.assertTrue(np.isfinite(value.astype(jnp.float32)).all())
    np.testing.assert_array_equal(patch_write(x,x,0),x)


if __name__=='__main__':unittest.main()

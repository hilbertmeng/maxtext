"""Target the BF16 cancellation mechanism and actual custom backward."""
import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import max_utils

spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('ce_gradient_precision_probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class CEPrecisionTest(unittest.TestCase):
  def test_accumulator_output_has_same_backward_as_native_for_same_cotangent(self):
    h=jax.random.normal(jax.random.PRNGKey(3),(2,7,6)).astype(jnp.bfloat16)
    w=jax.random.normal(jax.random.PRNGKey(4),(6,11)).astype(jnp.bfloat16)
    cotangent=jax.random.normal(jax.random.PRNGKey(5),(2,7,11)).astype(jnp.bfloat16)
    native,native_pull=jax.vjp(lambda h,w:jnp.matmul(h,w,precision=jax.lax.Precision.HIGHEST),h,w)
    output,pull=jax.vjp(probe.accumulator_dot_bf16_backward,h,w)
    self.assertEqual(output.dtype,jnp.float32)
    for a,b in zip(native_pull(cotangent),pull(cotangent.astype(jnp.float32))):
      np.testing.assert_array_equal(a,b)

  def test_confident_target_can_zero_while_tail_survives(self):
    z=jnp.array([[0.,-8.]],dtype=jnp.bfloat16); y=jnp.array([0])
    native,reference=probe.logit_derivatives(z,y,max_utils)
    self.assertEqual(float(native[0,0]),0.)
    self.assertGreater(float(native[0,1]),0.)
    self.assertLess(float(reference[0,0]),-.0003)
    self.assertLess(abs(float(reference.sum())),1e-7)

  def test_explicit_derivative_matches_custom_vjp(self):
    z=jnp.array([[2.,-1.,0.], [0.,-8.,-9.]],dtype=jnp.bfloat16); y=jnp.array([1,0])
    for index,dtype in enumerate((jnp.bfloat16,jnp.float32)):
      actual=jax.grad(lambda logits:max_utils.cross_entropy_with_logits(
          logits,jax.nn.one_hot(y,3),0.)[0].sum())(z.astype(dtype))
      expected=probe.logit_derivatives(z,y,max_utils)[index]
      np.testing.assert_array_equal(actual.astype(jnp.float32),expected)

  def test_ce_only_loss_gradient_has_expected_dtype_boundary(self):
    h=jax.random.normal(jax.random.PRNGKey(1),(2,7,6))
    w=jax.random.normal(jax.random.PRNGKey(2),(6,11))
    y=jnp.arange(14).reshape(2,7)%11
    for mode in probe.MODES:
      value,grad=jax.value_and_grad(lambda weights:probe.loss(h,weights,y,mode,max_utils))(w)
      self.assertTrue(np.isfinite(value))
      self.assertEqual(grad.shape,w.shape)
      # Restored parameters are FP32; BF16 cast VJP returns FP32 master gradient.
      self.assertEqual(grad.dtype,jnp.float32)
      np.testing.assert_array_equal(grad,grad.astype(jnp.bfloat16).astype(jnp.float32))


if __name__ == '__main__':
  unittest.main()

"""Independent autodiff controls for the frozen-boundary analytic reference."""
import os
import unittest

if hasattr(os, 'sched_getaffinity'):
  os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])

import jax
import jax.numpy as jnp
import numpy as np
from swiglu_local_jacobian import chain_rule


class ChainRuleTest(unittest.TestCase):
  def test_forward_and_both_derivative_paths(self):
    keys = jax.random.split(jax.random.PRNGKey(83), 5)
    x, dy, wg, wu, wo = [jax.random.normal(k, shape) for k, shape in zip(
        keys, [(3, 8), (3, 6), (8, 12), (8, 12), (12, 6)])]
    x = x.at[0].set(0).at[1].multiply(10)
    for gate, up in [(0., 0.), (4., 7.), (4., 25.)]:
      def forward(z, stop_gate=False, stop_up=False):
        u, v = z @ wg, z @ wu
        a = (gate * jnp.tanh(u / gate) if gate else u) * jax.nn.sigmoid(u)
        b = up * jnp.tanh(v / up) if up else v
        if stop_gate:
          a = jax.lax.stop_gradient(a)
        if stop_up:
          b = jax.lax.stop_gradient(b)
        return (a * b) @ wo

      result = chain_rule(x, dy, wg, wu, wo, gate, up)
      reference = jax.grad(lambda z: jnp.sum(forward(z) * dy))(x)
      np.testing.assert_allclose(result['y'], forward(x), rtol=2e-5, atol=2e-5)
      np.testing.assert_allclose(result['dx'], reference, rtol=3e-5, atol=5e-5)
      for name, stop_gate, stop_up in [('dx_gate', False, True), ('dx_value', True, False)]:
        reference = jax.grad(lambda z: jnp.sum(forward(z, stop_gate, stop_up) * dy))(x)
        np.testing.assert_allclose(result[name], reference, rtol=3e-5, atol=5e-5)
      np.testing.assert_array_equal(result['u'], x @ wg)
      np.testing.assert_array_equal(result['v'], x @ wu)


if __name__ == '__main__':
  unittest.main()

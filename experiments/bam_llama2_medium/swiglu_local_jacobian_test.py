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
  def test_identity_tap_has_identity_primal_and_vjp(self):
    from xlprop_identity_taps import make_identity_tap
    calls = []
    def record(direction, layer, site, value):
      calls.append((direction, int(layer), site, np.asarray(value).copy()))
    tap = make_identity_tap(record)
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    plain = jax.jit(jax.value_and_grad(lambda z: jnp.sum(jnp.sin(z))))(x)
    changed = jax.jit(jax.value_and_grad(
        lambda z: jnp.sum(jnp.sin(tap(z, jnp.int32(6), 'input')))))(x)
    jax.effects_barrier()
    for a, b in zip(plain, changed):
      np.testing.assert_array_equal(a, b)
    self.assertEqual({c[0] for c in calls}, {'forward', 'backward'})
    for direction, layer, site, value in calls:
      self.assertEqual((layer, site), (6, 'input'))
      np.testing.assert_array_equal(value, x if direction == 'forward' else jnp.cos(x))
    def scanned(z, capture):
      def block(value, layer):
        if capture:
          value = tap(value, layer, 'scan')
        return jnp.tanh(value), None
      return jnp.sum(jax.lax.scan(jax.checkpoint(block), z, jnp.arange(3))[0])
    plain = jax.jit(jax.value_and_grad(lambda z: scanned(z, False)))(x)
    changed = jax.jit(jax.value_and_grad(lambda z: scanned(z, True)))(x)
    jax.effects_barrier()
    for a, b in zip(plain, changed):
      np.testing.assert_array_equal(a, b)

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

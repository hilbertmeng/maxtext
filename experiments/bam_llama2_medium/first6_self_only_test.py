"""CPU checks for the diagnostic's endpoint and layer-selection semantics."""
import unittest
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers.attentions import _bam_fetch_op


class Cell(nn.Module):
  @nn.compact
  def __call__(self, carry, xs):
    return carry, self.get_variable('causal_ablation', 'cross_scale')


class Scanned(nn.Module):
  @nn.compact
  def __call__(self):
    return nn.scan(Cell, variable_axes={'causal_ablation': 0},
                   split_rngs={'params': False}, length=24)(name='layers')(
                       jnp.array(0.), None)[1]


class FirstSixTest(unittest.TestCase):
  def test_endpoints_and_half(self):
    alpha = jnp.asarray([[[[1,0,0],[.25,.75,0],[.25,.25,.5]]]])
    matrix = jnp.arange(12, dtype=jnp.float32).reshape(1,3,2,2)
    weight = jnp.ones((1,3,1))
    diagonal = jnp.eye(3,dtype=bool)
    def run(scale):
      return _bam_fetch_op(alpha,matrix,weight,diagonal,
                           diagonal_one=True,cross_scale=scale)
    original = run(None)
    np.testing.assert_array_equal(original,run(jnp.array(1.)))
    np.testing.assert_array_equal(matrix,run(jnp.array(0.)))
    np.testing.assert_allclose((original+matrix)/2,run(jnp.array(.5)))

  def test_scan_layer_selection(self):
    scales = jnp.array([0.]*6+[1.]*18)
    actual = Scanned().apply({'causal_ablation':{'layers':{'cross_scale':scales}}})
    np.testing.assert_array_equal(actual,scales)


if __name__ == '__main__':
  unittest.main()

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from xlprop_sink_probe import parameter_plans, patch_value

class SinkTest(unittest.TestCase):
  def test_native_and_other_layer_exact_and_gradients(self):
    v=jnp.arange(24.,dtype=jnp.float32).reshape(2,3,2,2)
    donor=jnp.full((2,2,2),-3.)
    for layer,target,mode in [(3,3,0),(0,3,1),(12,3,2)]:
      f=lambda x:patch_value(x,donor,layer,target,mode,0)
      np.testing.assert_array_equal(f(v),v)
      np.testing.assert_array_equal(jax.grad(lambda x:f(x).sum())(v),jnp.ones_like(v))
  def test_clamp_scope_and_frozen_identity_gradient_control(self):
    v=jnp.arange(24.,dtype=jnp.float32).reshape(2,3,2,2)
    donor=jnp.full((2,2,2),-3.)
    for mode in (1,2,3):
      f=lambda x:patch_value(x,donor,3,3,mode,1)
      y=f(v); np.testing.assert_array_equal(y[:,0],v[:,0]);np.testing.assert_array_equal(y[:,2],v[:,2])
      np.testing.assert_array_equal(y[:,1],jnp.zeros_like(donor) if mode==1 else donor if mode==2 else v[:,1])
      g=jax.grad(lambda x:f(x).sum())(v)
      np.testing.assert_array_equal(g[:,1],jnp.zeros_like(donor))
      np.testing.assert_array_equal(g[:,0],jnp.ones_like(donor))
  def test_plan_has_controls_and_three_comparison_layers(self):
    p=parameter_plans();self.assertEqual(len(p),23)
    self.assertEqual(p[0]['name'],'peak_control');self.assertEqual(p[1]['name'],'old_control')
    self.assertEqual({x['layer'] for x in p if 'layer' in x},{0,3,12})
    self.assertEqual(len({x['name'] for x in p}),len(p))
if __name__=='__main__':unittest.main()

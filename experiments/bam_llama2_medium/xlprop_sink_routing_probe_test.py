import unittest
import jax
import jax.numpy as jnp
import numpy as np
from xlprop_sink_routing_probe import patch_value, parameter_plans
from xlprop_routing_localization_probe import selected_parameter

class RoutingTest(unittest.TestCase):
 def test_joint_scope_and_control_grad(self):
  v=jnp.arange(2*140*2*2.,dtype=jnp.float32).reshape(2,140,2,2)
  donor=jnp.zeros((2,2,2))
  for layer in (0,3,5,6,27):
   for mode in (0,1,3,4,5):
    f=lambda x:patch_value(x,donor,layer,-63,mode,0)
    y=f(v);g=jax.grad(lambda x:f(x).sum())(v)
    width=8 if mode>=4 else 1
    expected=np.asarray(v).copy();grad=np.ones(v.shape)
    if layer<6 and mode:
     if mode in (1,4):expected[:,:width]=0
     grad[:,:width]=0
    np.testing.assert_array_equal(y,expected);np.testing.assert_array_equal(g,grad)
 def test_single_donor_and_interior(self):
  v=jnp.ones((2,140,2,2));d=jnp.full((2,2,2),3.)
  np.testing.assert_array_equal(patch_value(v,d,3,3,2,0)[:,0],d)
  y=patch_value(v,d,27,-((1<<28)-1),4,128)
  self.assertEqual(float(y[:,128:136].sum()),0)
  np.testing.assert_array_equal(y[:,:128],v[:,:128]);np.testing.assert_array_equal(y[:,136:],v[:,136:])
 def test_group_parameter_selection(self):
  family=lambda _: 'qk'
  self.assertEqual(selected_parameter(('layers',),(2,27,3),family,{'qk'},list(range(6))),(True,1))
  self.assertEqual(selected_parameter(('final_local_layer',),(2,3),family,{'qk'},list(range(6))),(False,None))
  self.assertEqual(len(parameter_plans()),27)
  old=jnp.zeros((2,27,3));peak=jnp.ones((2,27,3));out=peak.at[:,jnp.asarray(list(range(6))),:].set(old[:,jnp.asarray(list(range(6))),:])
  self.assertEqual(float(out[:,:6].sum()),0);self.assertEqual(float(out[:,6:].sum()),126.)
if __name__=='__main__':unittest.main()

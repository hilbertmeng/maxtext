import unittest,runpy
from pathlib import Path
import jax,jax.numpy as j
import numpy as np
m=runpy.run_path(str(Path(__file__).with_name('xlprop_attention_vjp_paths.py')))
class Tests(unittest.TestCase):
 def test_forward_identity_and_disjoint_tangents(self):
  x=j.array([-2.,.25,4.])
  for mode in range(5):
   for slot in range(4):
    np.testing.assert_array_equal(m['route_tangent'](x,j.int32(mode),slot),x)
    g=jax.grad(lambda y:j.sum(m['route_tangent'](y,j.int32(mode),slot)))(x)
    enabled=mode==0 or mode==1 and slot<2 or mode==2 and slot==2 or mode==3 and slot==3
    np.testing.assert_array_equal(g,j.ones_like(x) if enabled else j.zeros_like(x))
 def test_direct_path_not_repeated_in_branch_sum(self):
  def objective(x,mode):
   return x*x+sum((slot+1)*m['route_tangent'](x,j.int32(mode),slot) for slot in range(4))
  grad=[jax.grad(objective)(j.float32(3),i) for i in range(5)]
  self.assertEqual(float(sum(g-grad[4] for g in grad[1:4])+grad[4]),float(grad[0]))
  self.assertNotEqual(float(sum(grad[1:4])),float(grad[0]))
if __name__=='__main__':unittest.main()

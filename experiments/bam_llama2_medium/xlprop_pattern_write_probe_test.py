import unittest
import jax
import jax.numpy as jnp
import numpy as np
from xlprop_pattern_write_probe import routing_probabilities,write_reference,parameter_plans

class PatternWriteTest(unittest.TestCase):
 def test_strict_causal_segment_and_empty_rows(self):
  k=jnp.ones((1,4,1,2));q=k[:,jnp.array([0,1,3])];seg=jnp.array([[1,1,2,2]])
  p=jax.jit(lambda q,k:routing_probabilities(q,k,seg,(0,1,3)))(q,k)
  np.testing.assert_array_equal(p[0,0,0],0.)
  np.testing.assert_array_equal(p[0,0,1],jnp.array([1.,0,0,0]))
  np.testing.assert_array_equal(p[0,0,2],jnp.array([0.,0,1.,0]))
 def test_route_content_decomposition(self):
  a=jnp.array([.9,.1]);b=jnp.array([.4,.6]);u=jnp.array([[1.,0],[0,2.]]);v=jnp.array([[2.,1],[1.,3.]])
  np.testing.assert_allclose(b@v-a@u,(b-a)@v+a@(v-u),rtol=1e-6)
 def test_write_outer_axes_norm_and_rounding(self):
  content=jnp.array([[[[3.,4.],[1.,2.]]]],dtype=jnp.bfloat16);addr=jnp.array([[[[1.,2.,3.],[3.,2.,1.]]]],dtype=jnp.bfloat16);logit=jnp.zeros((1,1,2),jnp.bfloat16)
  for divisor in (None,2.):
   y=jax.jit(lambda c,a:write_reference(c,logit,a,1e-6,divisor))(content,addr)
   c=content.astype(jnp.float32);a=addr.astype(jnp.float32)
   c=c/jnp.sqrt(jnp.mean(c*c,-1,keepdims=True)+1e-6) if divisor is None else c/divisor
   a=a/jnp.sqrt(jnp.mean(a*a,-1,keepdims=True)+1e-6)
   expected=np.asarray(a.astype(jnp.bfloat16).astype(jnp.float32))[0,0].T@np.asarray((.5*c).astype(jnp.bfloat16).astype(jnp.float32))[0,0]
   self.assertEqual(y.shape,(1,1,3,2));np.testing.assert_allclose(y[0,0],expected,rtol=1e-6)
 def test_plan_scope(self):
  plans=parameter_plans();self.assertEqual(len(plans),7)
  for p in plans[2:]:self.assertEqual(p['layer'],list(range(6)))
if __name__=='__main__':unittest.main()

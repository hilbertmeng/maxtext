import unittest
import jax,jax.numpy as jnp,numpy as np
from xlprop_sink_route_probe import route_counterfactuals

def routing(q,k,seg,queries):
 scores=jnp.einsum('bqhd,bshd->bhqs',q,k)
 mask=jnp.arange(k.shape[1])[None,:]<=jnp.array(queries)[:,None]
 mask=mask[None,None]&(seg[:,None,None,:]!=0)&(seg[:,None,None,:]==seg[:,jnp.array(queries)][:,None,:,None])
 return jax.nn.softmax(jnp.where(mask,scores,-1e30),-1)
class RoutingTest(unittest.TestCase):
 def test_control_and_no_input_mutation(self):
  q=jax.random.normal(jax.random.key(1),(2,3,4,8));k=jax.random.normal(jax.random.key(2),(2,32,4,8));seg=jnp.ones((2,32),int)
  old=np.array(k);cf=route_counterfactuals(q,k,seg,[0,16,31],routing,6)
  np.testing.assert_array_equal(cf['full'],routing(q,k,seg,[0,16,31]));np.testing.assert_array_equal(k,old)
  for v in cf.values():np.testing.assert_allclose(v.sum(-1),1,atol=2e-7);np.testing.assert_array_equal(v[:,:,0,1:],0)
 def test_rope_removal_eliminates_constructed_first_key_sink(self):
  q=jnp.zeros((1,1,1,8)).at[...,6:].set(1);k=jnp.zeros((1,32,1,8)).at[:,0,:,6:].set(5)
  cf=route_counterfactuals(q,k,jnp.ones((1,32),int),[31],routing,6)
  self.assertGreater(float(cf['full'][0,0,0,0]),.99)
  for n in ['nope_only','first_rope_key_zero','first_rope_key_normmatched_token16']:
   self.assertAlmostEqual(float(cf[n][0,0,0,0]),1/32,places=6)
 def test_production_widths_abstract_trace(self):
  r=jax.eval_shape(lambda q,k:route_counterfactuals(q,k,jnp.ones((4,4096),int),[0,1,2,8,16,128,512,1211,2048,3072,4095],routing,96),jax.ShapeDtypeStruct((4,11,20,120),jnp.bfloat16),jax.ShapeDtypeStruct((4,4096,20,120),jnp.bfloat16))
  self.assertEqual(r['full'].shape,(4,20,11,4096))
if __name__=='__main__':unittest.main()

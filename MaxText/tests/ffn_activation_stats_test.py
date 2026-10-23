import unittest
import numpy as np
import jax.numpy as jnp
from ffn_activation_capture import statistics,balance_statistics
class StatisticsTest(unittest.TestCase):
 def test_signed_counts_include_zero_as_inactive(self):
  x=np.array([[[-2.,0.,1.],[3.,-1.,0.]]],dtype=np.float32)
  s={k:np.asarray(v) for k,v in statistics(jnp.array(x)).items()}
  np.testing.assert_array_equal(s['neuron_positive_count'],[[1,0,1]])
  np.testing.assert_array_equal(s['position_positive_count'],[[1,1]])
  np.testing.assert_array_equal(s['position_zero_count'],[[1,1]])
  np.testing.assert_allclose(s['position_rms'],np.sqrt(np.mean(x*x,-1)),rtol=1e-6)
 def test_output_balance_matches_dense_reference(self):
  rng=np.random.default_rng(18);a=rng.normal(size=(2,7,3,8)).astype(np.float32);b=rng.normal(size=a.shape).astype(np.float32)
  s={k:np.asarray(v) for k,v in balance_statistics(jnp.array(a),jnp.array(b),jnp.array(a+b)).items()}
  for name,v in [('mha_square',a*a),('m_square',b*b),('cross',a*b),('sum_square',(a+b)**2)]:np.testing.assert_allclose(s[name],np.mean(v,axis=(-1,-2)),rtol=2e-6,atol=1e-6)
  np.testing.assert_array_equal(s['reconstruction_error'],np.zeros((2,7)))
if __name__=='__main__':unittest.main()

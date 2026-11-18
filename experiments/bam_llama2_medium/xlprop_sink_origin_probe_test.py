import unittest
import jax,jax.numpy as jnp,numpy as np
from xlprop_sink_origin_probe import split_scores,effective_keys,centered_gram
class SinkOriginTest(unittest.TestCase):
 def test_nope_rope_exact_score_partition(self):
  q=jax.random.normal(jax.random.key(1),(2,3,4,10));k=jax.random.normal(jax.random.key(2),(2,7,4,10))
  a,b=split_scores(q,k,6);np.testing.assert_allclose(a+b,jnp.einsum('bqhd,bshd->bhqs',q,k),atol=2e-6)
 def test_normalized_address_small_raw_norm_is_not_small_effective_key(self):
  raw=jnp.ones((1,1,4,3))*1e-3;gate=jnp.zeros((1,1,4));static=jnp.zeros((4,3));sg=jnp.ones_like(gate)
  y=effective_keys(raw,gate,static,sg,1e-12,1.)
  self.assertGreater(float(jnp.linalg.norm(y)),1.)
 def test_fixed_read_source_decomposition(self):
  key=jnp.array([1.,-2.,3.]);a=jnp.arange(15.).reshape(3,5);b=-a*.8;c=jnp.ones_like(a)
  np.testing.assert_allclose(key@(a+b+c),key@a+key@b+key@c,atol=1e-5)
 def test_centered_unembedding_gram_removes_common_logit_shift(self):
  w=jax.random.normal(jax.random.key(3),(5,17));g=centered_gram(w);v=jnp.arange(5.)
  z=v@w;np.testing.assert_allclose(v@g@v,jnp.sum((z-z.mean())**2),rtol=1e-5)
  shifted=w+jnp.arange(5.)[:,None];np.testing.assert_allclose(centered_gram(shifted),g,atol=6e-5)
if __name__=='__main__':unittest.main()

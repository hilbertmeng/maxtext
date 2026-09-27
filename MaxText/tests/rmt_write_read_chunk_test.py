"""Numerical/parameter regressions for token-local RMT write/read chunking."""
import contextlib
import io
import math
from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import rmt_mediumprop_test
from layers import rmt


class RMTWriteReadChunkTest(absltest.TestCase):
  _config = rmt_mediumprop_test.RMTMediumPropTest._config

  def _check(self, dtype, full_width=False):
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32')
    cfg.get_keys().update(dtype=dtype, rmt_block_scan=False)
    if full_width:
      cfg.get_keys().update(emb_dim=1200, head_dim=75)
    mlp_width = 4078 if full_width else 128
    matrix = jax.random.normal(jax.random.key(901), (1,4,48,cfg.head_dim)).astype(dtype)
    args = (jnp.ones((1,4), jnp.int32), jnp.arange(4)[None], True, 0)
    baseline = rmt.RMTLayer(cfg, mlp_dim=mlp_width)
    p0 = nn.unbox(baseline.init(jax.random.key(902), matrix, *args)['params'])
    def perturb(path, value):
      # Zero read keys must be active to test the complete state/gradient path.
      return value + .015 * jax.random.normal(jax.random.key(len(path)+903), value.shape)
    params = jax.tree_util.tree_map_with_path(perturb, p0)
    def evaluate(module,p):
      (y,_),aux = module.apply({'params':p},matrix,*args,mutable=['intermediates'])
      return jnp.mean(y.astype(jnp.float32)**2),(y,aux)
    reference, refgrad = jax.value_and_grad(lambda p:evaluate(baseline,p),has_aux=True)(params)
    # Full merged, chunked split, chunked merged, and one-chunk boundary case.
    for chunk, merge, unroll in ((0,True,False),(2,False,False),(2,True,False),(4,True,False),
                                 (2,False,True),(2,True,True)):
      cfg.get_keys().update(rmt_write_read_chunk_size=chunk,rmt_mlp_merge_reads=merge,
                            rmt_write_read_chunk_unroll=unroll)
      module = rmt.RMTLayer(cfg,mlp_dim=mlp_width)
      initialized = nn.unbox(module.init(jax.random.key(902), matrix, *args)['params'])
      self.assertEqual(jax.tree.structure(p0),jax.tree.structure(initialized))
      for x,y in zip(jax.tree.leaves(p0),jax.tree.leaves(initialized)):
        np.testing.assert_array_equal(x,y)
      actual,grad = jax.value_and_grad(lambda p:evaluate(module,p),has_aux=True)(params)
      for x,y in zip(jax.tree.leaves(reference),jax.tree.leaves(actual)):
        x,y=np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
        self.assertTrue(np.isfinite(y).all())
        if dtype==jnp.float32:
          np.testing.assert_allclose(x,y,rtol=3e-4,atol=3e-5)
        else:
          error=np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-7)
          self.assertLess(error,.02,(chunk,merge,error))
      x=np.concatenate([np.asarray(x,dtype=np.float32).ravel() for x in jax.tree.leaves(refgrad)])
      y=np.concatenate([np.asarray(x,dtype=np.float32).ravel() for x in jax.tree.leaves(grad)])
      error=np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-7)
      self.assertLess(error,2e-5 if dtype==jnp.float32 else .02,(chunk,merge,error))
      print('NUMERICS',str(dtype),chunk,merge,'gradient_relative_l2',float(error),flush=True)
      cfg.get_keys().update(rmt_write_read_chunk_size=0,rmt_mlp_merge_reads=False,
                            rmt_write_read_chunk_unroll=False)

  def test_fp32_values_health_gradients_and_parameter_seeds(self):
    self._check(jnp.float32)

  def test_bf16_values_health_gradients_and_parameter_seeds(self):
    self._check(jnp.bfloat16)

  def test_full_width_fp32(self):
    self._check(jnp.float32, full_width=True)

  def test_full_width_bf16(self):
    self._check(jnp.bfloat16, full_width=True)

  def test_chunk_argument_validation(self):
    with self.assertRaisesRegex(ValueError,'must divide'):
      rmt._map_token_chunks(lambda x:(x,), (jnp.zeros((2,7,3)),),4)


if __name__=='__main__':
  absltest.main()

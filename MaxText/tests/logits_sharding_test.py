"""Run with four forced CPU devices to check global vocabulary/batch reductions."""
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from layers import logits


class LogitsShardingTest(unittest.TestCase):
  def setUp(self):
    self.assertGreaterEqual(jax.device_count(),4)
    self.mesh=Mesh(np.array(jax.devices()[:4]).reshape(2,2),('batch','vocab'))

  def test_global_quantiles_over_batch_and_vocab_shards(self):
    z=(jnp.arange(8,dtype=jnp.float32)[:,None,None]**3*10 +
       jnp.arange(16,dtype=jnp.float32)[None,:,None] +
       jnp.arange(32,dtype=jnp.float32)[None,None,:]/8)
    mask=jnp.ones((8,16),jnp.int32).at[1,:].set(0)
    def statistics(z,mask):
      samples=[logits.chunk_health_samples(z[:,lo:lo+5],mask[:,lo:lo+5],lo,16,32)
               for lo in range(0,16,5)]
      return logits.summarize_health(jnp.concatenate(samples))
    sharded=jax.jit(statistics,in_shardings=(NamedSharding(self.mesh,P('batch',None,'vocab')),
                                           NamedSharding(self.mesh,P('batch',None))),
                    out_shardings=NamedSharding(self.mesh,P()))
    actual=sharded(z,mask)
    indices=logits.health_sample_positions(8,16,32)
    values=np.asarray(z).reshape(-1,32)[indices];valid=np.asarray(mask).reshape(-1)[indices]>0
    expected=[np.quantile(np.abs(values.mean(-1)[valid]),.95),np.median(values.std(-1)[valid])]
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)

  def test_fp32_output_bf16_backward_on_partitioned_head(self):
    h=jax.random.normal(jax.random.PRNGKey(1),(8,4,16)).astype(jnp.bfloat16)
    w=jax.random.normal(jax.random.PRNGKey(2),(16,32)).astype(jnp.bfloat16)
    hs=NamedSharding(self.mesh,P('batch',None,None));ws=NamedSharding(self.mesh,P(None,'vocab'))
    fn=jax.jit(jax.value_and_grad(lambda h,w:logits.bf16_dot_fp32_output(h,w).mean(),argnums=(0,1)),
                in_shardings=(hs,ws))
    loss,(dh,dw)=fn(h,w)
    self.assertTrue(np.isfinite(loss))
    self.assertEqual(dh.dtype,jnp.bfloat16);self.assertEqual(dw.dtype,jnp.bfloat16)
    expected=jax.grad(lambda h,w:jnp.matmul(h,w).astype(jnp.float32).mean(),argnums=(0,1))(h,w)
    for a,b in zip((dh,dw),expected):
      np.testing.assert_allclose(a.astype(jnp.float32),b.astype(jnp.float32),atol=5e-4,rtol=.01)


if __name__=='__main__':
  unittest.main()

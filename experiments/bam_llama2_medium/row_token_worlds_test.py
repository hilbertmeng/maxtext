"""Exhaustive small-sequence counterfactual proof, plus compensated arithmetic."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import (_attention_op, _bam_fetch_op,
    _split_row_difference, _subtract_row_increment, _row_consumer_value_edges)


class TokenWorldTest(unittest.TestCase):
  def test_own_world_value_consumers_match_each_origin(self):
    rng = np.random.default_rng(19)
    value = jnp.asarray(rng.normal(size=(1,5,2,3)), jnp.float32)
    reference = value + .2
    alpha = jnp.asarray(rng.uniform(size=(1,2,5,5)), jnp.float32)
    alpha *= jnp.tril(jnp.ones((5,5)))[None,None]
    y = jnp.einsum('bnts,bsnd->btnd', alpha, value)
    diagonal = jnp.eye(5, dtype=bool)
    for self_scale, cross_scale in [(0,1),(1,0),(1,1)]:
      actual = _row_consumer_value_edges(y,alpha,value,reference,diagonal,
          self_scale,cross_scale,foreign_prefix=True)
      for origin in range(5):
        one_ref = value.at[:,origin].set(reference[:,origin])
        exhaustive = _row_consumer_value_edges(y,alpha,value,one_ref,diagonal,
            self_scale,cross_scale)
        np.testing.assert_array_equal(actual[:,origin],exhaustive[:,origin])
      if not self_scale:
        np.testing.assert_array_equal(actual,y)

  def test_compensated_difference(self):
    rng=np.random.default_rng(2)
    a=jnp.asarray(rng.choice([-1,1],1024)*np.exp2(rng.uniform(-35,15,1024)),jnp.bfloat16)
    b=jnp.asarray(rng.choice([-1,1],1024)*np.exp2(rng.uniform(-35,15,1024)),jnp.bfloat16)
    high,low=jax.jit(_split_row_difference)(a,b)
    np.testing.assert_array_equal(np.asarray(high,dtype=float)+np.asarray(low,dtype=float),
        np.asarray(a,dtype=float)-np.asarray(b,dtype=float))
    result=jax.jit(lambda a,b:_subtract_row_increment(a,*_split_row_difference(a,b)))(a,b)
    np.testing.assert_array_equal(result,b)
    self.assertGreater(np.count_nonzero(np.asarray((a.astype(jnp.float32)-high).astype(a.dtype))!=b),0)

  def test_all_origins_match_exhaustive_causal_worlds(self):
    rng=np.random.default_rng(4);t=5;n=2;d=2
    x=jnp.asarray(rng.normal(size=(1,t,n*d)),jnp.float32)
    z=jnp.asarray(rng.normal(size=x.shape)*.1,jnp.float32)
    weights=[jnp.asarray(rng.normal(size=(4,4))*.2,jnp.float32) for _ in range(3)]
    diagonal=jnp.eye(t,dtype=bool)
    valid=jnp.tril(jnp.ones((1,t,t),bool))

    def run(remove,donor=None):
      h=x-z*remove[None,:,None]
      M=jnp.zeros((1,t,2,2),jnp.float32);refs=[]
      for layer in range(3):
        norm=h*jax.lax.rsqrt(jnp.mean(h*h,-1,keepdims=True)+1e-3)
        q,k,v=[(norm@w).reshape(1,t,n,d) for w in weights]
        refs.append(dict(key=k,value=v,M=M))
        foreign={} if donor is None else dict(foreign_key=donor[layer]['key'],
            foreign_value=donor[layer]['value'],diagonal_mask=diagonal)
        y,alpha=_attention_op(q,k,v,valid,**foreign)
        mixed=jnp.tanh(norm[...,:n])
        bar=_bam_fetch_op(alpha,M,mixed,diagonal,diagonal_one=True,
            foreign_state=None if donor is None else donor[layer]['M'])
        h=h+.2*y.reshape(h.shape)+.1*bar.reshape(h.shape)
        h=h+.1*jnp.tanh(h)
        M=M+.1*h[...,:2,None]*h[...,None,2:]
      return h,refs

    clean,clean_refs=run(jnp.zeros(t))
    deleted,deleted_refs=run(jnp.ones(t))
    local,_=run(jnp.ones(t),clean_refs)
    earlier,_=run(jnp.zeros(t),deleted_refs)
    for i in range(t):
      one=jnp.zeros(t).at[i].set(1)
      own_true,_=run(one);other_true,_=run(1-one)
      np.testing.assert_allclose(local[:,i],own_true[:,i],atol=2e-6,rtol=2e-6)
      np.testing.assert_allclose(earlier[:,i],other_true[:,i],atol=2e-6,rtol=2e-6)
    np.testing.assert_array_equal(run(jnp.zeros(t),clean_refs)[0],clean)
    np.testing.assert_array_equal(run(jnp.ones(t),deleted_refs)[0],deleted)


if __name__=='__main__':unittest.main()

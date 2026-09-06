"""CPU contracts for point-source consumer probes."""
import os
os.environ.setdefault('BAM_RESIDUAL_ATTR_BASE_CONFIG', 'BamLlama2MediumV2')
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import normalizations
from layers.attentions import _row_consumer_value_edges, ROW_CONSUMER_NAMES
from row_consumer_positions import arms, intervention_tree, origin_positions, position_effects


class ConsumerTest(unittest.TestCase):
  def test_v_source_and_edges(self):
    q0, s0, q, s, n, d = 3, 1, 4, 6, 2, 3
    targets = jnp.arange(q0,q0+q)[:,None]
    sources = jnp.arange(s0,s0+s)[None,:]
    alpha = jax.nn.softmax(jnp.where((sources<=targets)[None,None],
        jax.random.normal(jax.random.key(1),(1,n,q,s)), -1e20),-1)
    value = jax.random.normal(jax.random.key(2),(1,s,n,d)).astype(jnp.bfloat16)
    reference = value.at[:,2].add(1)  # global source position 3
    y = jnp.einsum('bnqs,bsnd->bqnd',alpha,value.astype(jnp.float32))
    for a,b in [(0,0),(1,0),(0,1),(1,1)]:
      out = _row_consumer_value_edges(y,alpha,value,reference,sources==targets,a,b)
      expected = y + jnp.einsum('bnqs,bsnd->bqnd',alpha*jnp.where(
          (sources==targets)[None,None],a,b),reference.astype(jnp.float32)-value.astype(jnp.float32))
      np.testing.assert_allclose(out,expected,atol=1e-6)
      null = _row_consumer_value_edges(y,alpha,value,value,sources==targets,a,b)
      np.testing.assert_array_equal(null,y)
    self_only = _row_consumer_value_edges(y,alpha,value,reference,sources==targets,1,0)-y
    np.testing.assert_array_equal(self_only[:,1:],0)
    cross_only = _row_consumer_value_edges(y,alpha,value,reference,sources==targets,0,1)-y
    np.testing.assert_array_equal(cross_only[:,:1],0)

  def test_tree(self):
    c=jnp.zeros((24,len(ROW_CONSUMER_NAMES))).at[12,3].set(1)
    mask=jnp.array([[False,True,False]])
    z=jnp.zeros((1,3,4))
    for scanned in (False,True):
      layers=({'layers':{'self_attention':{'abs_v_cache_projection':jnp.zeros((24,32,8))}}}
              if scanned else {f'layers_{i}':{'self_attention':{
                'abs_v_cache_projection':jnp.zeros((32,8))}} for i in range(24)})
      tree=intervention_tree({'params':{'decoder':layers}},jnp.ones((24,3)),mask,c,z,scanned)
      if scanned:
        layer=tree['decoder']['layers']
        np.testing.assert_array_equal(layer['row_consumers'],c)
        self.assertEqual(layer['row_consumer_z'].shape,(24,1,3,4))
        self.assertEqual(layer['self_attention']['row_source_mask'].shape,(24,1,3))
      else:
        for i in range(24):
          np.testing.assert_array_equal(tree['decoder'][f'layers_{i}']['row_consumers'],c[i])

  def test_positions_arms_and_reductions(self):
    cohort=dict(sequence_hashes=np.array(['a','b']),targets_segmentation=np.ones((2,2048)))
    np.testing.assert_array_equal(origin_positions(cohort),origin_positions(cohort))
    self.assertTrue(np.all((origin_positions(cohort)>=64)&(origin_positions(cohort)<1792)))
    matrix=arms(11)
    self.assertEqual(len({a['name'] for a in matrix}),len(matrix))
    for a in matrix:
      np.testing.assert_array_equal(a['control'][:11],0)
      np.testing.assert_array_equal(a['control'][11,:8],0)
    token=np.zeros((1,2,20));token[0,1,5]=2;token[0,1,6]=3
    e=position_effects(token,np.ones((1,20),bool),np.array([5]))
    self.assertEqual(e[0,1,0],2);self.assertEqual(e[0,1,1],3)
    self.assertEqual(e[0,1,-1],0)

  def test_norm_parameter_reuse(self):
    class Pair(nn.Module):
      @nn.compact
      def __call__(self,x,z=None):
        norm=normalizations.RMSNorm(name='norm',dtype=jnp.bfloat16)
        y=norm(x)
        return y if z is None else (y,norm((x.astype(jnp.float32)-z).astype(x.dtype)))
    x=jnp.ones((1,4,8),jnp.bfloat16)
    module=Pair(); p=module.init(jax.random.key(0),x)
    p2=module.init(jax.random.key(0),x,jnp.zeros_like(x))
    self.assertEqual(jax.tree.structure(p),jax.tree.structure(p2))
    for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(p2)):
      np.testing.assert_array_equal(a,b)
    a,b=module.apply(p,x,jnp.zeros_like(x))
    np.testing.assert_array_equal(a,b)

  def test_interaction_arms(self):
    with patch.dict(os.environ, {'BAM_CONSUMER_ARM_SET':'interactions'}):
      matrix={a['name']:a['control'] for a in arms(11)}
    mlp=ROW_CONSUMER_NAMES.index('mlp');v=ROW_CONSUMER_NAMES.index('v_cross')
    joint=matrix['joint_source_mlp_cross_v']
    self.assertEqual(joint[11,mlp],1)
    np.testing.assert_array_equal(joint[12:16,v],1)
    self.assertEqual(joint.sum(),5)
    self.assertEqual(matrix['joint_source_and_downstream_mlp_cross_v'].sum(),9)
    self.assertEqual(matrix['joint_downstream_all_v'].sum(),8)

  def test_neighbor_arms(self):
    from row_neighbors import neighbor_arms
    names,scales=neighbor_arms(range(8,15))
    self.assertEqual(len(names),22)
    for layer in range(8,15):
      for name,expected in [('cross',[0,0,1]),('self',[1,1,0]),('both',[0,0,0])]:
        s=scales[names.index(f'L{layer}_{name}')]
        np.testing.assert_array_equal(s[layer],expected)
        np.testing.assert_array_equal(np.delete(s,layer,axis=0),1)


if __name__=='__main__':unittest.main()

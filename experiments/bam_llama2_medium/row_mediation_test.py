"""Verify mediator endpoints, actual source/target edge masks and layer addressing."""
import os
import sys
from pathlib import Path
import unittest
os.environ.setdefault('BAM_RESIDUAL_ATTR_BASE_CONFIG','BamLlama2MediumV2')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import _mediation_replace, _mediation_value_edges
from row_mediation import patch_tree, arms, REF_NAMES, CONTROL_NAMES


class MediationTest(unittest.TestCase):
  def test_value_edges(self):
    # Nonzero q0: diagonal is global source==target, not local block indices.
    q0=2; t=5; q=3; n=2; d=4
    logits=jax.random.normal(jax.random.key(1),(1,n,q,t))
    valid=jnp.arange(t)[None,:]<=jnp.arange(q0,q0+q)[:,None]
    alpha=jax.nn.softmax(jnp.where(valid[None,None],logits,-1e20),-1)
    v=jax.random.normal(jax.random.key(2),(1,t,n,d)); ref=v*2+.5
    diagonal=jnp.arange(t)[None,:]==jnp.arange(q0,q0+q)[:,None]
    y=jnp.einsum('bnqs,bsnd->bqnd',alpha,v)
    for s,c in [(0,0),(1,1),(1,0),(0,1)]:
      expected=y+jnp.einsum('bnqs,bsnd->bqnd',
          alpha*jnp.where(diagonal[None,None],s,c),ref-v)
      result=_mediation_value_edges(y,alpha,v,ref,diagonal,jnp.array([s,c]))
      np.testing.assert_allclose(result,expected,atol=1e-6)
    bf=y.astype(jnp.bfloat16)
    np.testing.assert_array_equal(_mediation_value_edges(bf,alpha,v,ref,diagonal,jnp.zeros(2)),bf)
    np.testing.assert_array_equal(_mediation_replace(bf,bf+1,0),bf)

  def test_tree_and_arms(self):
    refs={k:jnp.zeros((24,1,2,3)) for k in REF_NAMES}
    c=jnp.zeros((24,len(CONTROL_NAMES))).at[12,7].set(1)
    for scan in [True,False]:
      layers={'layers':{'self_attention':{'abs_v_cache_projection':jnp.zeros((24,32,8))}}} if scan else {
          f'layers_{i}':{'self_attention':{'abs_v_cache_projection':jnp.zeros((32,8))}} for i in range(24)}
      x=patch_tree({'params':{'decoder':layers}},jnp.ones((24,2)),refs,c,jnp.zeros((1,2,3)),scan)
      if scan:
        np.testing.assert_array_equal(x['decoder']['layers']['self_attention']['med_v_edges'],c[:,6:8])
        np.testing.assert_array_equal(x['decoder']['layers']['self_attention']['med_qk_routes'],c[:,8:10])
      else:
        for i in range(24):np.testing.assert_array_equal(
            x['decoder'][f'layers_{i}']['self_attention']['med_v_edges'],c[i,6:8])
    a=arms(11,'coarse'); names=[x['name'] for x in a]
    self.assertEqual(len(set(names)),len(names))
    self.assertNotIn('cut_L23_mlp',names)
    self.assertIn('cut_L11_attention',names)
    joint=arms(11,'joint')
    self.assertTrue(any(x['name']=='rescue_L12-23_std+mlp+full+M' for x in joint))
    for arm in joint:
      self.assertFalse(np.any(arm['control'][:12]))
    routing=arms(11,'routing',[12])
    by_name={a['name']:a for a in routing}
    qk=by_name['rescue_L12_qk_mha']['control'][12]
    np.testing.assert_array_equal(qk[6:10],[0,0,1,0])
    qkv=by_name['rescue_L12_qk_mha+v_self+v_cross']['control'][12]
    np.testing.assert_array_equal(qkv[6:10],[1,1,1,0])


if __name__=='__main__':unittest.main()

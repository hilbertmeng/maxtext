import unittest,runpy
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
m=runpy.run_path(str(Path(__file__).with_name('xlprop_localov_stopgrad_probe.py')))
class Tests(unittest.TestCase):
 def test_masks_exact_firstsix(self):
  arms=dict(m['arm_masks']());self.assertEqual(len(arms),17)
  self.assertEqual(np.argwhere(arms['O_first6']).tolist(),[[l,3]for l in range(6)])
  self.assertEqual(np.argwhere(arms['V_first6']).tolist(),[[l,2]for l in range(6)])
  self.assertFalse(arms['OV_first6'][6:].any());self.assertTrue(arms['identity'].sum()==0)
 def test_dynamic_cut_preserves_forward_and_WO(self):
  def f(a,w,cut):return jnp.sum((w@(m['cut_value'](a,cut)+2*a))**2)
  a=jnp.array([1.,2.]);w=jnp.array([[1.,.2],[.3,1.]])
  run=jax.jit(jax.value_and_grad(f,argnums=(0,1)))
  v0,(g0,w0)=run(a,w,jnp.array(False));v1,(g1,w1)=run(a,w,jnp.array(True))
  np.testing.assert_array_equal(v0,v1);np.testing.assert_array_equal(w0,w1)
  np.testing.assert_allclose(g1,g0*2/3,rtol=1e-6)
 def test_cancellation_can_increase_norm(self):
  def f(x,c):return jnp.sum(2*x- m['cut_value'](x,c))
  self.assertEqual(float(jax.grad(f)(1.,False)),1.)
  self.assertEqual(float(jax.grad(f)(1.,True)),2.)
 def test_donating_native_control_copies_all_state_leaves(self):
  native=jax.jit(lambda state:jax.tree.map(lambda x:x+1,state),donate_argnums=(0,))
  state={'params':jnp.array([1.,2.]),'moments':jnp.array([3.,4.])}
  before={k:np.asarray(v).copy() for k,v in state.items()}
  for _ in range(2):
   trial=jax.tree.map(lambda x:x.copy(),state)
   updated=native(trial);jax.block_until_ready(updated)
   for k in state:np.testing.assert_array_equal(state[k],before[k])
 def test_static_o_selection_excludes_explicit_terminal(self):
  names=['params/decoder/layers/block/self_attention/static_o_key','params/decoder/final_local_layer/block/self_attention/static_o_key']
  chosen=[n for n in names if '/decoder/layers/block/self_attention/static_o_key' in n]
  self.assertEqual(chosen,[names[0]])
 def test_matrix_address_gradient_and_svd(self):
  M=jnp.diag(jnp.array([3.,1.]));g=jnp.array([2.,1.]);a=jnp.array([.5,2.])
  gm,ga=jax.grad(lambda mm,aa:jnp.vdot(mm.T@aa,g),argnums=(0,1))(M,a)
  np.testing.assert_allclose(gm,jnp.outer(a,g));np.testing.assert_allclose(ga,M@g)
  _,s,vh=np.linalg.svd(M)
  self.assertAlmostEqual(float(jnp.linalg.norm(ga)**2),float(np.sum((s*(vh@np.asarray(g)))**2)),places=5)
if __name__=='__main__':unittest.main()

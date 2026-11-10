"""Static pre read: independent formula, exact initialization, gradients and model regression."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import residual_connections_test as ref
from layers.residual_connections import ManifoldHyperConnection
from layers.models import Transformer
import max_utils

BASE = 'BamMHAMediumPropC256TruePilemHC'
RUN = 'BamMHAMediumPropC256TruePilemHCstatic'

class StaticRouterTest(unittest.TestCase):
  def test_initial_selection_and_local_gradients(self):
    # Orthogonal streams let the read expose each of the four pre coefficients.
    h = jnp.eye(4).reshape(1, 1, 4, 4)
    cfg = ref.connection_config(mhc_pre_static=True, mhc_pre_bias_init=0., mhc_pre_selected_bias_init=1.)
    old = ManifoldHyperConnection(ref.connection_config(), 1)
    for index in range(4):
      m = ManifoldHyperConnection(cfg, index)
      p = nn.unbox(m.init(jax.random.key(3), h)['params'])
      read, post, mix = m.apply({'params': p}, h)
      np.testing.assert_array_equal(read, np.eye(4)[index].reshape(1,1,4))
      np.testing.assert_array_equal(post, np.ones((1,1,4)))
      # All unchanged output paths are numerically identical, including SK20.
      _, old_post, old_mix = old.apply({'params': p}, h)
      np.testing.assert_array_equal(post, old_post)
      np.testing.assert_array_equal(mix, old_mix)
      g = jax.grad(lambda pp: m.apply({'params': pp}, h)[0].sum())(p)
      np.testing.assert_array_equal(g['router_bias'][:4], np.ones(4))
      norm = np.asarray(h).reshape(1,1,16)
      norm /= np.sqrt(np.mean(norm**2, axis=-1, keepdims=True) + cfg.mhc_norm_epsilon)
      np.testing.assert_allclose(g['router_kernel'][:,:4], np.broadcast_to(norm[0,0,:,None]*.01*.25,(16,4)), atol=1e-8)
      np.testing.assert_array_equal(g['router_bias'][4:], np.zeros(20))

  def test_learned_signed_formula_and_gradients(self):
    cfg = ref.connection_config(mhc_pre_static=True, mhc_pre_bias_init=0., mhc_pre_selected_bias_init=1.)
    h = jax.random.normal(jax.random.key(2), (2,3,4,8))
    m = ManifoldHyperConnection(cfg, 2)
    p = nn.unbox(m.init(jax.random.key(3), h)['params'])
    p['router_kernel'] = jax.random.normal(jax.random.key(4),(32,24))*.3
    p['router_scale'] = jnp.array([.7,.4,.2])
    p['router_bias'] = p['router_bias'].at[:4].set(jnp.array([-.4,.2,1.2,-.1]))
    flat = np.asarray(h).reshape(2,3,32)
    norm = flat / np.sqrt(np.mean(flat**2, axis=-1,keepdims=True)+cfg.mhc_norm_epsilon)
    raw = norm @ np.asarray(p['router_kernel'])
    pre = np.asarray(p['router_bias'][:4]) + 1/(1+np.exp(-raw[...,:4]*.7))-.5
    self.assertLess(pre.min(),0)
    r,post,mix = m.apply({'params':p},h)
    np.testing.assert_allclose(r,(pre[...,None]*np.asarray(h)).sum(-2),rtol=2e-6,atol=1e-6)
    old = ManifoldHyperConnection(ref.connection_config(),2)
    _,op,om=old.apply({'params':p},h)
    np.testing.assert_array_equal(post,op);np.testing.assert_array_equal(mix,om)
    g=jax.grad(lambda pp: jnp.square(m.apply({'params':pp},h)[0]).sum())(p)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(g)))
    for key in ('router_kernel','router_bias','router_scale','norm_scale'):
      self.assertGreater(np.linalg.norm(g[key]),0)

class StaticModelTest(unittest.TestCase):
  setUp = ref.ResidualModelTest.setUp
  tearDown = ref.ResidualModelTest.tearDown
  config = ref.MLPWriteTest.config
  tiny = ref.ResidualModelTest.tiny
  check_model = ref.ResidualModelTest.check_model

  def test_recipe_parameter_budget_and_backbone_parity(self):
    a,b=self.config(BASE),self.config(RUN)
    allowed={'exp_class','model_name','compare_runs','mhc_pre_static','mhc_pre_bias_init','mhc_pre_selected_bias_init'}
    diff={k for k in a.get_keys() if str(a.get_keys()[k])!=str(b.get_keys()[k])}
    self.assertFalse(diff-allowed,diff)
    self.assertEqual(b.mlp_dim,3133);self.assertEqual(b.mhc_sinkhorn_iters,20)
    def initialize(cfg,abstract=False):
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        model=Transformer(cfg,mesh,quant=None)
        fn=lambda t: nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,t,t,t,decoder_segment_ids=t,enable_dropout=False)['params'])
        return jax.eval_shape(fn,jax.ShapeDtypeStruct((1,4),jnp.int32)) if abstract else fn(jnp.ones((1,4),jnp.int32))
    sa,sb=initialize(a,True),initialize(b,True)
    self.assertEqual(jax.tree.structure(sa),jax.tree.structure(sb))
    for x,y in zip(jax.tree.leaves(sa),jax.tree.leaves(sb)): self.assertEqual((x.shape,x.dtype),(y.shape,y.dtype))
    self.assertEqual(sum(x.size for x in jax.tree.leaves(sb)),432124577)
    pa,pb=map(flatten_dict,(initialize(self.tiny(BASE)),initialize(self.tiny(RUN))))
    for key in pa:
      if key[-1]=='router_bias':
        np.testing.assert_array_equal(pa[key][4:],pb[key][4:])
        self.assertEqual(float(pb[key][:4].sum()),1.)
      else: np.testing.assert_array_equal(pa[key],pb[key])

  def test_forward_backward_causality_remat_bf16(self):
    self.check_model(RUN)

if __name__=='__main__': unittest.main()

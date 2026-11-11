"""Controlled identity-mixing ablation: initialization, gradients and full model."""
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import residual_connections_test as ref
from layers.residual_connections import ManifoldHyperConnection, mhc_update
from layers.models import Transformer
import max_utils

BASE = 'BamMHAMediumPropC256TruePilemHC'
RUN = 'BamMHAMediumPropC256TruePileiHC'

class IdentityRouterTest(unittest.TestCase):
  def test_same_initialization_identity_and_live_pre_post(self):
    a = ref.connection_config()
    b = ref.connection_config(mhc_identity_mixing=True)
    state = jax.random.normal(jax.random.key(5), (2, 3, 4, 8))
    ordinary, identity = ManifoldHyperConnection(a, 3), ManifoldHyperConnection(b, 3)
    pa = nn.unbox(ordinary.init(jax.random.key(3), state)['params'])
    pb = nn.unbox(identity.init(jax.random.key(3), state)['params'])
    for x, y in zip(jax.tree.leaves(pa), jax.tree.leaves(pb)):
      np.testing.assert_array_equal(x, y)
    # Use learned, asymmetric gates: test beyond zero-projection initialization.
    pb['router_kernel'] = jax.random.normal(jax.random.key(6), (32, 24)) * .1
    pb['router_scale'] = jnp.array([.3, .2, .5])
    with patch('layers.residual_connections.sinkhorn', side_effect=AssertionError('iHC evaluated Sinkhorn')):
      read, post, mix = identity.apply({'params': pb}, state)
    np.testing.assert_array_equal(mix, np.broadcast_to(np.eye(4), (2, 3, 4, 4)))
    old_read, old_post, _ = ordinary.apply({'params': pb}, state)
    np.testing.assert_array_equal(read, old_read)
    np.testing.assert_array_equal(post, old_post)
    out = mhc_update(state, read, post, mix)
    np.testing.assert_allclose(out, state + post[..., None] * read[..., None, :], atol=1e-6)
    def objective(p):
      r, ppost, mm = identity.apply({'params': p}, state)
      return jnp.square(mhc_update(state, r, ppost, mm)).sum()
    grad = jax.grad(objective)(pb)
    for sl in (slice(0,4), slice(4,8)):
      self.assertGreater(float(jnp.linalg.norm(grad['router_kernel'][:, sl])), 0)
    np.testing.assert_array_equal(grad['router_kernel'][:, 8:], 0)
    np.testing.assert_array_equal(grad['router_bias'][8:], 0)
    self.assertEqual(float(grad['router_scale'][2]), 0)
    # BF16 state update is the FP32 sum rounded once, as in the parent.
    bf = state.astype(jnp.bfloat16)
    r, ppost, mm = identity.apply({'params': pb}, bf)
    expected = (bf.astype(jnp.float32) + ppost[..., None] * r.astype(jnp.float32)[...,None,:]).astype(bf.dtype)
    np.testing.assert_array_equal(mhc_update(bf, r, ppost, mm), expected)

class IdentityModelTest(unittest.TestCase):
  setUp = ref.ResidualModelTest.setUp
  tearDown = ref.ResidualModelTest.tearDown
  config = ref.MLPWriteTest.config
  tiny = ref.ResidualModelTest.tiny
  check_model = ref.ResidualModelTest.check_model

  def test_recipe_and_full_shape_initialization(self):
    a, b = self.config(BASE), self.config(RUN)
    allowed = {'exp_class', 'model_name', 'mhc_identity_mixing', 'compare_runs'}
    differences = {k for k in a.get_keys() if str(a.get_keys()[k]) != str(b.get_keys()[k])}
    self.assertFalse(differences - allowed, differences)
    self.assertEqual(a.mlp_dim, b.mlp_dim)
    self.assertFalse(a.mhc_identity_mixing)
    self.assertTrue(b.mhc_identity_mixing)
    # Abstract full recipe verifies the complete tree without allocating 432M weights.
    tokens = jax.ShapeDtypeStruct((1, 4), jnp.int32)
    def shapes(cfg):
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = Transformer(cfg, mesh, quant=None)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        return jax.eval_shape(lambda t: nn.unbox(model.init({'params': jax.random.key(7), 'dropout': jax.random.key(7)},
           t, t, t, t, decoder_segment_ids=t, enable_dropout=False)['params']), tokens)
    sa, sb = shapes(a), shapes(b)
    self.assertEqual(jax.tree.structure(sa), jax.tree.structure(sb))
    for x,y in zip(jax.tree.leaves(sa), jax.tree.leaves(sb)):
      self.assertEqual((x.shape,x.dtype), (y.shape,y.dtype))
    print('FULL_PARAMETER_COUNT', sum(x.size for x in jax.tree.leaves(sb)), 'MLP_DIM', b.mlp_dim, flush=True)
    # Actual tiny parameters prove same seed produces identical backbone and head.
    ta, tb = self.tiny(BASE), self.tiny(RUN)
    def initialize(cfg):
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        model = Transformer(cfg,mesh,quant=None)
        t = jnp.ones((1,4),jnp.int32)
        return nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,t,t,t,
          decoder_segment_ids=t,enable_dropout=False)['params'])
    for x,y in zip(jax.tree.leaves(initialize(ta)),jax.tree.leaves(initialize(tb))):
      np.testing.assert_array_equal(x,y)

  def test_forward_backward_causality_remat_and_bf16(self):
    self.check_model(RUN)

if __name__ == '__main__':
  unittest.main()

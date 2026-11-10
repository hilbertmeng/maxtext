"""SK3 controlled recipe: identical parameters, full-model gradient and numerical projection."""
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
RUN = 'BamMHAMediumPropC256TruePilemHCSK3'

class SK3ModelTest(unittest.TestCase):
  setUp = ref.ResidualModelTest.setUp
  tearDown = ref.ResidualModelTest.tearDown
  config = ref.MLPWriteTest.config
  tiny = ref.ResidualModelTest.tiny
  check_model = ref.ResidualModelTest.check_model

  def test_recipe_and_full_shape_initialization(self):
    a, b = self.config(BASE), self.config(RUN)
    allowed = {'exp_class', 'model_name', 'mhc_sinkhorn_iters', 'compare_runs'}
    differences = {k for k in a.get_keys() if str(a.get_keys()[k]) != str(b.get_keys()[k])}
    self.assertFalse(differences - allowed, differences)
    self.assertEqual(a.mlp_dim, b.mlp_dim)
    self.assertFalse(a.mhc_identity_mixing)
    self.assertFalse(b.mhc_identity_mixing)
    self.assertEqual(a.mhc_sinkhorn_iters,20)
    self.assertEqual(b.mhc_sinkhorn_iters,3)
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

class ProjectionTest(unittest.TestCase):
  def test_finite_projection_and_gradients(self):
    from layers.residual_connections import sinkhorn
    logits = jax.random.normal(jax.random.key(91),(64,4,4))*3
    for k in (3,20):
      m=sinkhorn(logits,k,1e-6)
      row=float(jnp.max(jnp.abs(m.sum(-1)-1)))
      col=float(jnp.max(jnp.abs(m.sum(-2)-1)))
      grad=jax.grad(lambda x:jnp.square(sinkhorn(x,k,1e-6)).sum())(logits)
      self.assertTrue(bool(jnp.isfinite(m).all() & jnp.isfinite(grad).all()))
      self.assertLess(col,1e-5)
      print('PROJECTION',k,'max_row_error',row,'max_col_error',col,flush=True)

if __name__ == '__main__':
  unittest.main()

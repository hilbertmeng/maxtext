"""Check standard-arm QKNorm scope against the historical full-head circuit."""
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import train_compile
from bam_full_qknorm_test import FullQKNormTest
from bam_prop_test_utils import MLPWriteTest
from layers import attentions, dc
from layers.models import Transformer


STANDARD = 'BamMediumPropMShiftReadG050StandardQKNormTruePile'
FULL = 'BamMediumPropMShiftReadG050FullQKNormTruePile'
PARENT = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'


class StandardQKNormTest(MLPWriteTest, unittest.TestCase):
  mesh = FullQKNormTest.mesh
  tiny = FullQKNormTest.tiny

  def test_parameter_trees_and_same_initial_parameters(self):
    trees = {}
    for arm, width in ((PARENT, None), (FULL, 75), (STANDARD, 18)):
      c = self.config(arm)
      args, *_ = train_compile.get_shaped_inputs(self.mesh(c), c)
      flat = flatten_dict(nn.unbox(args[0].params))
      trees[arm] = flat
      norms = {p: v for p, v in flat.items() if 'qk_norm' in p}
      self.assertEqual(sum(np.prod(v.shape) for v in norms.values()), 0 if width is None else 36*width)
      self.assertTrue(all(width in v.shape for v in norms.values()))
      self.assertEqual(c.normalization_layer_epsilon, 1e-6)
      if arm != PARENT:  # Historical no-QKNorm training retained generic health.
        self.assertFalse(c.record_training_health_metrics)
        self.assertFalse(c.record_internal_nn_metrics)
        self.assertFalse(c.bam_record_m_shift_health)
    base = trees[PARENT]
    for arm in (FULL, STANDARD):
      self.assertEqual({p: v.shape for p, v in trees[arm].items() if 'qk_norm' not in p},
                       {p: v.shape for p, v in base.items()})
    totals = {arm: sum(int(np.prod(v.shape)) for v in flat.values()) for arm, flat in trees.items()}
    self.assertEqual(totals[STANDARD]-totals[PARENT], 648)
    self.assertEqual(totals[FULL]-totals[PARENT], 2700)
    print('STANDARD_QKNORM_PARAMETER_COUNTS', totals, flush=True)
    # Adding the module must not perturb random initialization of the other parameters.
    initialized = {}
    for arm in (PARENT, STANDARD):
      c = self.tiny(arm)
      c.get_keys().update(scan_layers=False, remat_policy='none')
      mesh = self.mesh(c)
      model = Transformer(c, mesh, quant=None)
      tokens = jnp.array([[1, 2, 3, 4]])
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = model.init(jax.random.key(24), tokens, jnp.arange(4)[None], tokens,
                       jnp.ones_like(tokens), jnp.ones_like(tokens), enable_dropout=False)['params']
      initialized[arm] = flatten_dict(nn.unbox(p))
    for path, value in initialized[PARENT].items():
      np.testing.assert_array_equal(value, initialized[STANDARD][path], err_msg=str(path))

  def test_standard_arm_scope_and_rope_order(self):
    c = self.tiny(STANDARD)
    c.get_keys().update(scan_layers=False, remat_policy='none')
    mesh = self.mesh(c)
    model = Transformer(c, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]])
    events = []
    norm = dc.QKNorm.__call__
    rope = attentions.Attention.apply_rotary_embedding
    def capture_norm(module, q, k):
      events.append(('norm', q.shape[-1]))
      return norm(module, q, k)
    def capture_rope(module, x, *args, **kwargs):
      events.append(('rope', x.shape[-1]))
      return rope(module, x, *args, **kwargs)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), \
        mock.patch.object(dc.QKNorm, '__call__', capture_norm), \
        mock.patch.object(attentions.Attention, 'apply_rotary_embedding', capture_rope):
      model.init(jax.random.key(24), tokens, jnp.arange(4)[None], tokens,
                 jnp.ones_like(tokens), jnp.ones_like(tokens), enable_dropout=False)
    self.assertEqual(events, [('norm', 18), ('rope', 18), ('rope', 18)]*3)

  def test_scanned_gradients_and_causality(self):
    c = self.tiny(STANDARD)
    mesh = self.mesh(c)
    model = Transformer(c, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]])
    mask = jnp.ones_like(tokens)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      params = model.init(jax.random.key(24), tokens, jnp.arange(4)[None], tokens,
                          mask, mask, enable_dropout=False)['params']
      def forward(p, t):
        return model.apply({'params': p}, t, jnp.arange(4)[None], t, mask, mask,
                           enable_dropout=False)[0]
      loss, grads = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p, tokens)**2)))(params)
      flat = flatten_dict(nn.unbox(grads))
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
      for name in ('q_norm', 'k_norm', 'M_shift_gate'):
        self.assertGreater(sum(float(jnp.sum(v*v)) for path, v in flat.items() if name in path), 0.)
      np.testing.assert_allclose(forward(params, tokens)[:, :3],
          forward(params, tokens.at[:, 3].set(5))[:, :3], atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
  unittest.main()

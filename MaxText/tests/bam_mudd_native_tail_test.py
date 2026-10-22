"""Native Mudd reuse: joint controllers, exact budgets, identity and scanned gradients."""

import math
from types import SimpleNamespace
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import bam_mudd_tail_test as tail_reference
from bam_prop_test_utils import MLPWriteTest
from layers import fusion, mudd
from layers.models import Transformer
import max_utils
import pyconfig
import train_compile


QKV = 'BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile'
ONE = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile'
TWO = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile'
SWAP = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile'


class NativeTailTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, experiment=ONE):
    cfg = tail_reference.MuddTailTest.tiny(self, experiment)
    cfg.get_keys()['bam_pallas_core'] = False  # CPU reference; formal trees retain Pallas.
    return cfg

  def test_formal_budget_joint_layout_and_router_exemption(self):
    for exp, total, router_count, middle_width, last_width, targets in (
        (QKV, 432266060, 154380, 3803, 3026, 3),
        (ONE, 432266255, 154575, 3803, 3026, 4),
        (TWO, 432344635, 232155, 3027, 3026, 4),
        (SWAP, 432266255, 154575, 3026, 3803, 4)):
      with self.subTest(exp=exp):
        cfg = self.config(exp, bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                              query_chunk_size=256)  # v8t's token tiles require >=128.
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        shaped, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
        flat = flatten_dict(nn.unbox(shaped[0].params))
        reference_cfg = pyconfig.HyperParameters(SimpleNamespace(
            keys=dict(cfg.get_keys(), bam_pallas_core=False)))
        reference, _, _, _ = train_compile.get_shaped_inputs(mesh, reference_cfg)
        reference = flatten_dict(nn.unbox(reference[0].params))
        self.assertEqual(set(flat), set(reference))
        for path in flat:
          self.assertEqual((flat[path].shape, flat[path].dtype),
                           (reference[path].shape, reference[path].dtype), path)
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
        routers = {p: v for p, v in flat.items()
                   if 'qkvm_compose' in p or 'compose_final_three_source' in p}
        self.assertEqual(sum(math.prod(v.shape) for v in routers.values()), router_count)
        self.assertEqual(total - router_count - 432111680, 800 if exp == TWO else 0)
        prefix = ('params', 'decoder', 'tail_layers')
        layers = ('local_1', 'fetch_2') if exp == TWO else ('fetch_2',)
        for layer in layers:
          p = prefix + (layer, 'block', 'qkvm_compose', 'mlp')
          self.assertEqual(flat[p + ('dynamic_dense_conn1', 'kernel')].shape, (1200, 1, 64))
          self.assertEqual(flat[p + ('dynamic_dense_conn2', 'kernel')].shape, (64, 1, targets, 3))
          self.assertEqual(flat[p + ('dense_proj2.bias',)].shape, (targets, 1, 3))
          attention = prefix + (layer, 'block', 'self_attention')
          for name in ('W_R_up', 'static_o_key', 'M_shift_gate'):
            self.assertTrue(any(p[:len(attention)] == attention and name in p for p in flat))
          self.assertFalse(any(p[:len(attention)] == attention and
                               any(n in p for n in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0'))
                               for p in flat))
        self.assertEqual(flat[prefix + ('local_1', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                         (1200, 1, middle_width))
        self.assertEqual(flat[prefix + ('fetch_2', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                         (1200, 1, last_width))
        self.assertFalse(any(n in p for p in flat for n in
                             ('value_compose', 'q_compose', 'k_compose', 'm_compose')))

  def test_v8t_tail_layout_bridge_preserves_parameters_outputs_and_gradients(self):
    for exp, index in ((QKV, 8), (ONE, 8), (TWO, 7), (TWO, 8)):
      with self.subTest(exp=exp, index=index):
        cfg = self.tiny(exp)
        accelerated = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
            cfg.get_keys(), bam_pallas_core=True, bam_pallas_body='v8t')))
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        original = fusion.SubDecoderLayer(cfg, mesh, layer_inx=index)
        bridge = fusion.SubDecoderLayer(accelerated, mesh, layer_inx=index)
        inputs = tuple(jax.random.normal(jax.random.key(i + 70), (1, 4, 150))
                       for i in range(3))
        matrix = .2 * jax.random.normal(jax.random.key(73), (1, 4, 75, 32))
        tokens = jnp.ones((1, 4), jnp.int32)
        kwargs = dict(decoder_segment_ids=tokens, decoder_positions=jnp.arange(4)[None],
                      decoder_input_tokens=tokens, deep_embedding=None, deterministic=True,
                      model_mode='train', eos_sum=None, layer_index=index)
        def apply(module, parameters, x, m):
          return module.apply({'params': parameters}, x[0], **kwargs, M_in=m,
                              early_residuals=x[1:])
        rng = {'params': jax.random.key(74), 'dropout': jax.random.key(75)}
        with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
          parameters = nn.unbox(original.init(
              rng, inputs[0], **kwargs, M_in=matrix, early_residuals=inputs[1:])['params'])
          other = nn.unbox(bridge.init(rng, inputs[0], **kwargs,
              M_in=jnp.swapaxes(matrix, -1, -2), early_residuals=inputs[1:])['params'])
          a, b = flatten_dict(parameters), flatten_dict(other)
          self.assertEqual(set(a), set(b))
          for path in a:
            np.testing.assert_array_equal(a[path], b[path])
            if 'dynamic_dense_conn2' in path:
              a[path] = a[path] + .01 * jax.random.normal(jax.random.key(76), a[path].shape)
          parameters = unflatten_dict(a)
          expected, expected_m = apply(original, parameters, inputs, matrix)
          actual, actual_m = apply(bridge, parameters, inputs, jnp.swapaxes(matrix, -1, -2))
          np.testing.assert_array_equal(actual, expected)
          np.testing.assert_array_equal(actual_m, jnp.swapaxes(expected_m, -1, -2))
          np.testing.assert_array_equal(expected_m, matrix)
          def loss(module, p, x, m, transpose):
            result, _ = apply(module, p, x, jnp.swapaxes(m, -1, -2) if transpose else m)
            return jnp.mean(result ** 2)
          reference_grad = jax.jit(jax.grad(
              lambda p, x, m: loss(original, p, x, m, False), argnums=(0, 1, 2)))(
                  parameters, inputs, matrix)
          actual_grad = jax.jit(jax.grad(
              lambda p, x, m: loss(bridge, p, x, m, True), argnums=(0, 1, 2)))(
                  parameters, inputs, matrix)
          for a, b in zip(jax.tree.leaves(reference_grad), jax.tree.leaves(actual_grad)):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)

  def test_forced_native_controller_matches_original_and_heads_are_independent(self):
    cfg = self.tiny(ONE)
    native_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(), dense_conn=True)))
    h, e, l0 = [jax.random.normal(jax.random.key(i + 5), (1, 4, 12)) for i in range(3)]
    for targets in (3, 4):
      original = mudd.Compose(native_cfg, None, C=targets, compose=True)
      forced = mudd.Compose(cfg, None, C=targets, compose=True, force_router=True,
                            hidden_expand_override=1)
      sources = [e, l0, h]
      params = nn.unbox(original.init(jax.random.key(10), h, sources)['params'])
      forced_params = nn.unbox(forced.init(jax.random.key(10), h, sources)['params'])
      for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(forced_params)):
        np.testing.assert_array_equal(a, b)
      identity, _ = forced.apply({'params': params}, h, sources)
      for routed in identity:
        np.testing.assert_array_equal(routed, h)
      params['mlp']['dynamic_dense_conn2']['kernel'] = .03 * jax.random.normal(
          jax.random.key(11), (64, targets, 3))
      def loss(module, p):
        values, _ = module.apply({'params': p}, h, sources)
        return sum((i + 1) * jnp.mean(v ** 2) for i, v in enumerate(values))
      a, ga = jax.value_and_grad(lambda p: loss(original, p))(params)
      b, gb = jax.value_and_grad(lambda p: loss(forced, p))(params)
      np.testing.assert_array_equal(a, b)
      for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
        np.testing.assert_array_equal(x, y)
      self.assertGreater(float(jnp.linalg.norm(gb['mlp']['dynamic_dense_conn1']['kernel'])), 0.)
      before, _ = forced.apply({'params': params}, h, sources)
      params['mlp']['dense_proj2.bias'] = params['mlp']['dense_proj2.bias'].at[0, 0].add(.3)
      after, _ = forced.apply({'params': params}, h, sources)
      self.assertGreater(float(jnp.linalg.norm(before[0] - after[0])), 0.)
      for x, y in zip(before[1:], after[1:]):
        np.testing.assert_array_equal(x, y)

  def build(self, exp):
    cfg = self.tiny(exp)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.asarray([[1, 4, 8, 2]], jnp.int32)
    positions = jnp.arange(4)[None]
    segments = jnp.ones_like(tokens)
    rng = {name: jax.random.key(i + 21) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(model.init(rng, tokens, positions, tokens, segments,
                                   decoder_segment_ids=segments, enable_dropout=False)['params'])
    return cfg, mesh, model, params, tokens, positions, segments, rng

  def test_adding_m_costs195_and_preserves_initialized_model(self):
    base = self.build(QKV)
    full = self.build(ONE)
    cfg, mesh, model, params, t, pos, seg, rng = base
    _, _, next_model, next_params, _, _, _, _ = full
    a, b = flatten_dict(params), flatten_dict(next_params)
    self.assertEqual(set(a), set(b))
    self.assertEqual(sum(v.size for v in b.values()) - sum(v.size for v in a.values()), 195)
    for p in a:
      v = b[p]
      if 'qkvm_compose' in p and 'dynamic_dense_conn2' in p:
        v = v[:, :, :3, :]
      elif 'qkvm_compose' in p and p[-1] == 'dense_proj2.bias':
        v = v[:3, :, :]
      np.testing.assert_array_equal(a[p], v, err_msg='/'.join(p))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def apply(m, p):
        return m.apply({'params': p}, t, pos, t, seg, decoder_segment_ids=seg,
                       enable_dropout=False, rngs=rng)[0]
      np.testing.assert_allclose(apply(model, params), apply(next_model, next_params),
                                 rtol=1e-6, atol=1e-6)

  def test_last_two_scan_causality_carry_and_all_routes_gradients(self):
    cfg, mesh, model, params, tokens, pos, seg, rng = self.build(TWO)
    flat = flatten_dict(params)
    for path, value in flat.items():
      if 'dynamic_dense_conn2' in path:
        flat[path] = value + .01 * jax.random.normal(jax.random.key(44), value.shape)
    params = unflatten_dict(flat)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p, t=tokens, capture=False):
        return model.apply({'params': p}, t, pos, tokens, seg,
            decoder_segment_ids=seg, enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(module, fusion.FusionDecoderLayer))
                                  if capture else False,
            mutable=['intermediates'] if capture else False)
      output, capture = forward(params, capture=True)
      stages = capture['intermediates']['decoder']
      l0 = stages['layers']['local_0']['__call__'][0][0][3][0]
      for stage in stages.values():
        if not isinstance(stage, dict):
          continue
        for name in ('local_0', 'local_1', 'fetch_2'):
          if name not in stage:
            continue
          carry = stage[name]['__call__'][0][0]
          np.testing.assert_allclose(carry[3], jnp.broadcast_to(l0, carry[3].shape),
                                     rtol=2e-6, atol=2e-6)
      matrices = [stages['tail_layers'][name]['__call__'][0][0][1]
                  for name in ('local_0', 'local_1', 'fetch_2')]
      np.testing.assert_array_equal(matrices[0], matrices[1])
      np.testing.assert_array_equal(matrices[1], matrices[2])
      future = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, future)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0])))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      for layer in ('local_1', 'fetch_2'):
        for name in ('W_R_up', 'static_o_key', 'M_shift_gate', 'dynamic_dense_conn1', 'dynamic_dense_conn2'):
          selected = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p and name in p]
          self.assertTrue(selected, (layer, name))
          self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., (layer, name))
        head = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p
                and 'dynamic_dense_conn2' in p][0]
        for i in range(4):
          self.assertGreater(float(jnp.sum(head[:, :, i, :] ** 2)), 0., (layer, i))


if __name__ == '__main__':
  unittest.main()

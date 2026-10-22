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
from layers import bam_mudd, fusion, mudd
from layers.models import Transformer
import max_utils
import pyconfig
import train_compile


QKV = 'BamMediumPropMShiftReadG050MuddQKVSharedRouterLastBlockTruePile'
ONE = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockTruePile'
L2 = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile'
TWO = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersTruePile'
SWAP = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockSwapMLPTruePile'
REBALANCE = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastTwoLayersRebalanceMLPTruePile'
VM = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastTwoLayersTruePile'
VM_L2 = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile'
DUAL_QKVM = 'BamMediumPropMShiftReadG050MuddQKVMDualValueSharedRouterLastBlockTruePile'
DUAL_VM = 'BamMediumPropMShiftReadG050MuddVMDualValueSharedRouterLastBlockTruePile'
QKVM_MLP_WRITE = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockMLPWriteTruePile'
VM_ONE = 'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockTruePile'
MIXED_L2 = 'BamMediumPropMShiftReadG050MuddVMQKVMSharedRouterLastTwoLayersL2AnchorTruePile'
QKVM_L2_WRITE = 'BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorMLPWriteTruePile'


class NativeTailTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, experiment=ONE):
    cfg = tail_reference.MuddTailTest.tiny(self, experiment)
    cfg.get_keys()['bam_pallas_core'] = False  # CPU reference; formal trees retain Pallas.
    return cfg

  def test_formal_budget_joint_layout_and_router_exemption(self):
    self.check_formal_budget((QKV, ONE, TWO, SWAP, REBALANCE, L2, VM))

  def test_vm_l2_formal_budget_joint_layout_and_router_exemption(self):
    self.check_formal_budget((VM_L2,))

  def check_formal_budget(self, experiments):
    for exp, total, router_count, first_width, middle_width, last_width, targets in (
        (QKV, 432266060, 154380, 3802, 3803, 3026, 3),
        (ONE, 432266255, 154575, 3802, 3803, 3026, 4),
        (TWO, 432344635, 232155, 3802, 3027, 3026, 4),
        (SWAP, 432266255, 154575, 3802, 3026, 3803, 4),
        (REBALANCE, 432344635, 232155, 3027, 3414, 3414, 4),
        (L2, 432266580, 154900, 3802, 3803, 3026, 4),
        (VM, 432345167, 231375, 3802, 3414, 3540, 2),
        (VM_L2, 432267116, 154380, 3802, 3676, 3540, 2)):
      if exp not in experiments:
        continue
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
        two_layers = exp in (TWO, REBALANCE, VM)
        self.assertEqual(total - router_count - 432111680,
                         2112 if exp == VM else 1056 if exp == VM_L2 else 800 if two_layers else 0)
        prefix = ('params', 'decoder', 'tail_layers')
        layers = ('local_1', 'fetch_2') if two_layers else ('fetch_2',)
        sources = 4 if exp in (L2, VM_L2) else 3
        for layer in layers:
          p = prefix + (layer, 'block', 'qkvm_compose', 'mlp')
          self.assertEqual(flat[p + ('dynamic_dense_conn1', 'kernel')].shape, (1200, 1, 64))
          self.assertEqual(flat[p + ('dynamic_dense_conn2', 'kernel')].shape, (64, 1, targets, sources))
          self.assertEqual(flat[p + ('dense_proj2.bias',)].shape, (targets, 1, sources))
          attention = prefix + (layer, 'block', 'self_attention')
          for name in ('W_R_up', 'static_o_key', 'M_shift_gate'):
            self.assertTrue(any(p[:len(attention)] == attention and name in p for p in flat))
          self.assertFalse(any(p[:len(attention)] == attention and
                               any(n in p for n in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0'))
                               for p in flat))
        self.assertEqual(flat[prefix + ('local_0', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                         (1200, 1, first_width))
        self.assertEqual(flat[prefix + ('local_1', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                         (1200, 1, middle_width))
        self.assertEqual(flat[prefix + ('fetch_2', 'block', 'mlp', 'wi_0', 'kernel')].shape,
                         (1200, 1, last_width))
        self.assertFalse(any(n in p for p in flat for n in
                             ('value_compose', 'q_compose', 'k_compose', 'm_compose')))
        if exp in (VM, VM_L2):
          for layer in ('local_1', 'fetch_2'):
            block = prefix + (layer, 'block')
            for name in ('query', 'key'):
              self.assertEqual(flat[block + ('self_attention', name, 'kernel')].shape,
                               (1200, 1, 16, 18))
            self.assertFalse(any(p[:len(block)] == block and 'q_input_norm' in p for p in flat))
            self.assertFalse(any(p[:len(block)] == block and 'k_input_norm' in p for p in flat))
          mlp_write = {p: v for p, v in flat.items() if p[:len(prefix)] == prefix and
                       any(n in p for n in ('mlp_write_gate', 'mlp_write_gate_bias',
                                            'mlp_address_down', 'mlp_address_up'))}
          self.assertEqual(sum(math.prod(v.shape) for v in mlp_write.values()), 458000)
          self.assertTrue(all('local_1' in p for p in mlp_write))
          middle_attention = prefix + ('local_1', 'block', 'self_attention')
          for name in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0'):
            self.assertEqual(any(p[:len(middle_attention)] == middle_attention and name in p
                                 for p in flat), exp == VM_L2, name)

  def test_v8t_tail_layout_bridge_preserves_parameters_outputs_and_gradients(self):
    self.check_v8t_tail_layout_bridge(((QKV, 8), (ONE, 8), (TWO, 7), (TWO, 8),
        (VM, 7), (VM, 8), (VM_L2, 8), (DUAL_QKVM, 8), (DUAL_VM, 8)))

  def test_mixed_tail_v8t_layout_bridge(self):
    self.check_v8t_tail_layout_bridge(((MIXED_L2, 7), (MIXED_L2, 8)))

  def check_v8t_tail_layout_bridge(self, experiments):
    for exp, index in experiments:
      with self.subTest(exp=exp, index=index):
        cfg = self.tiny(exp)
        accelerated = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
            cfg.get_keys(), bam_pallas_core=True, bam_pallas_body='v8t')))
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        original = fusion.SubDecoderLayer(cfg, mesh, layer_inx=index)
        bridge = fusion.SubDecoderLayer(accelerated, mesh, layer_inx=index)
        inputs = tuple(jax.random.normal(jax.random.key(i + 70), (1, 4, 150))
                       for i in range(4 if cfg.bam_mudd_tail_l2_anchor else 3))
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
          writes = exp in (VM, MIXED_L2) and index == 7
          if writes:
            self.assertGreater(float(jnp.sum((expected_m - matrix) ** 2)), 0.)
          else:
            np.testing.assert_array_equal(expected_m, matrix)
          def loss(module, p, x, m, transpose):
            result, next_m = apply(module, p, x, jnp.swapaxes(m, -1, -2) if transpose else m)
            return jnp.mean(result ** 2) + (.01 * jnp.mean(next_m ** 2) if writes else 0.)
          reference_grad = jax.jit(jax.grad(
              lambda p, x, m: loss(original, p, x, m, False), argnums=(0, 1, 2)))(
                  parameters, inputs, matrix)
          actual_grad = jax.jit(jax.grad(
              lambda p, x, m: loss(bridge, p, x, m, True), argnums=(0, 1, 2)))(
                  parameters, inputs, matrix)
          for a, b in zip(jax.tree.leaves(reference_grad), jax.tree.leaves(actual_grad)):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)

  def test_mixed_tail_formal_budget_and_live_writers(self):
    cfg = self.config(MIXED_L2, bam_splash_attention=False)
    cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                          query_chunk_size=256)
    self.assertEqual([i for i in range(18) if bam_mudd.is_value_layer(cfg, i)], [16, 17])
    self.assertEqual([i for i in range(18) if bam_mudd.is_qk_layer(cfg, i)], [17])
    self.assertEqual([i for i in range(18) if bam_mudd.skips_attention_write(cfg, i)], [17])
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    state, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
    flat = flatten_dict(nn.unbox(state[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 432346556)
    routers = sum(math.prod(v.shape) for p, v in flat.items()
                  if 'qkvm_compose' in p or 'compose_final_three_source' in p)
    self.assertEqual(routers, 232220)
    self.assertEqual(432346556 - routers - 432111680, 2656)
    prefix = ('params', 'decoder', 'tail_layers')
    for layer, width in (('local_0', 3802), ('local_1', 3287), ('fetch_2', 3026)):
      self.assertEqual(flat[prefix + (layer, 'block', 'mlp', 'wi_0', 'kernel')].shape,
                       (1200, 1, width))
    for layer, targets, q_width in (('local_1', 2, 18), ('fetch_2', 4, 75)):
      block = prefix + (layer, 'block')
      for name in ('query', 'key'):
        self.assertEqual(flat[block + ('self_attention', name, 'kernel')].shape,
                         (1200, 1, 16, q_width))
      self.assertEqual(flat[block + ('qkvm_compose', 'mlp', 'dynamic_dense_conn2', 'kernel')].shape,
                       (64, 1, targets, 4))
      for name in ('W_R_up', 'static_o_key'):
        self.assertTrue(any(p[:len(block)] == block and name in p for p in flat))
      attention_write = {p: v for p, v in flat.items() if p[:len(block)] == block and
                         any(n in p for n in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0'))}
      mlp_write = {p: v for p, v in flat.items() if p[:len(block)] == block and
                   any(n in p for n in ('mlp_write_gate', 'mlp_write_gate_bias',
                                        'mlp_address_down', 'mlp_address_up'))}
      self.assertEqual(sum(math.prod(v.shape) for v in attention_write.values()),
                       458000 if layer == 'local_1' else 0)
      self.assertEqual(sum(math.prod(v.shape) for v in mlp_write.values()),
                       458000 if layer == 'local_1' else 0)

  def test_mixed_tail_writes_reach_readout_and_causal_gradients(self):
    cfg, mesh, model, params, tokens, pos, seg, rng = self.build(MIXED_L2)
    flat = flatten_dict(params)
    for path, value in flat.items():
      if 'dynamic_dense_conn2' in path:
        flat[path] = value + .01 * jax.random.normal(jax.random.key(91), value.shape)
    params = unflatten_dict(flat)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p, t=tokens, capture=False):
        return model.apply({'params': p}, t, pos, tokens, seg,
            decoder_segment_ids=seg, enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(module, fusion.FusionDecoderLayer))
                                  if capture else False,
            mutable=['intermediates'] if capture else False)
      output, capture = forward(params, capture=True)
      tail = capture['intermediates']['decoder']['tail_layers']
      matrices = [tail[name]['__call__'][0][0][1]
                  for name in ('local_0', 'local_1', 'fetch_2')]
      self.assertGreater(float(jnp.sum((matrices[1] - matrices[0]) ** 2)), 0.)
      np.testing.assert_array_equal(matrices[1], matrices[2])
      for gate in ('gw_b0', 'mlp_write_gate_bias'):
        ablated = dict(flat)
        selected = [p for p in flat if 'tail_layers' in p and 'local_1' in p and gate in p]
        self.assertTrue(selected, gate)
        for path in selected:
          ablated[path] = jnp.full_like(flat[path], -100.)
        ablated_output, ablated_capture = forward(unflatten_dict(ablated), capture=True)
        ablated_m = ablated_capture['intermediates']['decoder']['tail_layers']['local_1']['__call__'][0][0][1]
        self.assertGreater(float(jnp.sum((ablated_m - matrices[1]) ** 2)), 0., gate)
        self.assertGreater(float(jnp.sum((ablated_output[0] - output[0]) ** 2)), 0., gate)
      future = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, future)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0] ** 2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      for name in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0', 'mlp_address_down',
                   'mlp_address_up', 'mlp_write_gate', 'mlp_write_gate_bias'):
        selected = [v for p, v in gradients.items() if 'tail_layers' in p and 'local_1' in p and name in p]
        self.assertTrue(selected, name)
        self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)
      for layer, targets in (('local_1', 2), ('fetch_2', 4)):
        heads = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p
                 and 'qkvm_compose' in p and 'dynamic_dense_conn2' in p][0]
        self.assertEqual(heads.shape[-2:], (targets, 4))
        for i in range(targets):
          self.assertGreater(float(jnp.sum(heads[:, :, i, 2] ** 2)), 0., (layer, i, 'L2'))

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

  def test_single_layer_write_controls_formal_budgets(self):
    self.check_single_layer_write_controls((QKVM_MLP_WRITE, VM_ONE))

  def test_l2_qkvm_write_control_formal_budget(self):
    self.check_single_layer_write_controls((QKVM_L2_WRITE,))

  def check_single_layer_write_controls(self, experiments):
    for exp, total, routers, width, q_width, rounding, l2 in (
        (QKVM_MLP_WRITE, 432267055, 154575, 3026, 75, 800, False),
        (VM_ONE, 432266921, 154185, 3540, 18, 1056, False),
        (QKVM_L2_WRITE, 432267380, 154900, 3026, 75, 800, True)):
      if exp not in experiments:
        continue
      with self.subTest(exp=exp):
        cfg = self.config(exp, bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                              query_chunk_size=256)
        self.assertEqual(cfg.bam_mudd_tail_l2_anchor, l2)
        self.assertFalse(cfg.bam_mudd_tail_dual_value)
        self.assertFalse(cfg.bam_mudd_tail_skip_mlp_write)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        state, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
        flat = flatten_dict(nn.unbox(state[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
        self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items()
                             if 'qkvm_compose' in p or 'compose_final_three_source' in p), routers)
        self.assertEqual(total - routers - 432111680, rounding)
        prefix = ('params', 'decoder', 'tail_layers')
        for layer, expected in (('local_0', 3802), ('local_1', 3676), ('fetch_2', width)):
          self.assertEqual(flat[prefix + (layer, 'block', 'mlp', 'wi_0', 'kernel')].shape,
                           (1200, 1, expected))
        for name in ('query', 'key'):
          self.assertEqual(flat[prefix + ('fetch_2', 'block', 'self_attention', name, 'kernel')].shape,
                           (1200, 1, 16, q_width))
        mlp_write = {p: v for p, v in flat.items() if p[:len(prefix)] == prefix and
                     any(n in p for n in ('mlp_write_gate', 'mlp_write_gate_bias',
                                          'mlp_address_down', 'mlp_address_up'))}
        self.assertEqual(sum(math.prod(v.shape) for v in mlp_write.values()), 458000)
        self.assertTrue(all('local_1' in p for p in mlp_write))
        middle = prefix + ('local_1', 'block', 'self_attention')
        tail = prefix + ('fetch_2', 'block', 'self_attention')
        writes = ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0')
        for name in writes:
          self.assertTrue(any(p[:len(middle)] == middle and name in p for p in flat), name)
          self.assertFalse(any(p[:len(tail)] == tail and name in p for p in flat), name)

  def test_dual_value_formal_budget_reuses_vo_projection(self):
    for exp, total, routers, width in ((DUAL_QKVM, 432266399, 154575, 3015),
                                      (DUAL_VM, 432266265, 154185, 3529)):
      with self.subTest(exp=exp):
        cfg = self.config(exp, bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                              query_chunk_size=256)
        self.assertFalse(cfg.bam_mudd_tail_l2_anchor)
        self.assertFalse(cfg.bam_mudd_tail_skip_mlp_write)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        state, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
        flat = flatten_dict(nn.unbox(state[0].params))
        parent_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
            cfg.get_keys(), bam_mudd_tail_dual_value=False)))
        state, _, _, _ = train_compile.get_shaped_inputs(mesh, parent_cfg)
        parent = flatten_dict(nn.unbox(state[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
        self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items()
                             if 'qkvm_compose' in p or 'compose_final_three_source' in p), routers)
        extra = set(flat) - set(parent)
        self.assertEqual(sum(math.prod(flat[p].shape) for p in extra), 38944)
        self.assertEqual({p[-2] if p[-1] == 'kernel' else p[-1] for p in extra},
                         {'W_lv_gate', 'W_lv_gate_b0', 'static_v_key',
                          'general_v_static_gate', 'general_v_static_gate_bias'})
        for p in extra:
          self.assertIn('fetch_2', p)
          self.assertIn('self_attention', p)
        self.assertFalse(set(parent) - set(flat))
        for p in parent:
          self.assertEqual(parent[p].shape, flat[p].shape, p)
        prefix = ('params', 'decoder', 'tail_layers')
        for layer, expected in (('local_0', 3802), ('local_1', 3676), ('fetch_2', width)):
          self.assertEqual(flat[prefix + (layer, 'block', 'mlp', 'wi_0', 'kernel')].shape,
                           (1200, 1, expected))
        mlp_write = {p: v for p, v in flat.items() if p[:len(prefix)] == prefix and
                     any(n in p for n in ('mlp_write_gate', 'mlp_write_gate_bias',
                                          'mlp_address_down', 'mlp_address_up'))}
        self.assertEqual(sum(math.prod(v.shape) for v in mlp_write.values()), 458000)
        self.assertTrue(all('local_1' in p for p in mlp_write))

  def test_dual_value_isolation_causality_and_gradients(self):
    for exp in (DUAL_QKVM, DUAL_VM):
      with self.subTest(exp=exp):
        cfg, mesh, model, params, tokens, pos, seg, rng = self.build(exp)
        flat = flatten_dict(params)
        for path, value in flat.items():
          if 'dynamic_dense_conn2' in path:
            flat[path] = value + .01 * jax.random.normal(jax.random.key(88), value.shape)
        params = unflatten_dict(flat)
        reference_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
            cfg.get_keys(), bam_mudd_tail_dual_value=False)))
        reference = Transformer(reference_cfg, mesh, quant=None)
        with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
          def forward(p, t=tokens, module=model):
            return module.apply({'params': p}, t, pos, tokens, seg,
                decoder_segment_ids=seg, enable_dropout=False, rngs=rng)[0]
          output = forward(params)
          future = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
          np.testing.assert_allclose(forward(params, future)[:, :2], output[:, :2],
                                     rtol=3e-5, atol=3e-5)
          disabled = dict(flat)
          for path, value in disabled.items():
            if 'tail_layers' in path and 'fetch_2' in path:
              if path[-1] == 'W_lv_gate_b0':
                disabled[path] = jnp.full_like(value, -100.)
              elif path[-1] == 'static_v_key':
                disabled[path] = jnp.zeros_like(value)
          disabled = unflatten_dict(disabled)
          np.testing.assert_allclose(forward(disabled), forward(disabled, module=reference),
                                     rtol=3e-5, atol=3e-5)
          self.assertGreater(float(jnp.sum((output - forward(disabled)) ** 2)), 0.)
          loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p) ** 2)))(params)
          self.assertTrue(np.isfinite(float(loss)))
          self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
          gradients = flatten_dict(grad)
          for name in ('W_lv_gate', 'W_lv_gate_b0', 'static_v_key',
                       'general_v_static_gate', 'general_v_static_gate_bias', 'W_R_up'):
            selected = [v for p, v in gradients.items() if 'tail_layers' in p
                        and 'fetch_2' in p and name in p]
            self.assertTrue(selected, name)
            self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)
          for name in ('mlp_write_gate', 'mlp_address_down', 'mlp_address_up'):
            selected = [v for p, v in gradients.items() if 'tail_layers' in p
                        and 'local_1' in p and name in p]
            self.assertTrue(selected, name)
            self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)

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

  def test_vm_restores_qk_reads_and_mlp_only_write_with_causal_gradients(self):
    self.check_vm_causal_gradients(VM)

  def test_vm_l2_restores_last_qk_reads_and_mlp_write_with_causal_gradients(self):
    self.check_vm_causal_gradients(VM_L2)

  def check_vm_causal_gradients(self, experiment):
    cfg, mesh, model, params, tokens, pos, seg, rng = self.build(experiment)
    flat = flatten_dict(params)
    for path, value in flat.items():
      if 'dynamic_dense_conn2' in path:
        flat[path] = value + .01 * jax.random.normal(jax.random.key(48), value.shape)
    params = unflatten_dict(flat)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p, t=tokens, capture=False):
        return model.apply({'params': p}, t, pos, tokens, seg,
            decoder_segment_ids=seg, enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(module, fusion.FusionDecoderLayer))
                                  if capture else False,
            mutable=['intermediates'] if capture else False)
      output, capture = forward(params, capture=True)
      tail = capture['intermediates']['decoder']['tail_layers']
      matrices = [tail[name]['__call__'][0][0][1]
                  for name in ('local_0', 'local_1', 'fetch_2')]
      self.assertGreater(float(jnp.sum((matrices[1] - matrices[0]) ** 2)), 0.)
      np.testing.assert_array_equal(matrices[1], matrices[2])
      future = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, future)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0] ** 2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      for name in ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate', 'mlp_write_gate_bias'):
        selected = [v for p, v in gradients.items() if 'tail_layers' in p and 'local_1' in p and name in p]
        self.assertTrue(selected, name)
        self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)
      for layer in ('local_1', 'fetch_2'):
        for name in ('static_q_key', 'static_k_key', 'static_o_key'):
          selected = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p and name in p]
          self.assertTrue(selected, (layer, name))
          self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., (layer, name))
        if experiment == VM_L2 and layer == 'local_1':
          for name in ('P_loc_down', 'P_loc_up', 'W_gw', 'gw_b0'):
            selected = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p and name in p]
            self.assertTrue(selected, name)
            self.assertGreater(sum(float(jnp.sum(v * v)) for v in selected), 0., name)
          continue
        down = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p
                and 'qkvm_compose' in p and 'dynamic_dense_conn1' in p]
        self.assertGreater(sum(float(jnp.sum(v * v)) for v in down), 0., layer)
        heads = [v for p, v in gradients.items() if 'tail_layers' in p and layer in p
                 and 'qkvm_compose' in p and 'dynamic_dense_conn2' in p][0]
        self.assertEqual(heads.shape[-2:], (2, 4 if cfg.bam_mudd_tail_l2_anchor else 3))
        for i in range(2):
          self.assertGreater(float(jnp.sum(heads[:, :, i, :] ** 2)), 0., (layer, i))
          if cfg.bam_mudd_tail_l2_anchor:
            self.assertGreater(float(jnp.sum(heads[:, :, i, 2] ** 2)), 0., (layer, i, 'L2'))

  def test_l2_anchor_costs325_and_preserves_initialized_model(self):
    base = self.build(ONE)
    full = self.build(L2)
    cfg, mesh, model, params, t, pos, seg, rng = base
    _, _, next_model, next_params, _, _, _, _ = full
    a, b = flatten_dict(params), flatten_dict(next_params)
    self.assertEqual(set(a), set(b))
    self.assertEqual(sum(v.size for v in b.values()) - sum(v.size for v in a.values()), 325)
    for path in a:
      value = b[path]
      if ('qkvm_compose' in path or 'compose_final_three_source' in path) and (
          'dynamic_dense_conn2' in path or path[-1] == 'dense_proj2.bias'):
        value = jnp.take(value, jnp.asarray([0, 1, 3]), axis=-1)
      np.testing.assert_array_equal(a[path], value, err_msg='/'.join(path))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def apply(module, p):
        return module.apply({'params': p}, t, pos, t, seg, decoder_segment_ids=seg,
                            enable_dropout=False, rngs=rng)[0]
      np.testing.assert_allclose(apply(model, params), apply(next_model, next_params),
                                 rtol=1e-6, atol=1e-6)

  def test_l2_complete_residual_reaches_all_routes_and_backpropagates(self):
    cfg, mesh, model, params, tokens, pos, seg, rng = self.build(L2)
    flat = flatten_dict(params)
    # Select only L2 at both native sites. This isolates the cached anchor's
    # forward and gradient path from the normal current-residual connection.
    for path, value in flat.items():
      if ('qkvm_compose' in path or 'compose_final_three_source' in path) and path[-1] == 'dense_proj2.bias':
        flat[path] = jnp.zeros_like(value).at[..., 2].set(1.)
    params = unflatten_dict(flat)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p, t=tokens, capture=False):
        return model.apply({'params': p}, t, pos, tokens, seg,
            decoder_segment_ids=seg, enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(
                module, (fusion.FusionDecoderLayer, mudd.Compose))) if capture else False,
            mutable=['intermediates'] if capture else False)
      output, capture = forward(params, capture=True)
      stages = capture['intermediates']['decoder']
      first_fetch = stages['layers']['fetch_2']['__call__'][0][0]
      l2 = first_fetch[0][0]  # Complete first block, including L2 attention and MLP.
      np.testing.assert_array_equal(first_fetch[4], jnp.broadcast_to(l2, first_fetch[4].shape))
      for name in ('local_0', 'local_1'):
        anchors = stages['layers'][name]['__call__'][0][0][4]
        np.testing.assert_array_equal(anchors[0], jnp.zeros_like(l2))
        np.testing.assert_array_equal(anchors[1:], jnp.broadcast_to(l2, anchors[1:].shape))
      for name in ('local_0', 'local_1', 'fetch_2'):
        anchor = stages['tail_layers'][name]['__call__'][0][0][4]
        np.testing.assert_array_equal(anchor, jnp.broadcast_to(l2, anchor.shape))
      routes = stages['tail_layers']['fetch_2']['block']['qkvm_compose']['__call__'][0][0]
      self.assertEqual(len(routes), 4)
      for route in routes:
        np.testing.assert_array_equal(route, jnp.broadcast_to(l2, route.shape))
      future = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, future)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0] ** 2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      path = ('decoder', 'layers', 'fetch_2', 'block', 'mlp', 'wo', 'kernel')
      self.assertGreater(float(jnp.sum(gradients[path][:, 0, :] ** 2)), 0.)
      np.testing.assert_array_equal(gradients[path][:, 1:, :], 0.)
      for path, value in gradients.items():
        if 'tail_layers' in path:
          np.testing.assert_array_equal(value, 0., err_msg='/'.join(path))


if __name__ == '__main__':
  unittest.main()

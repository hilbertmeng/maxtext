"""Selective Mudd value paths: exact budgets, two-stage scan, causality and gradients."""

import math
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from bam_prop_test_utils import MLPWriteTest
import max_utils
import train_compile
from layers import bam_mudd, fusion
from layers.models import Transformer


PARENT = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
DOUBLE = 'BamMediumPropMShiftReadG050MuddValueLastTwoBlocksTruePile'
SINGLE = 'BamMediumPropMShiftReadG050MuddValueLastBlockTruePile'
VM = 'BamMediumPropMShiftReadG050MuddVMLastBlockTruePile'
QKV = 'BamMediumPropMShiftReadG050MuddQKVLastBlockTruePile'


class MuddTailTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budget_and_scan_parameter_layout(self):
    for exp, total, prefix, tail, width in (
        (PARENT, 432111680, 6, 0, 3802),
        (DOUBLE, 432114377, 4, 2, 3381),
        (SINGLE, 432111926, 5, 1, 3370),
        (VM, 432113321, 5, 1, 3349),
        (QKV, 432110860, 5, 1, 2813)):
      with self.subTest(exp=exp):
        cfg = self.config(exp, bam_splash_attention=False)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        shaped, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
        params = flatten_dict(nn.unbox(shaped[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in params.values()), total)
        self.assertEqual(params[('params', 'decoder', 'layers', 'local_0', 'block',
                                 'mlp', 'wi_0', 'kernel')].shape, (1200, prefix, 3802))
        if not tail:
          continue
        prefix_path = ('params', 'decoder', 'layers', 'local_1', 'block')
        tail_path = ('params', 'decoder', 'tail_layers')
        v_path = tail_path + ('fetch_2', 'block')
        self.assertEqual(params[v_path + ('mlp', 'wi_0', 'kernel')].shape, (1200, tail, width))
        self.assertEqual(params[v_path + ('self_attention', 'value', 'kernel')].shape,
                         (1200, tail, 16, 75))
        self.assertEqual(params[v_path + ('value_compose', 'dynamic_dense_conn1', 'kernel')].shape,
                         (1200, tail, 64))
        self.assertEqual(params[('params', 'decoder', 'compose_final_three_source',
                                 'dynamic_dense_conn1', 'kernel')].shape, (1200, 64))
        self.assertTrue(any(path[:len(prefix_path)] == prefix_path and 'mlp_address_down' in path
                            for path in params))
        for path in params:
          if path[:len(tail_path)] == tail_path:
            self.assertFalse(any(x in path for x in (
                'mlp_address_down', 'mlp_address_up', 'mlp_write_gate', 'mlp_write_gate_bias')))
          if path[:len(v_path)] == v_path:
            self.assertFalse(any(x in path for x in (
                'W_lv_gate', 'W_lv_gate_b0', 'static_v_key',
                'general_v_static_gate', 'general_v_static_gate_bias')))
        self.assertTrue(any(path[:len(v_path)] == v_path and 'W_R_up' in path for path in params))
        self.assertTrue(any(path[:len(v_path)] == v_path and 'static_o_key' in path for path in params))
        if exp == VM:
          self.assertEqual(params[v_path + ('m_compose', 'dynamic_dense_conn1', 'kernel')].shape,
                           (1200, 1, 64))
          self.assertEqual([i for i in range(18) if bam_mudd.is_mlp_route_layer(cfg, i)], [17])
          self.assertFalse(cfg.bam_mudd_tail_residual_qk)
        else:
          self.assertFalse(any('m_compose' in path for path in params))
        if exp == QKV:
          for arm in ('q', 'k'):
            name = 'query' if arm == 'q' else 'key'
            self.assertEqual(params[v_path + ('self_attention', name, 'kernel')].shape,
                             (1200, 1, 16, 75))
            self.assertEqual(params[v_path + (arm + '_compose', 'dynamic_dense_conn1', 'kernel')].shape,
                             (1200, 1, 64))
            for path in params:
              if path[:len(v_path)] == v_path:
                self.assertFalse(any(x in path for x in (
                    'W_l' + arm + '_c8_up', 'W_l' + arm + '_gate',
                    'static_' + arm + '_key', 'general_' + arm + '_pre_bias',
                    'general_' + arm + '_static_gate')))
        self.assertFalse(cfg.qk_norm)
        self.assertFalse(cfg.bam_share_read_attention_write_down)

  def test_router_mudd_initialization_rounding_and_signed_anchor_gradients(self):
    cfg = self.config(PARENT, dtype='float32', weight_dtype='float32')
    current, embedding, l0 = [jax.random.normal(jax.random.key(i), (1, 4, 12))
                              for i in range(3)]
    for expand in (1, 4):
      router = bam_mudd.ThreeSourceRouter(cfg, None, hidden_expand=expand)
      params = nn.unbox(router.init(jax.random.key(5), current, (embedding, l0))['params'])
      self.assertEqual(params['dynamic_dense_conn1']['kernel'].shape, (12, 64))
      np.testing.assert_array_equal(router.apply({'params': params}, current, (embedding, l0)), current)
      self.assertNotIn('pre_dense_proj1_norm', params)
      params['mix_bias'] = jnp.asarray([1., .2, -.3])
      actual = router.apply({'params': params}, current, (embedding, l0))
      np.testing.assert_allclose(actual, current + .2 * embedding - .3 * l0, rtol=2e-6, atol=2e-6)
      grads = jax.grad(lambda e, h: jnp.sum(router.apply({'params': params}, current, (e, h))),
                       argnums=(0, 1))(embedding, l0)
      np.testing.assert_allclose(grads[0], .2, rtol=1e-6)
      np.testing.assert_allclose(grads[1], -.3, rtol=1e-6)
      grad = jax.grad(lambda p: jnp.mean(router.apply({'params': p}, current, (embedding, l0)) ** 2))(params)
      self.assertGreater(float(jnp.sum(grad['dynamic_dense_conn2']['kernel'] ** 2)), 0.)

  def tiny(self, experiment=DOUBLE):
    cfg = self.config(experiment, bam_splash_attention=False, base_emb_dim=150,
                      base_num_query_heads=2, base_num_kv_heads=2,
                      base_num_decoder_layers=9, base_mlp_dim=48, vocab_size=32,
                      dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(
        emb_bam_num_head=2, bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, mlp_dim_by_block=[48, 40, 48],
        bam_mudd_tail_mlp_dims=[48, 48, 40],
        bam_layer_modes=['local_qk+local_v+local_o'] * 9)
    return cfg

  def test_two_stage_scan_l0_anchor_causality_and_active_gradients(self):
    self.check_scan_runtime(DOUBLE)

  def test_residual_qk_causal_scan_and_active_gradients(self):
    self.check_scan_runtime(QKV)

  def test_residual_mlp_causal_scan_and_active_gradients(self):
    self.check_scan_runtime(VM)

  def test_m_route_identity_and_attention_addition_mlp_residual_semantics(self):
    cfg = self.tiny(VM)
    parent_cfg = self.tiny(SINGLE)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    layer = fusion.SubDecoderLayer(cfg, mesh, layer_inx=8)
    parent = fusion.SubDecoderLayer(parent_cfg, mesh, layer_inx=8)
    current, embedding, l0 = [jax.random.normal(jax.random.key(i + 40), (1, 4, 150))
                              for i in range(3)]
    matrix = .2 * jax.random.normal(jax.random.key(45), (1, 4, 75, 32))
    tokens = jnp.ones((1, 4), jnp.int32)
    kwargs = dict(decoder_segment_ids=tokens, decoder_positions=jnp.arange(4)[None],
                  decoder_input_tokens=tokens, deep_embedding=None, deterministic=True,
                  model_mode='train', eos_sum=None, M_in=matrix, layer_index=8,
                  early_residuals=(embedding, l0))
    rngs = {name: jax.random.key(i + 51) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(layer.init(rngs, current, **kwargs)['params'])
      initial, initial_matrix = layer.apply({'params': params}, current, **kwargs)
      baseline, baseline_matrix = parent.apply({'params': params}, current, **kwargs)
      np.testing.assert_allclose(initial, baseline, rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(initial_matrix, baseline_matrix)
      params['m_compose']['mix_bias'] = jnp.asarray([.7, .2, -.1])
      traces = {}

      def trace(next_function, args, call_kwargs, context):
        if context.method_name == '__call__':
          if context.module.name == 'm_compose':
            traces['controller_current'] = args[0]
          elif context.module.name == 'post_self_attention_layer_norm':
            traces['mlp_input'] = args[0]
        result = next_function(*args, **call_kwargs)
        if context.method_name == '__call__':
          if context.module.name == 'self_attention':
            traces['attention'] = result[0]
          elif context.module.name == 'mlp':
            traces['mlp_output'] = result
        return result

      with nn.intercept_methods(trace):
        actual, actual_matrix = layer.apply({'params': params}, current, **kwargs)
      expected_residual = .7 * current + .2 * embedding - .1 * l0 + traces['attention']
      np.testing.assert_array_equal(traces['controller_current'], current)
      np.testing.assert_allclose(traces['mlp_input'], expected_residual, rtol=2e-6, atol=2e-6)
      np.testing.assert_allclose(actual, expected_residual + traces['mlp_output'],
                                 rtol=2e-6, atol=2e-6)
      # Mudd's m route must not change Q/K/V/O or attention write controls.
      np.testing.assert_array_equal(actual_matrix, baseline_matrix)

  def check_scan_runtime(self, experiment):
    cfg = self.tiny(experiment)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.asarray([[1, 4, 8, 2]], jnp.int32)
    positions = jnp.arange(4)[None]
    segments = jnp.ones_like(tokens)
    rng = {name: jax.random.key(i + 11) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(model.init(rng, tokens, positions, tokens, segments,
                                   decoder_segment_ids=segments, enable_dropout=False)['params'])
      flat = flatten_dict(params)
      # Activate the shallow paths so the checks cannot pass merely because routes start at identity.
      for path, value in flat.items():
        if path[-1] == 'mix_bias':
          flat[path] = value + jnp.asarray([-.1, .2, -.1]).reshape((3,) + (1,) * (value.ndim - 1))
        elif 'dynamic_dense_conn2' in path:
          flat[path] = value + .01 * jax.random.normal(jax.random.key(31), value.shape)
      params = unflatten_dict(flat)

      def forward(p, t=tokens, capture=False):
        return model.apply(
            {'params': p}, t, positions, tokens, segments, decoder_segment_ids=segments,
            enable_dropout=False, rngs=rng,
            capture_intermediates=(lambda module, method: isinstance(module, fusion.FusionDecoderLayer))
                                  if capture else False,
            mutable=['intermediates'] if capture else False)

      output, captured = forward(params, capture=True)
      leaves = captured['intermediates']['decoder']
      first = leaves['layers']['local_0']['__call__'][0][0]
      initial_l0 = first[0][0]
      np.testing.assert_allclose(first[3][0], initial_l0, rtol=1e-6, atol=1e-6)
      for stage in ('layers', 'tail_layers'):
        for layer in ('local_0', 'local_1', 'fetch_2'):
          carried = leaves[stage][layer]['__call__'][0][0]
          np.testing.assert_allclose(carried[3], jnp.broadcast_to(initial_l0, carried[3].shape),
                                     rtol=2e-6, atol=2e-6)
      changed = tokens.at[:, 2:].set(jnp.asarray([[9, 10]], jnp.int32))
      np.testing.assert_allclose(forward(params, changed)[0][:, :2], output[0][:, :2],
                                 rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)[0])))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients = flatten_dict(grad)
      for path, value in gradients.items():
        if ((any(name in path for name in ('q_compose', 'k_compose', 'value_compose', 'm_compose', 'compose_final_three_source')))
            and path[-1] == 'kernel'):
          self.assertGreater(float(jnp.sum(value * value)), 0., '/'.join(path))
      if experiment == VM:
        self.assertTrue(any('m_compose' in path for path in gradients))
      for name in (('value', 'query', 'key', 'W_R_up', 'M_shift_gate') if experiment == QKV
                   else ('value', 'W_lq_c8_up', 'W_lk_c8_up', 'W_R_up', 'M_shift_gate')):
        chosen = [v for path, v in gradients.items() if 'tail_layers' in path and name in path]
        self.assertTrue(chosen, name)
        self.assertGreater(sum(float(jnp.sum(x * x)) for x in chosen), 0., name)


if __name__ == '__main__':
  unittest.main()

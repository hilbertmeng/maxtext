"""Formula oracles and end-to-end regressions for Full AttnRes / mHC.

Run with the pinned CPU test runner; no checkpoints or accelerator required.
"""
import copy
from types import SimpleNamespace
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import exp
import common_types
import max_utils
from layers.models import Transformer
from layers.fusion import SubDecoderLayer
from layers.residual_connections import (
    FullAttentionResidual, ManifoldHyperConnection, MHCLearnedStreamReduce, mhc_update, sinkhorn)
from residual_config import validate_residual_config
from tests.bam_prop_test_utils import MLPWriteTest


def numpy_sinkhorn(logits, repeat=20, eps=1e-6):
  # Independent transcription of DeepSeek TileKernels' torch reference.
  matrix = np.exp(logits - logits.max(axis=-1, keepdims=True))
  matrix = matrix / matrix.sum(axis=-1, keepdims=True) + eps
  matrix = matrix / (matrix.sum(axis=-2, keepdims=True) + eps)
  for _ in range(repeat - 1):
    matrix = matrix / (matrix.sum(axis=-1, keepdims=True) + eps)
    matrix = matrix / (matrix.sum(axis=-2, keepdims=True) + eps)
  return matrix


def connection_config(**overrides):
  fields = {k: v for cls in (exp.AttnResConfig, exp.MHCConfig)
            for k, v in vars(cls).items() if not k.startswith('_')}
  return SimpleNamespace(**dict(fields, weight_dtype=jnp.float32, **overrides))


class ResidualFormulaTest(unittest.TestCase):
  def test_full_attention_zero_query_and_single_source(self):
    module = FullAttentionResidual(connection_config())
    sources = tuple(jnp.asarray(x, jnp.float32) for x in ([[[2., -1.]]], [[[9., 3.]]], [[[-5., 7.]]]))
    variables = module.init(jax.random.key(0), sources)
    np.testing.assert_allclose(module.apply(variables, sources), np.mean(sources, axis=0), atol=1e-6)
    self.assertTrue(np.all(np.asarray(variables['params']['query']) == 0))
    self.assertTrue(np.all(np.asarray(variables['params']['key_scale']) == 1))
    single = module.init(jax.random.key(0), sources[:1])
    self.assertNotIn('params', single)
    np.testing.assert_array_equal(module.apply(single, sources[:1]), sources[0])

  def test_full_attention_key_norm_values_and_depth_axis(self):
    cfg = connection_config()
    module = FullAttentionResidual(cfg)
    sources = tuple(jnp.asarray(x) for x in ([[[.1, 2., -3.], [4., 1., 2.]]],
                                             [[[8., -5., 2.], [1., -3., 6.]]]))
    params = {'query': jnp.array([.4, -.7, .8]), 'key_scale': jnp.array([1., .3, 2.])}
    values = np.stack(sources, axis=-2)
    keys = values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + cfg.attnres_norm_epsilon)
    scores = (keys * np.asarray(params['key_scale']) * np.asarray(params['query'])).sum(-1)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = (weights[..., None] * values).sum(-2)
    np.testing.assert_allclose(module.apply({'params': params}, sources), expected, atol=2e-6)
    gradients = jax.grad(lambda s: jnp.square(module.apply({'params': params}, s)).sum())(sources)
    self.assertTrue(all(np.linalg.norm(x) > 0 for x in gradients))

  def test_sinkhorn_matches_official_order_and_constraints(self):
    logits = np.random.default_rng(7).normal(size=(2, 3, 4, 4)).astype('float32')
    actual = sinkhorn(jnp.asarray(logits), 20, 1e-6)
    np.testing.assert_allclose(actual, numpy_sinkhorn(logits), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=4e-6)
    np.testing.assert_allclose(actual.sum(-2), 1, atol=4e-6)
    self.assertTrue(np.isfinite(sinkhorn(jnp.asarray(logits * 1e4), 20, 1e-6)).all())
    grad = jax.grad(lambda x: jnp.sum(sinkhorn(x, 20, 1e-6)**2))(jnp.asarray(logits))
    self.assertTrue(np.isfinite(grad).all())
    self.assertGreater(np.linalg.norm(grad), 0)

  def test_mhc_official_source_destination_orientation(self):
    # Nonsymmetric cyclic permutation catches a transposed mixing matrix.
    state = jnp.array([[[[1., 2.], [3., 4.], [5., 6.]]]])
    mixing = jnp.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    branch = jnp.array([[[10., 20.]]])
    post = jnp.array([.1, .2, .3])
    expected = np.asarray(state)[..., [2, 0, 1], :] + np.asarray(post)[:, None] * np.asarray(branch)[..., None, :]
    np.testing.assert_allclose(mhc_update(state, branch, post, mixing), expected)
    # One stream and identity mixing is the ordinary additive residual.
    np.testing.assert_array_equal(mhc_update(state[..., :1, :], branch, jnp.ones(1), jnp.ones((1, 1)))[..., 0, :],
                                  state[..., 0, :] + branch)

  def test_mhc_router_official_forward_and_gradients(self):
    cfg = connection_config()
    module = ManifoldHyperConnection(cfg)
    state = jnp.asarray(np.random.default_rng(2).normal(size=(1, 2, 4, 3)), jnp.float32)
    params = module.init(jax.random.key(3), state)['params']
    # Probe a learned controller, not the zero-projection initialization where
    # alpha and norm-weight gradients are deliberately zero on the first step.
    params['router_kernel'] = jnp.asarray(np.random.default_rng(5).normal(scale=.2, size=(12, 24)), jnp.float32)
    params['router_scale'] = jnp.array([.3, .2, .5])
    params['norm_scale'] = jnp.linspace(.5, 1.5, 12)

    def oracle(p):
      flat = np.asarray(state).reshape(1, 2, 12)
      # Official code absorbs the affine norm weight into transposed Phi.
      phi = np.asarray(p['router_kernel']).T * np.asarray(p['norm_scale'])
      logits = (flat @ phi.T) / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + cfg.mhc_norm_epsilon)
      alpha = np.repeat(np.asarray(p['router_scale']), [4, 4, 16])
      logits = logits * alpha + np.asarray(p['router_bias'])
      pre = 1 / (1 + np.exp(-logits[..., :4])) + cfg.mhc_pre_epsilon
      post = 2 / (1 + np.exp(-logits[..., 4:8]))
      mix = numpy_sinkhorn(logits[..., 8:].reshape(1, 2, 4, 4))
      read = np.sum(np.asarray(state) * pre[..., None], axis=-2)
      return read, post, mix

    actual = module.apply({'params': params}, state)
    for a, b in zip(actual, oracle(params)):
      np.testing.assert_allclose(a, b, atol=1e-6, rtol=2e-6)
    self.assertTrue(np.all((actual[1] > 0) & (actual[1] < 2)))

    def objective(p):
      read, post, mix = module.apply({'params': p}, state)
      return jnp.square(mhc_update(state, read, post, mix)).sum()

    grad = jax.grad(objective)(params)
    # Finite differences cover each independently parameterized routing group.
    for name, index in [('router_kernel', (1, 2)), ('router_kernel', (2, 7)),
                        ('router_kernel', (3, 10)), ('router_scale', (2,)),
                        ('router_bias', (5,)), ('norm_scale', (2,))]:
      delta = .003
      plus, minus = copy.deepcopy(params), copy.deepcopy(params)
      plus[name] = params[name].at[index].add(delta)
      minus[name] = params[name].at[index].add(-delta)
      def numpy_objective(p):
        read, post, mix = oracle(p)
        updated = np.einsum('...ij,...id->...jd', mix, state) + post[..., None] * read[..., None, :]
        return np.square(updated).sum()
      finite_diff = (numpy_objective(plus) - numpy_objective(minus)) / (2 * delta)
      np.testing.assert_allclose(grad[name][index], finite_diff, atol=.02, rtol=.03)

  def test_mhc_initialization(self):
    cfg = connection_config()
    state = jnp.asarray(np.random.default_rng(7).normal(size=(1, 2, 4, 8)), jnp.float32)
    residual_bias = np.full((4, 4), -3., np.float32)
    np.fill_diagonal(residual_bias, 3.)
    for index in range(8):
      module = ManifoldHyperConnection(cfg, layer_index=index)
      params = module.init(jax.random.key(7), state)['params']
      expected_pre_bias = np.full(4, -3., np.float32)
      expected_pre_bias[index % 4] = 3.
      np.testing.assert_array_equal(params['router_kernel'], jnp.zeros((32, 24)))
      np.testing.assert_array_equal(params['router_scale'], jnp.full(3, .01))
      np.testing.assert_array_equal(params['norm_scale'], jnp.ones(32))
      np.testing.assert_array_equal(params['router_bias'][:4], expected_pre_bias)
      np.testing.assert_array_equal(params['router_bias'][4:8], np.zeros(4))
      np.testing.assert_array_equal(params['router_bias'][8:].reshape(4, 4), residual_bias)
      read, post, mixing = module.apply({'params': params}, state)
      pre = 1 / (1 + np.exp(-expected_pre_bias)) + 1e-6
      np.testing.assert_allclose(read, np.einsum('...nd,n->...d', state, pre), atol=1e-6)
      np.testing.assert_allclose(post, 1)
      np.testing.assert_allclose(mixing, np.broadcast_to(numpy_sinkhorn(residual_bias), mixing.shape), atol=2e-7)
    gradients = jax.grad(lambda p: sum(jnp.square(x).sum() for x in module.apply({'params': p}, state)))(params)
    np.testing.assert_array_equal(gradients['router_scale'], jnp.zeros(3))
    np.testing.assert_array_equal(gradients['norm_scale'], jnp.zeros(32))
    self.assertGreater(float(jnp.linalg.norm(gradients['router_kernel'])), 0)

  def test_mhc_learned_head_initialization(self):
    module = MHCLearnedStreamReduce(connection_config())
    state = jnp.zeros((1, 2, 4, 8))
    params = module.init(jax.random.key(9), state)['params']
    self.assertEqual(params['head_kernel'].shape, (32, 4))
    self.assertTrue(.01 < float(params['head_kernel'].std()) < .03)
    np.testing.assert_array_equal(params['norm_scale'], jnp.ones(32))
    np.testing.assert_array_equal(params['head_scale'], jnp.full(1, .01))
    np.testing.assert_array_equal(params['head_bias'], jnp.zeros(4))
    np.testing.assert_array_equal(module.apply({'params': params}, state), np.zeros((1, 2, 8)))

  def test_mhc_learned_head_forward_and_gradients(self):
    cfg = connection_config()
    module = MHCLearnedStreamReduce(cfg)
    state = jnp.asarray(np.random.default_rng(3).normal(size=(1, 2, 4, 3)), jnp.float32)
    params = module.init(jax.random.key(9), state)['params']
    params['head_kernel'] = jnp.asarray(np.random.default_rng(4).normal(scale=.3, size=(12, 4)), jnp.float32)
    params['head_scale'] = jnp.array([.7])
    params['head_bias'] = jnp.array([1.5, -.8, .3, -1.])
    params['norm_scale'] = jnp.linspace(.5, 1.5, 12)

    def oracle(p):
      flat = np.asarray(state, dtype=np.float64).reshape(1, 2, 12)
      normalized = flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + cfg.mhc_head_norm_epsilon)
      logits = (normalized * np.asarray(p['norm_scale'])) @ np.asarray(p['head_kernel'])
      gates = 1 / (1 + np.exp(-(logits * np.asarray(p['head_scale']) + np.asarray(p['head_bias'])))) + cfg.mhc_pre_epsilon
      return np.sum(np.asarray(state) * gates[..., None], axis=-2)

    actual = module.apply({'params': params}, state)
    np.testing.assert_allclose(actual, oracle(params), atol=1e-6, rtol=2e-6)
    self.assertGreater(float(jnp.linalg.norm(actual - state.mean(-2))), .1)
    grad = jax.grad(lambda p: jnp.square(module.apply({'params': p}, state)).sum())(params)
    for name, index in [('head_kernel', (1, 2)), ('head_scale', (0,)),
                        ('head_bias', (2,)), ('norm_scale', (1,))]:
      plus, minus = copy.deepcopy(params), copy.deepcopy(params)
      delta = .003
      plus[name] = params[name].at[index].add(delta)
      minus[name] = params[name].at[index].add(-delta)
      finite_diff = (np.square(oracle(plus)).sum() - np.square(oracle(minus)).sum()) / (2 * delta)
      self.assertTrue(np.isfinite(grad[name]).all())
      np.testing.assert_allclose(grad[name][index], finite_diff, atol=.002, rtol=.02)
    bf16 = module.apply({'params': params}, state.astype(jnp.bfloat16))
    self.assertEqual(bf16.dtype, jnp.bfloat16)
    self.assertTrue(np.isfinite(bf16).all())


class ResidualModelTest(MLPWriteTest, unittest.TestCase):
  def config(self, name, **kwargs):
    # Test reusable mixins without restoring the deleted formal Llama runs.
    mixins = {'Llama2MediummHC': exp.MHCConfig, 'Llama2MediumAttnRes': exp.AttnResConfig}
    if name in mixins:
      settings = {k: v for k, v in vars(mixins[name]).items() if not k.startswith('_')}
      settings.update(kwargs)
      return super().config('Llama2Medium', **settings)
    return super().config(name, **kwargs)

  def tiny(self, name, **kwargs):
    cfg = self.config(name, **kwargs)
    cfg.get_keys().update(emb_dim=16, mlp_dim=32, num_decoder_layers=2,
        num_query_heads=2, num_kv_heads=2, head_dim=8, vocab_size=32,
        dtype=jnp.float32, attention='dot_product', remat_policy='none')
    if cfg.decoder_block == 'pythia':
      cfg.get_keys()['pythia_rotary_dim'] = 2
    return cfg

  def test_configs_and_budget(self):
    baseline = self.config('Llama2Medium')
    attnres = self.config('Llama2MediumAttnRes')
    mhc = self.config('Llama2MediummHC')
    self.assertEqual(attnres.mlp_dim, 2816)
    self.assertEqual(mhc.mlp_dim, 2749)
    for cfg in (attnres, mhc):
      for key in ('emb_dim', 'num_decoder_layers', 'head_dim', 'num_query_heads', 'num_kv_heads',
                  'adam_weight_decay', 'learning_rate', 'steps', 'init_weights_seed'):
        self.assertEqual(getattr(cfg, key), getattr(baseline, key), key)
    extra = 48 * (4096 * 24 + 4096 + 24 + 3) + 4096 * 4 + 4096 + 4 + 1
    self.assertEqual(extra, 4936981)
    self.assertEqual(extra - 67 * 24 * 3 * 1024, -2795)
    self.assertEqual(self.config('Llama2MediummHC', mhc_match_parameter_budget=False).mlp_dim, 2816)
    # A mixin works on other backbone dimensions without a hard-coded width.
    small = self.config('Llama2MediummHC', base_emb_dim=64, base_mlp_dim=160)
    self.assertEqual(small.mlp_dim, 93)
    # At one layer the learned head changes the rounded MLP deduction.
    shallow = dict(base_emb_dim=64, base_mlp_dim=160, base_num_decoder_layers=1)
    self.assertEqual(self.config('Llama2MediummHC', **shallow).mlp_dim, 86)
    self.assertEqual(self.config('Llama2MediummHC', **shallow, mhc_learned_head=False).mlp_dim, 93)

  def test_reject_incompatible_architectures(self):
    for name in ('Llama2MediummHC', 'Llama2MediumAttnRes'):
      keys = dict(self.config(name).get_keys())
      for key, value in [('scan_layers', True), ('bam_enabled', True), ('dense_conn', True),
                         ('mtp_num_layers', 1), ('num_experts', 2), ('partial_scan_layers', True),
                         ('deep_embed_type', '4xmlp'), ('using_pipeline_parallelism', True)]:
        bad = dict(keys, **{key: value})
        with self.assertRaises(ValueError, msg=key):
          validate_residual_config(bad)
      for key in ('me_nums', 'bam_last_mlp_dim', 'bam_final_local_mlp_dim'):
        with self.assertRaises(ValueError, msg=key):
          validate_residual_config(dict(keys, **{key: 32}))
    keys = dict(self.config('Llama2MediummHC').get_keys())
    for key, value in [('mhc_num_streams', 0), ('mhc_sinkhorn_iters', True), ('mhc_norm_epsilon', 0),
                       ('mhc_alpha_init', float('nan')), ('mhc_sinkhorn_epsilon', -1),
                       ('mhc_pre_epsilon', -1), ('mhc_learned_head', 1), ('mhc_head_norm_epsilon', 0),
                       ('mhc_head_init_std', float('inf')), ('mhc_head_alpha_init', True),
                       ('mhc_residual_offdiag_bias', float('nan')), ('mhc_pre_selected_bias_init', float('inf'))]:
      with self.assertRaises(ValueError, msg=key):
        validate_residual_config(dict(keys, **{key: value}))

  def check_model(self, name):
    cfg = self.tiny(name)
    if cfg.residual_connection == 'mhc':
      # Exercise a nonzero learned controller: zero-projection initialization
      # deliberately starts with identical streams, regardless of alpha.
      cfg.get_keys()['mhc_router_init_std'] = .02
      cfg.get_keys()['mhc_alpha_init'] = 1.0
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    positions = jnp.arange(4)[None, :]
    mask = jnp.ones_like(tokens)
    rng = jax.random.key(7)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init({'params': rng, 'dropout': rng}, tokens, positions, tokens, mask,
                             decoder_segment_ids=mask, enable_dropout=False)
      params = nn.unbox(variables['params'])
      def forward(p, t=tokens):
        return model.apply({'params': p}, t, positions, tokens, mask,
                           decoder_segment_ids=mask, enable_dropout=False)[0]
      expected = forward(params)
      _, captured = model.apply({'params': params}, tokens, positions, tokens, mask,
          decoder_segment_ids=mask, enable_dropout=False, mutable=['intermediates'],
          capture_intermediates=lambda module, _: isinstance(module, (
              SubDecoderLayer, FullAttentionResidual, ManifoldHyperConnection, MHCLearnedStreamReduce)))
      stack = captured['intermediates']['decoder']['residual_stack']
      state0 = stack['layers_0']['__call__'][0][0]
      state1 = stack['layers_1']['__call__'][0][0]
      if cfg.residual_connection == 'attnres':
        self.assertEqual(len(state0), 3)
        self.assertEqual(len(state1), 5)
        for a, b in zip(state0, state1[:3]):
          np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(stack['output_connection']['__call__'][0],
                                   np.mean(state1, axis=0), atol=2e-6)
      else:
        self.assertEqual(state0.shape, (1, 4, 4, 16))
        self.assertEqual(state1.shape, state0.shape)
        self.assertGreater(float(jnp.linalg.norm(state0[..., 0, :] - state0[..., 1, :])), 1e-5)
        # Layer 1 must route the actual expanded layer-0 state.
        router_params = params['decoder']['residual_stack']['layers_1']['attn_connection']
        oracle = ManifoldHyperConnection(cfg, layer_index=2).apply({'params': router_params}, state0)
        for a, b in zip(stack['layers_1']['attn_connection']['__call__'][0], oracle):
          np.testing.assert_allclose(a, b, atol=2e-6)
        head_params = params['decoder']['residual_stack']['output_connection']
        read = MHCLearnedStreamReduce(cfg).apply({'params': head_params}, state1)
        np.testing.assert_allclose(stack['output_connection']['__call__'][0], read, atol=2e-6)
      np.testing.assert_allclose(forward(params, tokens.at[:, 2:].set(9))[:, :2], expected[:, :2], atol=2e-6)
      loss, gradients = jax.jit(jax.value_and_grad(lambda p: forward(p).mean()))(params)
      self.assertTrue(np.isfinite(float(loss)))
      flat = flatten_dict(gradients)
      self.assertTrue(all(np.isfinite(v).all() for v in flat.values()))
      for branch in ('self_attention', 'mlp'):
        for layer in ('layers_0', 'layers_1'):
          leaves = [v for path, v in flat.items() if layer in path and branch in path]
          self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in leaves), 0, (layer, branch))
      router_name = 'query' if name.endswith('AttnRes') else 'router_kernel'
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for p, v in flat.items() if router_name in p), 0)
      cfg.get_keys()['remat_policy'] = 'full'
      rematted = Transformer(cfg, mesh, quant=None)
      def remat_loss(p):
        return rematted.apply({'params': p}, tokens, positions, tokens, mask,
                              decoder_segment_ids=mask, enable_dropout=False)[0].mean()
      remat_value, remat_grad = jax.jit(jax.value_and_grad(remat_loss))(params)
      np.testing.assert_allclose(remat_value, loss, atol=1e-6)
      for a, b in zip(jax.tree.leaves(gradients), jax.tree.leaves(remat_grad)):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=2e-4)
      cfg.get_keys()['dtype'] = jnp.bfloat16
      bf16 = Transformer(cfg, mesh, quant=None)
      value = bf16.apply({'params': params}, tokens, positions, tokens, mask,
                         decoder_segment_ids=mask, enable_dropout=False)[0]
      self.assertTrue(np.isfinite(value).all())

  def test_attnres_forward_backward_remat_bf16(self):
    self.check_model('Llama2MediumAttnRes')

  def test_mhc_forward_backward_remat_bf16(self):
    self.check_model('Llama2MediummHC')

  def test_mhc_default_model_bias_cycle_and_head_switch(self):
    for learned in (True, False):
      cfg = self.tiny('Llama2MediummHC')
      cfg.get_keys()['mhc_learned_head'] = learned
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = Transformer(cfg, mesh, quant=None)
      tokens = jnp.array([[1, 2, 3]], jnp.int32)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        variables = model.init(jax.random.key(7), tokens, tokens-1, tokens, jnp.ones_like(tokens),
                               decoder_segment_ids=jnp.ones_like(tokens), enable_dropout=False)
        params = nn.unbox(variables['params'])
        stack = params['decoder']['residual_stack']
        for layer in range(2):
          for offset, branch in enumerate(('attn_connection', 'mlp_connection')):
            expected = np.full(4, -3.)
            expected[(2 * layer + offset) % 4] = 3.
            np.testing.assert_array_equal(stack[f'layers_{layer}'][branch]['router_bias'][:4], expected)
            self.assertTrue(np.all(np.asarray(stack[f'layers_{layer}'][branch]['router_kernel']) == 0))
        self.assertEqual('output_connection' in stack, learned)
        logits = model.apply({'params': params}, tokens, tokens-1, tokens, jnp.ones_like(tokens),
                             decoder_segment_ids=jnp.ones_like(tokens), enable_dropout=False)[0]
        self.assertTrue(np.isfinite(logits).all())

  def test_full_size_abstract_parameter_counts(self):
    counts = {}
    cases = [('Llama2Medium', True), ('Llama2MediumAttnRes', True),
             ('Llama2MediummHC', True), ('Llama2MediummHC', False)]
    for name, learned in cases:
      cfg = self.config(name, mhc_learned_head=learned)
      cfg.get_keys().update(attention='dot_product')  # Exercise formal full-remat init.
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = Transformer(cfg, mesh, quant=None)
      tokens = jnp.ones((1, 4), jnp.int32)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        shaped = jax.eval_shape(lambda: model.init(jax.random.key(1), tokens, tokens-1, tokens, tokens,
            decoder_segment_ids=tokens, enable_dropout=False))
      key = name if learned else name + '_mean'
      counts[key] = sum(x.size for x in jax.tree.leaves(nn.unbox(shaped['params'])))
    self.assertEqual(counts['Llama2MediumAttnRes'] - counts['Llama2Medium'], 98304)
    self.assertEqual(counts['Llama2MediummHC'] - counts['Llama2Medium'], -2795)
    self.assertEqual(counts['Llama2MediummHC_mean'] - counts['Llama2Medium'], -23280)
    print('ABSTRACT_PARAMETER_COUNTS', counts, flush=True)

  def test_pythia_mhc_config_and_parameter_budget(self):
    baseline = self.config('PythiaMediumPropT2048SerialMLP')
    cfg = self.config('PythiaMediumPropT2048SerialMLPmHC')
    self.assertEqual(cfg.mlp_dim, 4699)
    self.assertEqual(cfg.residual_connection, 'mhc')
    self.assertTrue(cfg.pythia_use_fusion)
    self.assertFalse(cfg.scan_layers)
    for key in ('emb_dim', 'num_decoder_layers', 'head_dim', 'num_query_heads',
                'num_kv_heads', 'mlp_activations', 'decoder_block', 'init_method',
                'qkv_bias', 'normalization_layer_epsilon', 'pythia_rotary_dim',
                'pythia_parallel_residual', 'adam_weight_decay', 'learning_rate',
                'steps', 'init_weights_seed', 'data_shuffle_seed'):
      self.assertEqual(getattr(cfg, key), getattr(baseline, key), key)
    for overrides in ({'pythia_use_fusion': False}, {'pythia_parallel_residual': True}):
      with self.assertRaises(ValueError):
        validate_residual_config(dict(cfg.get_keys(), **overrides))
    counts = []
    for c in (baseline, cfg):
      c.get_keys().update(attention='dot_product')
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quant=None)
      tokens = jnp.ones((1, 4), jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        shaped = jax.eval_shape(lambda: model.init(jax.random.key(1), tokens, tokens-1,
            tokens, tokens, decoder_segment_ids=tokens, enable_dropout=False))
      counts.append(sum(x.size for x in jax.tree.leaves(nn.unbox(shaped['params']))))
    self.assertEqual(counts, [432360000, 432339959])
    print('PYTHIA_MHC_PARAMETER_COUNTS', counts, flush=True)

  def test_pythia_mhc_forward_backward_remat_bf16(self):
    self.check_model('PythiaMediumPropT2048SerialMLPmHC')

  def test_prefill_and_autoregressive_cache(self):
    for name in ('Llama2MediumAttnRes', 'Llama2MediummHC'):
      cfg = self.tiny(name)
      cfg.get_keys()['max_prefill_predict_length'] = 2
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = Transformer(cfg, mesh, quant=None)
      tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
      positions = jnp.arange(4)[None, :]
      mask = jnp.ones_like(tokens)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        variables = model.init(jax.random.key(2), tokens, positions, tokens, mask,
                               decoder_segment_ids=mask, enable_dropout=False)
        full = model.apply(variables, tokens, positions, tokens, mask,
                           decoder_segment_ids=mask, enable_dropout=False)[0]
        result, cache = model.apply(variables, tokens[:, :2], positions[:, :2], tokens[:, :2], mask[:, :2],
            decoder_segment_ids=mask[:, :2], enable_dropout=False,
            model_mode=common_types.MODEL_MODE_PREFILL, mutable=['cache'])
        np.testing.assert_allclose(result[0], full[:, :2], atol=3e-6, rtol=3e-6)
        for index in (2, 3):
          variables = dict(variables, **cache)
          result, cache = model.apply(variables, tokens[:, index:index+1], positions[:, index:index+1],
              tokens[:, index:index+1], mask[:, index:index+1], enable_dropout=False,
              model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE, mutable=['cache'])
          np.testing.assert_allclose(result[0], full[:, index:index+1], atol=3e-6, rtol=3e-6)


if __name__ == '__main__':
  unittest.main()

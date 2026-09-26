"""Checks for the matrix-stream RMT/ALiBi port and static BAM endpoint."""

import contextlib
import io
from pathlib import Path
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from layers import attentions, models


class RMTMediumPropTest(absltest.TestCase):

  def test_single_outer_write_matches_values_gradients_and_health(self):
    from layers import rmt
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNorm')
    cfg.get_keys().update(dtype=jnp.float32, weight_dtype=jnp.float32)
    x = jax.random.normal(jax.random.key(2), (1, 2, cfg.emb_dim))
    data = jax.random.normal(jax.random.key(3), (1, 2, 16, 75))
    data *= jnp.linspace(.2, 2., 16)[None, None, :, None]
    address = .1 * jax.random.normal(jax.random.key(4), (16, 48))
    module = rmt.RMTDynamicWrite(cfg, 48)
    variables = module.init(jax.random.key(5), x, data)

    def original(p, x, y, a):
      dynamic, _ = module.apply(p, x, y)
      return dynamic + jnp.einsum('btnv,nk->btkv', y, a)

    def fused(p, x, y, a):
      return module.apply(p, x, y, a)[0]

    expected = original(variables, x, data, address)
    actual, gate, health = module.apply(variables, x, data, address)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    probe = jax.random.normal(jax.random.key(6), actual.shape)
    objective = lambda fn: lambda p, x, y, a: jnp.sum(fn(p, x, y, a) * probe)
    grad_old = jax.grad(objective(original), argnums=(0, 1, 2, 3))(variables, x, data, address)
    grad_new = jax.grad(objective(fused), argnums=(0, 1, 2, 3))(variables, x, data, address)
    for a, b in zip(jax.tree.leaves(grad_old), jax.tree.leaves(grad_new)):
      np.testing.assert_allclose(a, b, rtol=3e-4, atol=2e-5)
    dynamic, _ = module.apply(variables, x, data)
    static = jnp.einsum('btnv,nk->btkv', data, address)
    expected_health = tuple(v for part in (slice(None, 16), slice(16, None))
                            for v in rmt._write_health(dynamic[..., part, :], static[..., part, :]))
    np.testing.assert_allclose(health, expected_health, rtol=2e-5, atol=2e-6)

    cfg.get_keys()['dtype'] = jnp.bfloat16
    x, data, address = x.astype(jnp.bfloat16), data.astype(jnp.bfloat16), address.astype(jnp.bfloat16)
    old_bf16 = original(variables, x, data, address).astype(jnp.float32)
    new_bf16 = fused(variables, x, data, address).astype(jnp.float32)
    relative_error = jnp.linalg.norm(new_bf16-old_bf16)/jnp.linalg.norm(old_bf16)
    self.assertLess(float(relative_error), .01)

  def test_single_outer_model_preserves_parameters_and_health_schema(self):
    from layers import rmt
    old_model, args, old_params = self._run('RMTMediumPropK48DynamicFull48RoPE18VectorNorm')
    model, _, params = self._run('RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite')
    self.assertEqual(jax.tree.structure(params), jax.tree.structure(old_params))
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(old_params)):
      np.testing.assert_array_equal(a, b)
    with contextlib.redirect_stdout(io.StringIO()):
      _, intermediate = model.apply({'params': params}, **args, mutable=['intermediates'])
    health = np.asarray(intermediate['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
    self.assertEqual(health.shape, (3, len(rmt.RMT_DYNAMIC_HEALTH_NAMES)))
    self.assertTrue(np.all(np.isfinite(health)))

  def test_alibi_source_slopes_and_causal_segment_mask(self):
    bias = attentions._alibi_bias(2, 1, 2, 0, 3)
    slopes = np.geomspace(2 ** -4, 2 ** -8, 2)
    np.testing.assert_allclose(np.asarray(bias[:, 0, :]),
                               -slopes[:, None] * np.array([[1, 0, -1]]),
                               rtol=1e-6, atol=1e-8)
    q = jnp.zeros((1, 1, 2, 1))
    k = jnp.zeros((1, 3, 2, 1))
    v = jnp.array([0., 10., 100.])[None, :, None, None]
    v = jnp.broadcast_to(v, (1, 3, 2, 1))
    valid = jnp.array([[[False, True, False]]])
    y, alpha = attentions._attention_op(
        q, k, v, valid, float32_logits=True, additive_bias=bias)
    np.testing.assert_allclose(np.asarray(y), 10.)
    np.testing.assert_allclose(np.asarray(alpha[..., 1]), 1.)

  def test_causal_prefix_matches_full_masked_attention(self):
    q = jax.random.normal(jax.random.key(1), (1, 4, 2, 3))
    k = jax.random.normal(jax.random.key(2), (1, 4, 2, 3))
    v = jax.random.normal(jax.random.key(3), (1, 4, 2, 3))
    segments = jnp.array([[1, 1, 2, 2]])
    for q0 in (0, 2):
      q1 = q0 + 2
      target = jnp.arange(q0, q1)[:, None]
      full_valid = ((jnp.arange(4)[None, :] <= target)[None]
                    & (segments[:, q0:q1, None] == segments[:, None, :]))
      prefix_valid = full_valid[..., :q1]
      full, _ = attentions._attention_op(
          q[:, q0:q1], k, v, full_valid, float32_logits=True,
          additive_bias=attentions._alibi_bias(2, q0, q1, 0, 4))
      prefix, _ = attentions._attention_op(
          q[:, q0:q1], k[:, :q1], v[:, :q1], prefix_valid,
          float32_logits=True,
          additive_bias=attentions._alibi_bias(2, q0, q1, 0, q1))
      np.testing.assert_allclose(np.asarray(prefix), np.asarray(full),
                                 rtol=1e-6, atol=1e-6)

  def _config(self, name):
    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    Path(output.name, 'test').mkdir()
    dynamic_rmt = name.startswith(('RMTMediumPropAlibiK48Dynamic',
                                   'RMTMediumPropK48Dynamic'))
    heads = 16 if dynamic_rmt else 2
    head_dim = (20 if 'RoPE18' in name else 3) if dynamic_rmt else 75
    with contextlib.redirect_stdout(io.StringIO()):
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
          exp_class=name, run_name='test', enable_checkpointing=False,
          base_output_directory=output.name + '/', jax_cache_dir='',
          log_config=False, dataset_type='synthetic', base_emb_dim=heads * head_dim,
          base_num_query_heads=heads, base_num_kv_heads=heads, base_num_decoder_layers=3,
          base_mlp_dim=128, head_dim=head_dim, max_target_length=4,
          max_prefill_predict_length=4, query_chunk_size=2,
          per_device_batch_size=1.)
    cfg.get_keys().update(emb_bam_num_head=heads,
                          bam_write_v_bottleneck_dim=32,
                          mlp_dim_by_block=[128, 128, 128])
    if name.startswith('BamMediumPropK75AllLocal'):
      cfg.get_keys()['bam_layer_modes'] = ['local_qk+local_v+local_o'] * 3
    return cfg

  def _run(self, name):
    cfg = self._config(name)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None)
    args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                decoder_positions=jnp.arange(4)[None],
                decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                decoder_segment_ids=jnp.ones((1, 4), jnp.int32),
                enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params = model.init(jax.random.key(1), **args)['params']
      output = model.apply({'params': params}, **args)
    self.assertEqual(output[0].shape, (1, 4))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in output))
    return model, args, params

  def test_rmt_matrix_path_and_gradients(self):
    model, args, params = self._run('RMTMediumPropAlibiK48')
    self.assertEqual(params['decoder']['seed_key'].shape, (2, 48))
    layer = params['decoder']['layers']
    self.assertEqual(layer['qkv_key'].shape, (3, 3, 2, 48))
    with contextlib.redirect_stdout(io.StringIO()):
      grads = jax.grad(lambda p: jnp.sum(model.apply({'params': p}, **args)[0]))(params)
    for key in ('qkv_key', 'attn_write_key', 'mlp_read_key', 'mlp_write_key'):
      self.assertGreater(float(jnp.linalg.norm(grads['decoder']['layers'][key])), 0.)

  def test_dynamic_rmt_write_scope_and_health(self):
    from layers import rmt
    for name, address_dim, read_dim in (
        ('RMTMediumPropAlibiK48DynamicTail32', 32, 32),
        ('RMTMediumPropAlibiK48DynamicFull48', 48, 32),
        ('RMTMediumPropAlibiK48DynamicFull48NoO', 48, 32),
        ('RMTMediumPropAlibiK48DynamicReadWriteFull48', 48, 48),
    ):
      cfg = self._config(name)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_positions=jnp.arange(4)[None],
                  decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32),
                  enable_dropout=False)
      with contextlib.redirect_stdout(io.StringIO()):
        params = model.init(jax.random.key(1), **args)['params']
        _, intermediates = model.apply(
            {'params': params}, **args, mutable=['intermediates'])
      layers = params['decoder']['layers']
      self.assertEqual(layers['dynamic_qk']['basis_bias'].value.shape,
                       (4, 3, read_dim))
      self.assertEqual(layers['dynamic_vo']['compression'].value.shape,
                       (read_dim, 3, 8))
      self.assertEqual(layers['dynamic_mlp_read']['compression'].value.shape,
                       (read_dim, 3, 8))
      self.assertEqual(layers['dynamic_attn_write']['address_up_bias'].value.shape,
                       (16, 3, address_dim))
      self.assertEqual(layers['dynamic_mlp_write']['address_up_bias'].value.shape,
                       (16, 3, address_dim))
      for module, parameter in (
          ('dynamic_qk', 'basis_kernel'),
          ('dynamic_qk', 'q_mix_kernel'),
          ('dynamic_vo', 'key_kernel'),
          ('dynamic_mlp_read', 'key_kernel'),
          ('dynamic_attn_write', 'address_down'),
          ('dynamic_attn_write', 'address_up'),
          ('dynamic_mlp_write', 'address_down'),
          ('dynamic_mlp_write', 'address_up'),
      ):
        self.assertEqual(layers[module][parameter].names[0], 'embed',
                         f'{module}/{parameter} must shard its input axis')
      health = intermediates['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
      self.assertEqual(health.shape, (3, len(rmt.RMT_DYNAMIC_HEALTH_NAMES)))
      self.assertTrue(bool(jnp.all(jnp.isfinite(health))))
      if name.endswith('NoO'):
        self.assertEqual(layers['dynamic_vo']['gate_kernel'].value.shape[-1],
                         cfg.num_query_heads)
        for metric in ('o_dynamic_rms', 'o_ratio', 'o_gate_mean'):
          index = rmt.RMT_DYNAMIC_HEALTH_NAMES.index(metric)
          np.testing.assert_array_equal(np.asarray(health[:, index]), 0.)
      else:
        self.assertEqual(layers['dynamic_vo']['gate_kernel'].value.shape[-1],
                         2 * cfg.num_query_heads)
      index = rmt.RMT_DYNAMIC_HEALTH_NAMES.index('attn_write_first16_ratio')
      first16_ratio = np.asarray(health[:, index])
      if address_dim == 32:
        np.testing.assert_array_equal(first16_ratio, 0.)
      else:
        self.assertTrue(np.all(first16_ratio > 0.))

  def test_dynamic_rmt_branches_receive_gradients_at_initialization(self):
    model, args, params = self._run('RMTMediumPropAlibiK48DynamicTail32')
    with contextlib.redirect_stdout(io.StringIO()):
      grads = jax.grad(lambda p: jnp.sum(model.apply({'params': p}, **args)[0]))(params)
    layer = grads['decoder']['layers']
    for module, parameter in (
        ('dynamic_qk', 'basis_kernel'),
        ('dynamic_qk', 'q_mix_kernel'),
        ('dynamic_vo', 'key_kernel'),
        ('dynamic_mlp_read', 'key_kernel'),
        ('dynamic_attn_write', 'address_up'),
        ('dynamic_mlp_write', 'address_up'),
    ):
      value = layer[module][parameter].value
      self.assertTrue(bool(jnp.all(jnp.isfinite(value))), f'{module}/{parameter}')
      self.assertGreater(float(jnp.linalg.norm(value)), 0., f'{module}/{parameter}')

  def test_static_bam_has_no_dynamic_matrix_read_or_write_keys(self):
    _, _, params = self._run('BamMediumPropK75AllLocalStaticAlibiRMTBudget')
    from flax.traverse_util import flatten_dict
    paths = ['/'.join(p) for p in flatten_dict(params)]
    for name in ('P_loc_down', 'P_loc_up', 'W_local_packed', 'W_R', 'W_gw'):
      self.assertFalse(any(name in path for path in paths), name)
    self.assertTrue(any('P_loc_static_bias' in path for path in paths))

  def test_rope_bridge_runs_with_eighteen_dimensional_standard_qk(self):
    _, _, params = self._run('BamMediumPropK75AllLocalQK57RoPE18RMTBudget')
    from flax.traverse_util import flatten_dict
    projection_shapes = [getattr(v, 'value', v).shape for p, v in
                         flatten_dict(params).items()
                         if p[-2:] in (('query', 'kernel'), ('key', 'kernel'))]
    self.assertLen(projection_shapes, 6)
    self.assertTrue(all(shape[-1] == 18 for shape in projection_shapes))

  def test_dynamic_rmt_rope_bridge_has_independent_qk(self):
    _, _, params = self._run('RMTMediumPropK48DynamicFull48RoPE18')
    layer = params['decoder']['layers']
    for arm in ('q', 'k'):
      self.assertEqual(layer[f'{arm}_rope_kernel'].value.shape, (320, 3, 16 * 18))

  def test_vector_norm_replaces_only_layer_matrix_norms(self):
    _, _, params = self._run('RMTMediumPropK48DynamicFull48RoPE18VectorNorm')
    decoder = params['decoder']
    layer = decoder['layers']
    self.assertNotIn('attn_norm', layer)
    self.assertNotIn('mlp_norm', layer)
    self.assertEqual(layer['attn_vector_norm']['scale'].value.shape, (320, 3))
    self.assertEqual(layer['mlp_vector_norm']['scale'].value.shape, (320, 3))
    self.assertEqual(decoder['final_matrix_norm']['scale'].shape, (48, 20))

  def test_static_mlp_ablation_keeps_dynamic_attention(self):
    from layers import rmt
    name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLP'
    cfg = self._config(name)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None)
    args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                decoder_positions=jnp.arange(4)[None],
                decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                decoder_segment_ids=jnp.ones((1, 4), jnp.int32),
                enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params = model.init(jax.random.key(1), **args)['params']
      _, intermediates = model.apply(
          {'params': params}, **args, mutable=['intermediates'])
    layer = params['decoder']['layers']
    for name in ('dynamic_qk', 'dynamic_vo', 'dynamic_attn_write'):
      self.assertIn(name, layer)
    self.assertEqual(layer['dynamic_vo']['gate_kernel'].value.shape[-1],
                     2 * cfg.num_query_heads)
    for name in ('dynamic_mlp_read', 'dynamic_mlp_write', 'mlp_vector_norm'):
      self.assertNotIn(name, layer)
    for name in ('mlp_read_key', 'mlp_write_key', 'attn_vector_norm'):
      self.assertIn(name, layer)
    health = np.asarray(
        intermediates['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
    for name in ('mlp_dynamic_rms', 'mlp_ratio', 'mlp_gate_mean',
                 'mlp_write_gate_mean', 'mlp_write_first16_ratio',
                 'mlp_write_tail32_ratio'):
      np.testing.assert_array_equal(
          health[:, rmt.RMT_DYNAMIC_HEALTH_NAMES.index(name)], 0.)

  def test_static_mlp_read_pre_norm_is_after_matrix_read(self):
    model, args, params = self._run(
        'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPPreNorm')
    layer = params['decoder']['layers']
    self.assertEqual(layer['mlp_read_vector_norm']['scale'].value.shape, (320, 3))
    self.assertIn('mlp_read_key', layer)
    self.assertNotIn('dynamic_mlp_read', layer)
    self.assertNotIn('mlp_norm', layer)
    with contextlib.redirect_stdout(io.StringIO()):
      grads = jax.grad(lambda p: jnp.sum(model.apply({'params': p}, **args)[0]))(params)
    scale_grad = grads['decoder']['layers']['mlp_read_vector_norm']['scale'].value
    self.assertTrue(bool(jnp.all(jnp.isfinite(scale_grad))))
    self.assertGreater(float(jnp.linalg.norm(scale_grad)), 0.)

  def test_dynamic_mlp_read_with_static_write_is_trainable(self):
    from layers import rmt
    model, args, params = self._run(
        'RMTMediumPropK48DynamicFull48RoPE18VectorNormStaticMLPDynamicRead')
    layer = params['decoder']['layers']
    for name in ('dynamic_mlp_read', 'mlp_vector_norm', 'mlp_read_key', 'mlp_write_key'):
      self.assertIn(name, layer)
    for name in ('dynamic_mlp_write', 'mlp_read_vector_norm', 'mlp_norm'):
      self.assertNotIn(name, layer)
    with contextlib.redirect_stdout(io.StringIO()):
      grads = jax.grad(lambda p: jnp.sum(model.apply({'params': p}, **args)[0]))(params)
      _, intermediate = model.apply({'params': params}, **args, mutable=['intermediates'])
    key_grad = grads['decoder']['layers']['dynamic_mlp_read']['key_kernel'].value
    self.assertTrue(bool(jnp.all(jnp.isfinite(key_grad))))
    self.assertGreater(float(jnp.linalg.norm(key_grad)), 0.)
    health = np.asarray(
        intermediate['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
    for name in ('mlp_write_gate_mean', 'mlp_write_first16_ratio', 'mlp_write_tail32_ratio'):
      np.testing.assert_array_equal(health[:, rmt.RMT_DYNAMIC_HEALTH_NAMES.index(name)], 0.)


if __name__ == '__main__':
  absltest.main()

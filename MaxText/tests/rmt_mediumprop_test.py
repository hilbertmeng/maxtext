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

  def test_dynamic_embedding_matches_original_bam_write(self):
    from flax import linen as nn
    from layers import initializers, rmt
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbedding')
    cfg.get_keys().update(dtype=jnp.float32, bam_k=cfg.head_dim, bam_v=48,
                          emb_bam_v_bottleneck_dim=256, bam_sqrt_n_scale=False)
    x = jax.random.normal(jax.random.key(101), (1, 4, cfg.emb_dim))
    original = models.EmbeddingBamWrite(
        cfg, num_write_heads=16, dtype=jnp.float32, weight_dtype=jnp.float32,
        quant=None, kernel_init=initializers.get_init_method(cfg.init_method))
    params = nn.unbox(original.init(jax.random.key(102), x)['params'])
    dynamic = rmt.RMTDynamicWrite(cfg, address_dim=48)
    converted = dict(
        address_down=params['W_emb_v_down']['kernel'],
        address_up=params['W_emb_v_up']['kernel'].reshape((256, 16 * 48)),
        address_up_bias=params['W_emb_v_up']['bias'],
        gate_kernel=params['W_emb_g']['kernel'], gate_bias=params['emb_gw_b0'])
    def new_write(p, z):
      data = jnp.einsum('btd,dnv->btnv', z, p['W_emb_u']['kernel'])
      mapped = dict(converted)
      mapped.update(address_down=p['W_emb_v_down']['kernel'],
                    address_up=p['W_emb_v_up']['kernel'].reshape((256, 16 * 48)),
                    address_up_bias=p['W_emb_v_up']['bias'],
                    gate_kernel=p['W_emb_g']['kernel'], gate_bias=p['emb_gw_b0'])
      return dynamic.apply({'params': mapped}, z, data)[0]
    expected = jnp.swapaxes(original.apply({'params': params}, x), -2, -1)
    np.testing.assert_allclose(new_write(params, x), expected, rtol=2e-5, atol=2e-5)
    old_grad = jax.grad(lambda p, z: jnp.mean(original.apply({'params': p}, z)**2), (0, 1))(params, x)
    new_grad = jax.grad(lambda p, z: jnp.mean(new_write(p, z)**2), (0, 1))(params, x)
    for a, b in zip(jax.tree.leaves(old_grad), jax.tree.leaves(new_grad)):
      np.testing.assert_allclose(a, b, rtol=5e-4, atol=2e-5)

  def test_dynamic_boundaries_health_initialization_and_gradients(self):
    from flax import linen as nn
    from layers import rmt
    base = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget'
    for arm in ('Embedding', 'Unembedding', 'UnembeddingDirect32'):
      cfg = self._config(base + 'Dynamic' + arm)
      cfg.get_keys().update(dtype=jnp.float32, rmt_mlp_dim_by_block=[128, 128, 128])
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.array([[1, 2, 3, 4]], jnp.int32),
                  decoder_positions=jnp.arange(4)[None], decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32), enable_dropout=False)
      with contextlib.redirect_stdout(io.StringIO()):
        params = nn.unbox(model.init(jax.random.key(103), **args)['params'])
        output, aux = model.apply({'params': params}, **args, mutable=['intermediates'])
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in output))
      decoder = params['decoder']
      name = 'embedding' if arm == 'Embedding' else 'unembedding'
      self.assertIn('seed_key', decoder)
      self.assertIn('final_read_key', decoder)
      self.assertIn('dynamic_' + name + ('_write' if arm == 'Embedding' else '_read'), decoder)
      health = aux['intermediates']['decoder']['rmt_' + name + '_health'][0]
      self.assertEqual(health.shape, (len(rmt.RMT_BOUNDARY_HEALTH_NAMES),))
      if arm.startswith('Unembedding'):
        read = decoder['dynamic_unembedding_read']
        if arm == 'Unembedding':
          self.assertEqual(read['compression'].shape, (32, 8))
          self.assertEqual(read['key_kernel'].shape, (cfg.emb_dim, 16 * 8))
        else:
          self.assertNotIn('compression', read)
          self.assertEqual(read['key_kernel'].shape, (cfg.emb_dim, 16 * 32))
        np.testing.assert_array_equal(read['key_kernel'], 0.)
        self.assertEqual(float(health[2]), 0.)
        self.assertAlmostEqual(float(health[4]), .05, places=6)
        parent_cfg = self._config(base)
        parent_cfg.get_keys().update(dtype=jnp.float32, rmt_mlp_dim_by_block=[128, 128, 128])
        parent = models.Transformer(config=parent_cfg, mesh=mesh, quant=None)
        with contextlib.redirect_stdout(io.StringIO()):
          parent_params = nn.unbox(parent.init(jax.random.key(103), **args)['params'])
          parent_output = parent.apply({'params': parent_params}, **args)
        for a, b in zip(jax.tree.leaves(output), jax.tree.leaves(parent_output)):
          np.testing.assert_array_equal(a, b)
      with contextlib.redirect_stdout(io.StringIO()):
        gradient = jax.grad(lambda p: jnp.sum(model.apply({'params': p}, **args)[0]))(params)
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(gradient)))
      if arm == 'Embedding':
        for key in ('address_down', 'address_up', 'gate_kernel'):
          self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['dynamic_embedding_write'][key])), 0.)
        self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['embedding_write_content']['kernel'])), 0.)
      else:
        self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['dynamic_unembedding_read']['key_kernel'])), 0.)


  def test_direct32_unembedding_matches_full_tail_read_and_gradients(self):
    from flax import linen as nn
    from layers import rmt
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicUnembeddingDirect32')
    cfg.get_keys()['dtype'] = jnp.float32
    module = rmt.RMTDynamicC8Read(cfg, destinations=1, compress_state=False)
    x = jax.random.normal(jax.random.key(105), (1, 4, cfg.emb_dim))
    matrix = jax.random.normal(jax.random.key(106), (1, 4, cfg.head_dim, 32))
    params = nn.unbox(module.init(jax.random.key(107), x, matrix)['params'])
    self.assertNotIn('compression', params)
    params['key_kernel'] = jax.random.normal(jax.random.key(108), params['key_kernel'].shape)
    params['gate_kernel'] = .01 * jax.random.normal(jax.random.key(109), params['gate_kernel'].shape)
    def actual(p, z, M):
      return module.apply({'params': p}, z, M)[0][0]
    def expected(p, z, M):
      raw = jnp.einsum('btd,dr->btr', z, p['key_kernel']).reshape((1, 4, 16, 32))
      key = rmt.normalizations.rms_norm(
          raw, dtype=z.dtype, epsilon=rmt._read_epsilon(cfg), statistics_dtype=jnp.float32)
      logits = jnp.einsum('btd,dr->btr', z, p['gate_kernel']).reshape((1, 4, 16, 1))
      gate = jax.nn.sigmoid(logits + p['gate_bias'])
      return .2 * gate[..., 0, None] * jnp.einsum('btvc,btnc->btnv', M, key)
    np.testing.assert_allclose(actual(params, x, matrix), expected(params, x, matrix), rtol=1e-5, atol=1e-5)
    old = jax.grad(lambda p, z, M: jnp.sum(expected(p, z, M)**2), (0, 1, 2))(params, x, matrix)
    new = jax.grad(lambda p, z, M: jnp.sum(actual(p, z, M)**2), (0, 1, 2))(params, x, matrix)
    for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
      np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

  def test_headwise_mlp_preserves_initialization_values_and_gradients(self):
    from flax import linen as nn
    from layers import initializers, linears
    cfg = self._config('RMTVectorNormMHABudgetHeadwiseMLPProfile')
    heads, width = 16, cfg.head_dim
    x = jax.random.normal(jax.random.key(95), (1, 4, heads, width))
    for dtype, tolerance in ((jnp.float32, 2e-5), (jnp.bfloat16, .025)):
      kwargs = dict(config=cfg, intermediate_dim=128, activations=cfg.mlp_activations,
                    intermediate_dropout_rate=0., dtype=dtype, weight_dtype=jnp.float32,
                    kernel_init=initializers.get_init_method(cfg.init_method))
      old, new = linears.MlpBlock(**kwargs), linears.MlpBlock(**kwargs, headwise=True)
      old_params = nn.unbox(old.init(jax.random.key(96), x.reshape((1, 4, -1)), deterministic=True)['params'])
      new_params = nn.unbox(new.init(jax.random.key(96), x, deterministic=True)['params'])
      for a, b in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params)):
        np.testing.assert_array_equal(a.reshape(-1), b.reshape(-1))
      old_fn = lambda p, z: old.apply({'params': p}, z.reshape((1, 4, -1)), deterministic=True).reshape(x.shape)
      new_fn = lambda p, z: new.apply({'params': p}, z, deterministic=True)
      np.testing.assert_allclose(new_fn(new_params, x), old_fn(old_params, x),
                                 rtol=tolerance, atol=tolerance)
      _, old_grad = jax.value_and_grad(lambda p, z: jnp.sum(old_fn(p, z).astype(jnp.float32)**2), (0, 1))(old_params, x)
      _, new_grad = jax.value_and_grad(lambda p, z: jnp.sum(new_fn(p, z).astype(jnp.float32)**2), (0, 1))(new_params, x)
      for a, b in zip(jax.tree.leaves(old_grad), jax.tree.leaves(new_grad)):
        np.testing.assert_allclose(a.reshape(-1), b.reshape(-1), rtol=tolerance, atol=tolerance)

  def test_headwise_rmt_block_kernel_conversion_preserves_forward(self):
    from flax import linen as nn
    for profile, parent in (
        ('RMTVectorNormMHABudgetHeadwiseMLPProfile',
         'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget'),
        ('RMTVectorNormMHABudgetHeadwiseMLPTransposedCarryProfile',
         'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget'),
        ('RMTVectorNormMHABudgetLLFSharedVOHeadwiseMLPProfile',
         'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLFSharedVO')):
      cfg = self._config(profile)
      cfg.get_keys()['rmt_mlp_dim_by_block'] = [128, 128, 128]
      cfg.get_keys()['dtype'] = jnp.float32
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.array([[1, 2, 3, 4]], jnp.int32),
                  decoder_positions=jnp.arange(4)[None], decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32), enable_dropout=False)
      with contextlib.redirect_stdout(io.StringIO()):
        params = nn.unbox(model.init(jax.random.key(97), **args)['params'])
      old_cfg = self._config(parent)
      old_cfg.get_keys().update(dtype=jnp.float32, rmt_mlp_dim_by_block=[128, 128, 128])
      old_model = models.Transformer(config=old_cfg, mesh=mesh, quant=None)
      def flatten_mlp(path, value):
        keys = [getattr(key, 'key', None) for key in path]
        if 'mlp' not in keys or keys[-1] != 'kernel':
          return value
        axis = cfg.param_scan_axis
        v = jnp.moveaxis(value, axis, 0)
        if keys[-2].startswith('wi'):
          v = v.reshape((v.shape[0], cfg.emb_dim, v.shape[-1]))
        elif keys[-2] == 'wo':
          v = v.reshape((v.shape[0], v.shape[1], cfg.emb_dim))
        else:
          raise AssertionError(keys)
        return jnp.moveaxis(v, 0, axis)
      old_params = jax.tree.map_with_path(flatten_mlp, params)
      with contextlib.redirect_stdout(io.StringIO()):
        expected = old_model.apply({'params': old_params}, **args)
        actual = model.apply({'params': params}, **args)
      for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)

  def test_fetch_c8_shared_and_independent_keys(self):
    from layers import rmt
    from flax import linen as nn
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget')
    x = jax.random.normal(jax.random.key(80), (1, 4, cfg.emb_dim))
    matrix = jax.random.normal(jax.random.key(81), (1, 4, cfg.head_dim, 32))
    for independent in (False, True):
      module = rmt.RMTDynamicC8Read(
          cfg, destinations=2, fetch_output=True, independent_output_key=independent)
      params = nn.unbox(module.init(jax.random.key(82), x, matrix)['params'])
      params['key_kernel'] = jax.random.normal(jax.random.key(83), params['key_kernel'].shape)
      if independent:
        params['o_key_kernel'] = jax.random.normal(jax.random.key(84), params['o_key_kernel'].shape)
      v, gates, compressed, output_key = module.apply({'params': params}, x, matrix)
      expected_compressed = jnp.einsum('btvc,cr->btvr', matrix, params['compression'])
      np.testing.assert_allclose(compressed, expected_compressed, rtol=1e-5, atol=1e-5)
      raw_v = jnp.einsum('btd,dr->btr', x, params['key_kernel']).reshape((1, 4, 16, 8))
      v_key = rmt.normalizations.rms_norm(
          raw_v, dtype=x.dtype, epsilon=rmt._read_epsilon(cfg), statistics_dtype=jnp.float32)
      np.testing.assert_allclose(v, .2 * gates[..., 0, None] * jnp.einsum(
          'btvc,btnc->btnv', compressed, v_key), rtol=1e-5, atol=1e-5)
      if independent:
        self.assertGreater(float(jnp.max(jnp.abs(output_key-v_key))), .1)
      else:
        np.testing.assert_array_equal(output_key, v_key)
      np.testing.assert_allclose(gates, .05, rtol=1e-5)
      self.assertEqual('o_key_kernel' in params, independent)

  def test_llf_fetch_causality_segments_health_and_gradients(self):
    from flax import linen as nn
    from layers import rmt
    for suffix in ('SharedVO', 'IndependentVO'):
      cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetLLF' + suffix)
      cfg.get_keys()['rmt_mlp_dim_by_block'] = [128, 128, 128]
      cfg.get_keys()['dtype'] = jnp.float32
      module = rmt.RMTLayer(cfg, is_fetch=True, mlp_dim=128)
      matrix = jax.random.normal(jax.random.key(85), (1, 4, 48, cfg.head_dim))
      segments = jnp.array([[1, 1, 2, 2]])
      positions = jnp.arange(4)[None]
      params = nn.unbox(module.init(jax.random.key(86), matrix, segments, positions, True, 2)['params'])
      vo = params['dynamic_vo']
      vo['key_kernel'] = .03 * jax.random.normal(jax.random.key(87), vo['key_kernel'].shape)
      if suffix == 'IndependentVO':
        vo['o_key_kernel'] = .03 * jax.random.normal(jax.random.key(88), vo['o_key_kernel'].shape)
      apply = lambda p, m: module.apply({'params': p}, m, segments, positions, True, 2)[0]
      value = apply(params, matrix)
      future_changed = matrix.at[:, 3].add(3.)
      np.testing.assert_allclose(apply(params, future_changed)[:, :3], value[:, :3], rtol=1e-5, atol=1e-5)
      other_segment_changed = matrix.at[:, :2].add(3.)
      np.testing.assert_allclose(apply(params, other_segment_changed)[:, 2:], value[:, 2:], rtol=1e-5, atol=1e-5)
      _, gradient = jax.value_and_grad(lambda p: jnp.mean(apply(p, matrix)**2))(params)
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(gradient)))
      self.assertGreater(float(jnp.linalg.norm(gradient['fetch_head_mix_kernel'])), 0.)
      self.assertGreater(float(jnp.linalg.norm(gradient['dynamic_vo']['gate_kernel'])), 0.)
      (_, health), aux = module.apply(
          {'params': params}, matrix, segments, positions, True, 2, mutable=['intermediates'])
      self.assertEqual(health.shape, (41,))
      self.assertEqual(aux['intermediates']['rmt_fetch_route_sums'][0].shape, (6,))

  def test_mha_budget_block_scan_health(self):
    from layers import rmt
    for suffix in ('', 'LLFSharedVO', 'LLFIndependentVO'):
      name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget' + suffix
      cfg = self._config(name)
      cfg.get_keys()['rmt_mlp_dim_by_block'] = [128, 128, 128]
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_positions=jnp.arange(4)[None],
                  decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32), enable_dropout=False)
      with contextlib.redirect_stdout(io.StringIO()):
        params = model.init(jax.random.key(89), **args)['params']
        output, aux = model.apply({'params': params}, **args, mutable=['intermediates'])
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in output))
      decoder = aux['intermediates']['decoder']['layers']
      self.assertEqual(decoder['rmt_dynamic_health'][0].shape, (1, 3, 41))
      self.assertEqual('layer_2' in decoder, bool(suffix))
      if suffix:
        self.assertEqual(decoder['layer_2']['rmt_fetch_route_sums'][0].shape, (1, 6))

  def test_row_reduced_write_health_matches_partition_statistics(self):
    from layers import rmt
    for dtype in (jnp.float32, jnp.bfloat16):
      dynamic = jax.random.normal(jax.random.key(71), (2, 7, 48, 75)).astype(dtype)
      reference = jax.random.normal(jax.random.key(72), dynamic.shape).astype(dtype)
      for a, b in ((dynamic, reference), (dynamic, dynamic),
                   (jnp.zeros_like(dynamic), reference)):
        expected = tuple(v for part in (slice(None, 16), slice(16, None))
                         for v in rmt._write_health(a[..., part, :], b[..., part, :]))
        np.testing.assert_allclose(rmt._row_reduced_write_health(a, b),
                                   expected, rtol=2e-5, atol=2e-6)

  def test_transposed_carry_preserves_parameters_values_and_gradients(self):
    for optimized, parent in (
        ('RMTVectorNormRowHealthTransposedCarryProfile',
         'RMTVectorNormRowReducedWriteHealthProfile'),
        ('RMTVectorNormDynamicOnlyRowHealthTransposedCarryProfile',
         'RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile')):
      old_model, args, old_params = self._run(parent, dtype=jnp.float32)
      model, _, params = self._run(optimized, dtype=jnp.float32)
      self.assertEqual(jax.tree.structure(params), jax.tree.structure(old_params))
      for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(old_params)):
        np.testing.assert_array_equal(a, b)
      args.update(decoder_input_tokens=jnp.array([[1, 2, 3, 4]], jnp.int32),
                  decoder_target_tokens=jnp.array([[2, 3, 4, 5]], jnp.int32))
      with contextlib.redirect_stdout(io.StringIO()):
        old_value, old_grad = jax.value_and_grad(
            lambda p: jnp.mean(old_model.apply({'params': p}, **args)[0]))(old_params)
        value, grad = jax.value_and_grad(
            lambda p: jnp.mean(model.apply({'params': p}, **args)[0]))(params)
      np.testing.assert_allclose(value, old_value, rtol=1e-6, atol=1e-6)
      squared_error = squared_norm = 0.
      for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(old_grad)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)
        squared_error += float(jnp.sum(jnp.square(a-b)))
        squared_norm += float(jnp.sum(jnp.square(b)))
      self.assertLess((squared_error / squared_norm) ** .5, 3e-6)

  def test_write_health_toggle_preserves_other_metrics_and_parameters(self):
    from layers import rmt
    expected_names = tuple(n for n in rmt.RMT_DYNAMIC_HEALTH_NAMES
                          if not ('_write_' in n and n.endswith(('_ratio', '_cosine'))))
    self.assertEqual(rmt.dynamic_health_names(False), expected_names)
    self.assertLen(expected_names, 33)
    for suffix, parent in (
        ('NoWriteHealthProfile', 'RMTMediumPropK48DynamicFull48RoPE18VectorNorm'),
        ('SingleOuterNoWriteHealthProfile', 'RMTMediumPropK48DynamicFull48RoPE18VectorNormSingleOuterWrite'),
        ('DynamicOnlyNoWriteHealthProfile', 'RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite')):
      _, _, old_params = self._run(parent)
      model, args, params = self._run('RMTVectorNorm' + suffix)
      self.assertEqual(jax.tree.structure(params), jax.tree.structure(old_params))
      for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(old_params)):
        np.testing.assert_array_equal(a, b)
      with contextlib.redirect_stdout(io.StringIO()):
        _, intermediate = model.apply({'params': params}, **args, mutable=['intermediates'])
      health = np.asarray(intermediate['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
      self.assertEqual(health.shape, (3, 33))
      self.assertTrue(np.all(np.isfinite(health)))

  def test_row_reduced_health_preserves_model_and_all_metrics(self):
    pairs = (
        ('RMTVectorNormRowReducedWriteHealthProfile',
         'RMTMediumPropK48DynamicFull48RoPE18VectorNorm'),
        ('RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile',
         'RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite'))
    from layers import rmt
    write_slots = [i for i, name in enumerate(rmt.RMT_DYNAMIC_HEALTH_NAMES)
                   if '_write_' in name and name.endswith(('_ratio', '_cosine'))]
    other_slots = [i for i in range(41) if i not in write_slots]
    for dtype in (jnp.float32, jnp.bfloat16):
      for optimized, parent in pairs:
        old_model, old_args, old_params = self._run(parent, dtype=dtype)
        model, args, params = self._run(optimized, dtype=dtype)
        self.assertEqual(jax.tree.structure(params), jax.tree.structure(old_params))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(old_params)):
          np.testing.assert_array_equal(a, b)
        with contextlib.redirect_stdout(io.StringIO()):
          old_output, old_aux = old_model.apply({'params': old_params}, **old_args,
                                                mutable=['intermediates'])
          output, aux = model.apply({'params': params}, **args, mutable=['intermediates'])
        np.testing.assert_array_equal(output[0], old_output[0])
        old_health = np.asarray(old_aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
        health = np.asarray(aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
        self.assertEqual(health.shape, (3, 41))
        np.testing.assert_allclose(health[:, other_slots], old_health[:, other_slots],
                                   rtol=2e-6, atol=2e-7)
        # First prove the reordered formulas at FP32 model precision; BF16
        # scan fusion can change auxiliary statistics at the carry boundary.
        atol = 2e-6 if dtype == jnp.float32 else 1e-4
        np.testing.assert_allclose(health[:, write_slots], old_health[:, write_slots],
                                   rtol=2e-5, atol=atol)

  def test_reused_reference_rms_preserves_model_and_all_metrics(self):
    for dtype in (jnp.float32, jnp.bfloat16):
      old_model, args, old_params = self._run(
          'RMTVectorNormDynamicOnlyRowReducedWriteHealthProfile', dtype=dtype)
      for name in ('RMTVectorNormDynamicOnlyReusedInputHealthProfile',
                   'RMTVectorNormDynamicOnlyReusedInputHealthTransposedCarryProfile'):
        model, _, params = self._run(name, dtype=dtype)
        self.assertEqual(jax.tree.structure(params), jax.tree.structure(old_params))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(old_params)):
          np.testing.assert_array_equal(a, b)
        with contextlib.redirect_stdout(io.StringIO()):
          old_output, old_aux = old_model.apply({'params': old_params}, **args,
                                                mutable=['intermediates'])
          output, aux = model.apply({'params': params}, **args, mutable=['intermediates'])
        np.testing.assert_array_equal(output[0], old_output[0])
        old_health = np.asarray(old_aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
        health = np.asarray(aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
        self.assertEqual(health.shape, (3, 41))
        np.testing.assert_allclose(health, old_health, rtol=2e-5,
                                   atol=2e-6 if dtype == jnp.float32 else 1e-4)

  def test_write_contractions_preserve_values_and_gradients(self):
    from layers import rmt
    address = jax.random.normal(jax.random.key(51), (2, 3, 16, 48))
    data = jax.random.normal(jax.random.key(52), (2, 3, 16, 75))
    probe = jax.random.normal(jax.random.key(53), (2, 3, 48, 75))
    objective = lambda method: lambda a, y: jnp.sum(rmt._dynamic_outer_write(a, y, method) * probe)
    base = rmt._dynamic_outer_write(address, data, 'dot')
    gradients = jax.grad(objective('dot'), argnums=(0, 1))(address, data)
    for method in ('dot_transposed', 'mul_reduce'):
      actual = rmt._dynamic_outer_write(address, data, method)
      np.testing.assert_allclose(actual, base, atol=3e-6, rtol=2e-5)
      for a, b in zip(gradients, jax.grad(objective(method), argnums=(0, 1))(address, data)):
        self.assertLess(float(jnp.linalg.norm(a-b)/jnp.linalg.norm(a)), 2e-6)
      a, y = address.astype(jnp.bfloat16), data.astype(jnp.bfloat16)
      expected = rmt._dynamic_outer_write(a, y, 'dot').astype(jnp.float32)
      actual = rmt._dynamic_outer_write(a, y, method).astype(jnp.float32)
      self.assertLess(float(jnp.linalg.norm(actual-expected)/jnp.linalg.norm(expected)), .005)

  def test_factorized_health_uses_fp32_accumulation_for_bf16_operands(self):
    from layers import rmt
    dynamic = jax.random.normal(jax.random.key(31), (2, 3, 16, 48), dtype=jnp.bfloat16)
    static = jax.random.normal(jax.random.key(32), (16, 48), dtype=jnp.bfloat16)
    data = jax.random.normal(jax.random.key(33), (2, 3, 16, 75), dtype=jnp.bfloat16)
    expected = rmt._factorized_write_health(dynamic.astype(jnp.float32),
                                            static.astype(jnp.float32), data.astype(jnp.float32))
    actual = rmt._factorized_write_health(dynamic, static, data)
    self.assertTrue(all(value.dtype == jnp.float32 for value in actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

  def test_dynamic_only_writes_remove_static_keys_and_preserve_other_initialization(self):
    from layers import rmt
    _, _, old_params = self._run('RMTMediumPropK48DynamicFull48RoPE18VectorNorm')
    model, args, params = self._run('RMTMediumPropK48DynamicFull48RoPE18VectorNormDynamicOnlyWrite')
    layer = params['decoder']['layers']
    self.assertNotIn('attn_write_key', layer)
    self.assertNotIn('mlp_write_key', layer)
    def leaves(tree):
      return {jax.tree_util.keystr(path): np.asarray(value)
              for path, value in jax.tree_util.tree_leaves_with_path(tree)}
    old, new = leaves(old_params), leaves(params)
    removed = old.keys() - new.keys()
    self.assertLen(removed, 2)
    self.assertEqual(sum(old[key].size for key in removed), 3 * 2 * 16 * 48)
    for key in new:
      np.testing.assert_array_equal(new[key], old[key], err_msg=key)
    with contextlib.redirect_stdout(io.StringIO()):
      _, intermediate = model.apply({'params': params}, **args, mutable=['intermediates'])
      grads = jax.grad(lambda p: jnp.mean(model.apply({'params': p}, **args)[0]))(params)
    health = np.asarray(intermediate['intermediates']['decoder']['layers']['rmt_dynamic_health'][0])
    self.assertEqual(health.shape, (3, len(rmt.RMT_DYNAMIC_HEALTH_NAMES)))
    self.assertTrue(np.all(np.isfinite(health)))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(grads)))
    for arm in ('dynamic_attn_write', 'dynamic_mlp_write'):
      self.assertGreater(float(jnp.linalg.norm(grads['decoder']['layers'][arm]['address_up'].value)), 0.)

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
    largest_relative_error = 0.
    for a, b in zip(jax.tree.leaves(grad_old), jax.tree.leaves(grad_new)):
      relative_error = float(jnp.linalg.norm(a-b) / jnp.maximum(jnp.linalg.norm(a), 1e-12))
      largest_relative_error = max(largest_relative_error, relative_error)
      self.assertLess(relative_error, 3e-6)
      # Near-zero coordinates can cancel hundreds-scale summands differently.
      # Bound every coordinate relative to the leaf's peak gradient as well.
      peak_error = float(jnp.max(jnp.abs(a-b)) / jnp.maximum(jnp.max(jnp.abs(a)), 1e-12))
      self.assertLess(peak_error, 3e-6)
    print('SINGLE_WRITE_FP32_GRAD_MAX_REL', largest_relative_error)
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
    print('SINGLE_WRITE_BF16_FORWARD_REL', float(relative_error))
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
    import exp
    experiment = getattr(exp, name)
    dynamic_rmt = bool(getattr(experiment, 'rmt_dynamic_enabled', False))
    heads = 16 if dynamic_rmt else 2
    head_dim = (20 if getattr(experiment, 'rmt_rope_qk_dim', 0) else 3) if dynamic_rmt else 75
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

  def _run(self, name, dtype=None):
    cfg = self._config(name)
    if dtype is not None:
      cfg.get_keys()['dtype'] = dtype
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

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
    dynamic_rmt = name.startswith('RMTMediumPropAlibiK48Dynamic')
    heads = 16 if dynamic_rmt else 2
    head_dim = 3 if dynamic_rmt else 75
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


if __name__ == '__main__':
  absltest.main()

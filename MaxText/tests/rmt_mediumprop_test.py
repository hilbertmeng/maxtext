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

  def test_combined_boundaries_health_initialization_and_gradients(self):
    self._check_dynamic_boundary('EmbeddingUnembeddingDirect32')

  def _check_dynamic_boundary(self, arm):
    from flax import linen as nn
    from layers import rmt
    base = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudget'
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
    has_embedding = arm in ('Embedding', 'EmbeddingUnembeddingDirect32')
    has_unembedding = arm != 'Embedding'
    name = 'embedding' if has_embedding else 'unembedding'
    self.assertIn('seed_key', decoder)
    self.assertIn('final_read_key', decoder)
    self.assertIn('dynamic_' + name + ('_write' if has_embedding else '_read'), decoder)
    health = aux['intermediates']['decoder']['rmt_' + name + '_health'][0]
    self.assertEqual(health.shape, (len(rmt.RMT_BOUNDARY_HEALTH_NAMES),))
    if has_unembedding:
      read = decoder['dynamic_unembedding_read']
      if arm == 'Unembedding':
        self.assertEqual(read['compression'].shape, (32, 8))
        self.assertEqual(read['key_kernel'].shape, (cfg.emb_dim, 16 * 8))
      else:
        self.assertNotIn('compression', read)
        self.assertEqual(read['key_kernel'].shape, (cfg.emb_dim, 16 * 32))
      np.testing.assert_array_equal(read['key_kernel'], 0.)
      read_health = aux['intermediates']['decoder']['rmt_unembedding_health'][0]
      self.assertEqual(read_health.shape, (len(rmt.RMT_BOUNDARY_HEALTH_NAMES),))
      self.assertEqual(float(read_health[2]), 0.)
      self.assertAlmostEqual(float(read_health[4]), .05, places=6)
      parent_cfg = self._config(base + 'DynamicEmbedding' if has_embedding else base)
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
    if has_embedding:
      np.testing.assert_allclose(jax.nn.sigmoid(decoder['dynamic_embedding_write']['gate_bias']), .1, rtol=1e-6)
      self.assertGreater(float(health[4]), .09)
      self.assertLess(float(health[4]), .11)
      for key in ('address_down', 'address_up', 'gate_kernel'):
        self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['dynamic_embedding_write'][key])), 0.)
      self.assertGreater(float(jnp.linalg.norm(gradient['decoder']['embedding_write_content']['kernel'])), 0.)
    if has_unembedding:
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

class RMTMergedRuntimeTest(absltest.TestCase):
  """Parameter budget, NoO behavior and compatibility with the trained runtime."""

  _config = RMTMediumPropTest._config

  def test_additive_attention_bias_preserves_logits_precision(self):
    from layers import rmt
    query = jax.random.normal(jax.random.key(811), (1,2,2,4))
    key = jax.random.normal(jax.random.key(812), (1,4,2,4))
    value = jax.random.normal(jax.random.key(813), (1,4,2,4))
    valid = jnp.arange(4)[None,None,:] <= jnp.array([2,3])[None,:,None]
    bias = rmt._alibi_bias(2,2,4,0,4)
    for dtype in (jnp.float32,jnp.bfloat16):
      q,k,v = (x.astype(dtype) for x in (query,key,value))
      for fp32 in (False,True):
        logits = jnp.einsum('bqnd,bsnd->bnqs',q,k)
        logits = jnp.where(valid[:,None],logits,attentions.DEFAULT_MASK_VALUE)
        if fp32:
          logits = logits.astype(jnp.float32)
        logits = logits + bias.astype(logits.dtype)
        expected_alpha = jax.nn.softmax(logits,axis=-1)
        expected = jnp.einsum('bnqs,bsnd->bqnd',expected_alpha,v)
        actual,alpha = attentions._attention_op(q,k,v,valid,float32_logits=fp32,additive_bias=bias)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(alpha,expected_alpha)
        self.assertEqual(alpha.dtype,expected_alpha.dtype)

  def test_full_parameter_budgets_and_no_o_savings(self):
    import math
    prefix = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32'
    for suffix, expected in (('', 432119360), ('L22', 432083008)):
      with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        Path(directory, 'audit').mkdir()
        cfg = pyconfig.initialize(
            [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
            exp_class=prefix + suffix, run_name='audit', enable_checkpointing=False,
            base_output_directory=directory + '/', jax_cache_dir='', log_config=False,
            dataset_type='synthetic', max_target_length=4,
            max_prefill_predict_length=4, query_chunk_size=2, per_device_batch_size=1.)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        args = dict(decoder_input_tokens=jnp.ones((1,4), jnp.int32),
                    decoder_positions=jnp.arange(4)[None],
                    decoder_target_tokens=jnp.ones((1,4), jnp.int32),
                    decoder_target_mask=jnp.ones((1,4), jnp.float32),
                    decoder_segment_ids=jnp.ones((1,4), jnp.int32), enable_dropout=False)
        def count():
          model = models.Transformer(config=cfg, mesh=mesh, quant=None)
          shapes = jax.eval_shape(lambda key: model.init(key, **args)['params'], jax.random.key(1))
          return sum(math.prod(x.shape) for x in jax.tree.leaves(shapes))
        self.assertEqual(count(), expected)
        cfg.get_keys()['rmt_dynamic_o_enabled'] = False
        # The shared V/O key remains; only one O gate kernel/bias per layer disappears.
        self.assertEqual(count(), expected - cfg.num_decoder_layers * (cfg.emb_dim + 1) * cfg.num_query_heads)

  def test_no_o_removes_only_o_gates_and_health(self):
    from flax import linen as nn
    from layers import rmt
    cfg = self._config('RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32')
    cfg.get_keys().update(dtype=jnp.float32, rmt_dynamic_o_enabled=False, rmt_block_scan=False)
    x = jax.random.normal(jax.random.key(801), (1,4,cfg.emb_dim))
    matrix = jax.random.normal(jax.random.key(802), (1,4,48,cfg.head_dim))
    layer = rmt.RMTLayer(cfg, mlp_dim=128)
    args = (jnp.ones((1,4), jnp.int32), jnp.arange(4)[None], True, 0)
    params = nn.unbox(layer.init(jax.random.key(803), matrix, *args)['params'])
    vo = params['dynamic_vo']
    self.assertEqual(vo['gate_kernel'].shape, (cfg.emb_dim, cfg.num_query_heads))
    self.assertEqual(vo['gate_bias'].shape, (cfg.num_query_heads,1))
    self.assertEqual(vo['key_kernel'].shape, (cfg.emb_dim, cfg.num_query_heads * 8))
    # Exercise nonzero dynamic keys, not just zero-initialized forward equivalence.
    vo['key_kernel'] = .01 * jax.random.normal(jax.random.key(804), vo['key_kernel'].shape)
    def forward(p):
      (out, _), aux = layer.apply({'params':p}, matrix, *args, mutable=['intermediates'])
      return jnp.mean(out**2), aux
    (loss, aux), grad = jax.value_and_grad(forward, has_aux=True)(params)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad)))
    health = aux['intermediates']['rmt_dynamic_health'][0]
    for name in ('o_dynamic_rms', 'o_ratio', 'o_gate_mean', 'o_gate_frac_gt_050'):
      self.assertEqual(float(health[rmt.RMT_DYNAMIC_HEALTH_NAMES.index(name)]), 0.)
    self.assertGreater(float(health[rmt.RMT_DYNAMIC_HEALTH_NAMES.index('v_dynamic_rms')]), 0.)
    self.assertIn('attn_write_key', params)
    self.assertIn('dynamic_attn_write', params)


if __name__ == '__main__':
  absltest.main()

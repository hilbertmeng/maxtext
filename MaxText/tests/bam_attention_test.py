"""Focused tests for BAM runtime read-key transforms."""

from absl.testing import absltest
from pathlib import Path
import tempfile
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from layers import initializers
from layers import normalizations

from layers.attentions import (
    BamAttention,
    GroupedRMSNorm,
    _attention_op,
    _bam_fetch_op,
    _bam_fetch_route_sums,
    _dynamic_bam_fetch_mix_weights,
    _fit_bam_read_to_head,
    _fetched_read_gate_bin_stats,
    _BamReadArm,
    _packed_local_arms_init,
    _packed_local_layout,
    _pack_fetched_bam_heads,
    _paired_parameter_init,
    _split_read_keys,
    _transform_bam_read_key,
    _update_bam_matrix,
    bam_read,
    factorized_head_bam_read,
)

_RMS_EPSILON = normalizations.DEFAULT_RMS_EPSILON


# ---- kwargs -> _BamReadArm adapters: the tests below spell every read setting
# out explicitly; production reads carry the same settings on the arm. ----

def _kw_arm(k_dim, v_dim, *, num_heads=1, key_mode='none', key_scale=1.0,
            rms_epsilon=_RMS_EPSILON, rms_statistics_dtype=jnp.float32,
            gate_activation=jax.nn.sigmoid, implementation='mul_reduce_btn',
            read_side='both', rank=1, rank_routing='legacy',
            second_implementation='mul_reduce', gram_implementation='mul_reduce',
            gram_statistics_dtype=jnp.float32):
  return _BamReadArm(
      name='t', k_dim=k_dim, v_dim=v_dim, num_heads=num_heads, read_side=read_side,
      rank=rank, rank_routing=rank_routing, key_mode=key_mode, key_scale=key_scale,
      rms_epsilon=rms_epsilon, rms_statistics_dtype=rms_statistics_dtype,
      gate_activation=gate_activation, implementation=implementation,
      second_implementation=second_implementation,
      gram_implementation=gram_implementation,
      gram_statistics_dtype=gram_statistics_dtype)


def _kw_project(x, W_R):
  return W_R(x) if callable(W_R) else jnp.broadcast_to(W_R, x.shape[:-1] + W_R.shape)


def _kw_transform(r, mode='none', scale=1.0, *, rms_epsilon, gate_logits=None, **kw):
  arm = _kw_arm(1, 1, key_mode=mode, key_scale=scale, rms_epsilon=rms_epsilon, **kw)
  return _transform_bam_read_key(r, arm, gate_logits)


def _kw_split_keys(row_width, x, W_R, *, key_gate_logits=None, **kw):
  """Legacy (raw_row, raw_col, r_row, r_col) view of the projected keys."""
  key = _kw_project(x, W_R)
  arm = _kw_arm(row_width, key.shape[-1] - row_width, **kw)
  raw_row, raw_col = jnp.split(key, [row_width], axis=-1)
  return (raw_row, raw_col) + tuple(_split_read_keys(key, arm, key_gate_logits))


def _kw_bam_read(M, x, W_R, *, key_gate_logits=None, return_sides=True, **kw):
  Mr = M[1] if isinstance(M, tuple) else M
  key = _kw_project(x, W_R)
  arm = _kw_arm(Mr.shape[-2], key.shape[-1] - Mr.shape[-2], **kw)
  y = bam_read(M, key, arm, gate_logits=key_gate_logits)
  return y if return_sides else jnp.concatenate(y, axis=-1)


def _kw_factorized_read(M, x, W_R, W_head_mix, *, key_gate_logits=None,
                        v_projection=None, basis_cache=None,
                        return_rank_gate=False, return_sides=True, **kw):
  key, mix = _kw_project(x, W_R), _kw_project(x, W_head_mix)
  arm = _kw_arm(M.shape[-2], M.shape[-1], num_heads=mix.shape[-3],
                rank=mix.shape[-1], **{k: v for k, v in kw.items() if k != 'rank'})
  result = factorized_head_bam_read(
      M, key, mix, arm, gate_logits=key_gate_logits, v_projection=v_projection,
      basis_cache=basis_cache, return_rank_gate=return_rank_gate)
  if return_sides:
    return result
  sides, gate = result if return_rank_gate else (result, None)
  joined = jnp.concatenate(sides, axis=-1)
  return (joined, gate) if return_rank_gate else joined


def _factorized_read_joined(*args, **kwargs):
  """Exercise default side tuples while retaining joined reference assertions."""
  return _kw_factorized_read(*args, return_sides=False, **kwargs)


class BamReadKeyTransformTest(absltest.TestCase):
  def test_compress_m_preserves_full_state_and_matches_v_projection(self):
    from types import SimpleNamespace
    state = jnp.arange(120, dtype=jnp.float32).reshape(1, 2, 5, 12)
    projection = jax.random.normal(jax.random.key(7), (12, 3))
    receiver = SimpleNamespace(_abs_v_dim=3, abs_v_cache_projection=projection)
    compress = lambda m: BamAttention._compress_m.__wrapped__(receiver, m)
    reference = lambda m: m @ projection
    np.testing.assert_array_equal(compress(state), reference(state))
    np.testing.assert_array_equal(jax.grad(lambda m: compress(m).sum())(state),
                                  jax.grad(lambda m: reference(m).sum())(state))
    receiver._abs_v_dim = None
    self.assertIs(compress(state), state)

  def test_fixed_write_matches_explicit_outer_product(self):
    from types import SimpleNamespace
    norm = lambda x: normalizations.rms_norm(x, dtype=x.dtype, epsilon=1e-6)
    x = jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 6) / 10
    output = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 2, 4) / 10
    state = jnp.ones((1, 2, 2, 3), jnp.float32)
    projection = lambda x: x.reshape(1, 2, 2, 3) + .1
    receiver = SimpleNamespace(
        config=SimpleNamespace(bam_sqrt_n_scale=False, bam_lambda_decay=1.),
        _force_activation_dtype=False, bam_k=2, _write_v_bottleneck_dim=None,
        _write_data=lambda o, x: o[..., :2],
        P_loc=projection, gw_b0=jnp.zeros((2,)), W_gw=lambda x: jnp.zeros((1, 2, 2)),
        _write_data_rms=True, write_data_norm=norm, write_address_norm=norm)
    expected = state + .5 * jnp.einsum('btnk,btnv->btkv', norm(output[..., :2]), norm(projection(x)))
    for implementation in ('dot', 'mul_reduce'):
      receiver._write_outer_implementation = implementation
      actual, gate = BamAttention._write.__wrapped__(receiver, output, x, state)
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(gate, jnp.full((1, 2, 2), .5))

  def test_softplus_read_gate_matched_opening_and_unbounded_output(self):
    r = jnp.array([[1., -2., 3.]], dtype=jnp.float32)
    p = jnp.asarray(.005)
    sigmoid_bias = jnp.log(p / (1 - p))
    softplus_bias = jnp.log(jnp.expm1(p))
    def read(logit, activation):
      return _kw_transform(
          r, 'rms_gate', 2., rms_epsilon=_RMS_EPSILON,
          gate_logits=logit, gate_activation=activation)
    np.testing.assert_allclose(
        read(sigmoid_bias, jax.nn.sigmoid),
        read(softplus_bias, jax.nn.softplus), rtol=2e-6, atol=1e-8)
    self.assertGreater(float(jnp.linalg.norm(read(4., jax.nn.softplus))),
                       4 * float(jnp.linalg.norm(read(4., jax.nn.sigmoid))))


  def test_clean_gelu_full_module_initializes_and_records_route(self):
    self._check_mix_full_module('BamLlama2MediumV2C256ScanAotCleanGeluAlphaMix')

  def test_scale_only_full_module_initializes_and_records_route(self):
    self._check_mix_full_module('BamLlama2MediumV2C256ScanAotOldMixScaleOnly')

  def _check_mix_full_module(self, exp_class):
    import max_utils
    import pyconfig
    from flax.traverse_util import flatten_dict

    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    (Path(output.name) / 'test-clean-gelu').mkdir()
    cfg = pyconfig.initialize(
        [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
        exp_class=exp_class,
        run_name='test-clean-gelu', enable_checkpointing=False,
        base_output_directory=output.name + '/',
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        base_emb_dim=128, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=2, base_mlp_dim=256, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8,
        query_chunk_size=4, per_device_batch_size=1.0)
    cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    attention = BamAttention(
        config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
        attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        layer_mode='local_qk+full', attention_type=cfg.attention_type)
    x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
    matrix = jnp.ones((1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    variables = attention.init(
        {'params': jax.random.key(2), 'aqt': jax.random.key(3)},
        *args, M_in=matrix, deterministic=True, layer_index=1)
    (y, m), updates = attention.apply(
        variables, *args, M_in=matrix, deterministic=True, layer_index=1,
        mutable=['intermediates'])
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(m.shape, matrix.shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
    paths = ['/'.join(p) for p in flatten_dict(variables['params'])]
    self.assertIn('fetch_mix_scale', paths)
    self.assertIn('fetch_route_sums', updates['intermediates'])
    self.assertIn('fetch_mix_scale', updates['intermediates'])

  def test_concat_gate_recalibration_preserves_read_amplitude(self):
    import dataclasses
    import math
    matrix = jax.random.normal(jax.random.key(71), (1,4,8,8))
    key = jax.random.normal(jax.random.key(72), (1,4,4,8))
    mix = jax.random.normal(jax.random.key(73), (1,4,2,1,4))
    for routing, scale in [('legacy',2.), ('effective_key',2.), ('head_gate_r',1.)]:
      arm = _BamReadArm(name='q', k_dim=8, v_dim=8, num_heads=2, rank=4,
                       read_side='col', prune_row=True, rank_routing=routing,
                       key_scale=scale)
      def read(opening, factor):
        logits = jnp.full((1,4)+arm.gate_shape, math.log(opening/(1-opening)))
        return factorized_head_bam_read(
            matrix, key, mix, dataclasses.replace(arm,key_scale=scale*factor),
            gate_logits=logits)[0]
      np.testing.assert_allclose(read(.005,1.), read(.05,.1), rtol=2e-6, atol=1e-8)

  def test_local_vo_shared_read_prunes_unused_parameters_and_reuses_one_read(self):
    import max_utils
    import pyconfig
    from flax.core import unfreeze
    for donor, suffix in (('local_v', 'Rank4'), ('local_o', 'C8')):
      name = 'BamMediumIndependentLLFQKConcatStaticLocalVOShared' + suffix + 'MLPPerLayer'
      with self.subTest(donor=donor), tempfile.TemporaryDirectory() as out:
        Path(out, 'shared').mkdir()
        cfg = pyconfig.initialize(
            [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
            exp_class=name, run_name='shared', enable_checkpointing=False,
            base_output_directory=out+'/', jax_cache_dir='', log_config=False,
            dataset_type='synthetic', base_emb_dim=128, base_num_query_heads=2,
            base_num_kv_heads=2, head_dim=64, max_target_length=8,
            max_prefill_predict_length=8, query_chunk_size=4, per_device_batch_size=1.)
        cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
            max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
            attention_kernel='dot_product_chunk', dtype=cfg.dtype,
            layer_mode='local_qk+local_o', read_side='col', attention_type=cfg.attention_type)
        x = jax.random.normal(jax.random.key(141), (1,8,128), dtype=cfg.dtype)
        m = jax.random.normal(jax.random.key(142), (1,8,32,32), dtype=cfg.dtype)
        args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
        kw = dict(M_in=m, deterministic=True, layer_index=1)
        params = unfreeze(module.init({'params':jax.random.key(143)}, *args, **kw)['params'])
        if donor == 'local_v':
          self.assertNotIn('W_R', params)
          self.assertNotIn('W_R_gate', params)
          self.assertNotIn('abs_v_cache_projection', params)
          self.assertIn('W_lv_bias', params)
          key_name = 'W_local_packed'
        else:
          self.assertNotIn('W_lv_bias', params)
          self.assertNotIn('W_lv_gate_b0', params)
          self.assertIn('W_R', params)
          self.assertIn('abs_v_cache_projection', params)
          key_name = 'W_R'
        leaf = params[key_name]['kernel']
        params[key_name]['kernel'] = leaf.replace(value=.1*jax.random.normal(
            jax.random.key(144), leaf.value.shape, leaf.value.dtype))
        (y, m_out), collections = module.apply({'params':params}, *args, **kw,
            capture_intermediates=lambda mod, method: method in ('_shared_local_vo', '_read_local', '_read_fetched_m'),
            mutable=['intermediates'])
        c = collections['intermediates']
        self.assertLen(c['_shared_local_vo'], 1)
        self.assertLen(c['_read_local'], 3 if donor == 'local_v' else 2)
        self.assertEqual(len(c.get('_read_fetched_m', ())), 0 if donor == 'local_v' else 1)
        np.testing.assert_array_equal(c['concat_local_v_gate'][0], c['concat_local_o_gate'][0])
        shared = c['_shared_local_vo'][0]
        self.assertGreater(float(jnp.linalg.norm(shared.astype(jnp.float32))), 0.)
        np.testing.assert_array_equal(shared[..., 32:], jnp.zeros_like(shared[..., 32:]))
        grad = jax.grad(lambda p: sum(jnp.mean(z.astype(jnp.float32)**2)
             for z in module.apply({'params':p}, *args, **kw)))(params)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(z))) for z in jax.tree.leaves(grad)))
        self.assertGreater(float(jnp.linalg.norm(grad[key_name]['kernel'].value.astype(jnp.float32))), 0.)
        # Sharing only affects L: F retains the original full V and fetched read.
        fp = module.clone(layer_mode='local_qk+full').init({'params':jax.random.key(143)}, *args, **kw)['params']
        self.assertIn('W_R', fp)
        self.assertEqual(fp['value']['kernel'].value.shape, (128,2,64))

  def test_shared_c8_k64_keeps_qk32_and_expands_vo_write(self):
    import max_utils
    import pyconfig
    from flax.core import unfreeze
    results = []
    for width, qk_width, suffix in ((32, 32, ''), (64, 32, 'K64Truncate'),
                                    (64, 48, 'K64QK48Truncate')):
      with tempfile.TemporaryDirectory() as out:
        Path(out, 'k64').mkdir()
        cfg = pyconfig.initialize(
            [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
            exp_class='BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8' + suffix + 'MLPPerLayer',
            run_name='k64', enable_checkpointing=False, base_output_directory=out+'/',
            jax_cache_dir='', log_config=False, dataset_type='synthetic',
            base_emb_dim=128, base_num_query_heads=2, base_num_kv_heads=2,
            head_dim=64, max_target_length=8, max_prefill_predict_length=8,
            query_chunk_size=4, per_device_batch_size=1.)
        cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
            head_dim=64, bam_k=width, bam_v=32, max_target_length=8,
            max_prefill_predict_length=8, mesh=mesh, attention_kernel='dot_product_chunk',
            dtype=cfg.dtype, layer_mode='local_qk+local_o', read_side='col',
            attention_type=cfg.attention_type)
        x = jax.random.normal(jax.random.key(201), (1,8,128), dtype=cfg.dtype)
        full_m = jax.random.normal(jax.random.key(202), (1,8,64,32), dtype=cfg.dtype)
        m = full_m[..., :width, :]
        args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
        kw = dict(M_in=m, deterministic=True, layer_index=1)
        params = unfreeze(module.init({'params':jax.random.key(203)}, *args, **kw)['params'])
        self.assertEqual(params['query']['kernel'].value.shape, (128,2,64-qk_width))
        self.assertEqual(params['key']['kernel'].value.shape, (128,2,64-qk_width))
        self.assertEqual(params['value']['kernel'].value.shape, (128,2,64))
        # Exercise nonzero dynamic AND static reads, not their zero-init special case.
        leaf = params['W_R']['kernel']
        params['W_R']['kernel'] = leaf.replace(value=.1*jax.random.normal(
            jax.random.key(204), leaf.value.shape, leaf.value.dtype))
        for arm in ('q','k'):
          leaf = params['static_'+arm+'_key']
          params['static_'+arm+'_key'] = leaf.replace(value=.1*jax.random.normal(
              jax.random.key(205 if arm == 'q' else 206), leaf.value.shape, leaf.value.dtype))
        (y, mout), c = module.apply({'params':params}, *args, **kw,
            capture_intermediates=lambda mod, method: method in ('_add_local_qk', '_shared_local_vo'),
            mutable=['intermediates'])
        c = c['intermediates']
        self.assertEqual(mout.shape, m.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
        vo = c['_shared_local_vo'][0]
        self.assertGreater(float(jnp.linalg.norm(vo[..., :width].astype('float32'))), 0.)
        if width == 64:
          self.assertGreater(float(jnp.linalg.norm(vo[..., 32:].astype('float32'))), 0.)
        else:
          np.testing.assert_array_equal(vo[..., 32:], jnp.zeros_like(vo[..., 32:]))
        o = jax.random.normal(jax.random.key(207), (1,8,2,64), dtype=cfg.dtype)
        written = module.apply({'params':params}, o, x, method=module._write_data)
        np.testing.assert_array_equal(written, o[..., :width])
        results.append((params,c['_add_local_qk'][0],vo))
        if width == 64:
          # QK cannot consume the extra K coordinates; both V and O can.
          changed = m.at[...,qk_width:,:].multiply(3)
          _, c2 = module.apply({'params':params}, *args, **dict(kw,M_in=changed),
              capture_intermediates=lambda mod, method: method in ('_add_local_qk', '_shared_local_vo'),
              mutable=['intermediates'])
          for a,b in zip(c['_add_local_qk'][0],c2['intermediates']['_add_local_qk'][0]):
            np.testing.assert_array_equal(a,b)
          grad = jax.grad(lambda p: jnp.mean(module.apply({'params':p},*args,**kw)[0].astype('float32')**2))(params)
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(z))) for z in jax.tree.leaves(grad)))
          for arm in ('q','k'):
            self.assertGreater(float(jnp.linalg.norm(grad['static_'+arm+'_key'].value.astype('float32'))),0.)
          self.assertGreater(float(jnp.linalg.norm(grad['W_R']['kernel'].value.astype('float32'))),0.)
          # FetchedO must also fit a full64 column without a phantom row tail.
          fetched = module.clone(layer_mode='local_qk+full')
          fp = unfreeze(fetched.init({'params':jax.random.key(203)},*args,**kw)['params'])
          fp['W_R'] = params['W_R']
          (fy,fm), fc = fetched.apply({'params':fp},*args,**kw,
              capture_intermediates=lambda mod, method: method == '_read_fetched_m', mutable=['intermediates'])
          self.assertEqual(fm.shape,m.shape)
          self.assertTrue(bool(jnp.all(jnp.isfinite(fy))))
          self.assertGreater(float(jnp.linalg.norm(fc['intermediates']['_read_fetched_m'][0][0][...,32:].astype('float32'))),0.)
    # Identical parameter tree/initialization and exactly the same QK, including RoPE.
    self.assertEqual(jax.tree.structure(results[0][0]),jax.tree.structure(results[1][0]))
    for a,b in zip(jax.tree.leaves(results[0][0]),jax.tree.leaves(results[1][0])):
      np.testing.assert_array_equal(a,b)
    for a,b in zip(results[0][1],results[1][1]):
      np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(results[0][2][...,:32],results[1][2][...,:32])

  def test_shared_c8_independent_gates_initialization_and_separate_gradients(self):
    self._check_shared_c8_independent_gates(32, 32)

  def test_shared_c8_independent_gates_k64(self):
    for qk_width in (32, 48):
      with self.subTest(qk_width=qk_width):
        self._check_shared_c8_independent_gates(64, qk_width)

  def _check_shared_c8_independent_gates(self, k_dim, qk_width):
    import copy
    import max_utils
    import pyconfig
    from flax.core import unfreeze
    with tempfile.TemporaryDirectory() as out:
      Path(out, 'gates').mkdir()
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
          exp_class='BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8MLPPerLayer',
          run_name='gates', enable_checkpointing=False, base_output_directory=out+'/',
          jax_cache_dir='', log_config=False, dataset_type='synthetic', base_emb_dim=128,
          base_num_query_heads=2, base_num_kv_heads=2, head_dim=64,
          max_target_length=8, max_prefill_predict_length=8, query_chunk_size=4,
          per_device_batch_size=1.)
      cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
      cfg.get_keys()['bam_k'] = k_dim
      cfg.get_keys()['bam_local_qk_col_output_dim'] = qk_width
      cfg.get_keys()['bam_partial_rope_nope_dim'] = qk_width
      from types import SimpleNamespace
      independent = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys())))
      independent.get_keys()['bam_local_vo_independent_gates'] = True
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      parent = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64, bam_k=k_dim,
          max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode='local_qk+local_o', read_side='col', attention_type=cfg.attention_type)
      child = parent.clone(config=independent)
      x = jax.random.normal(jax.random.key(171), (1,8,128), dtype=cfg.dtype)
      m = jax.random.normal(jax.random.key(172), (1,8,k_dim,32), dtype=cfg.dtype)
      args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
      kw = dict(M_in=m, deterministic=True, layer_index=1)
      init = lambda mod: unfreeze(mod.init({'params':jax.random.key(173)}, *args, **kw)['params'])
      old, new = init(parent), init(child)
      self.assertEqual(set(new)-set(old), {'W_lv_gate', 'W_lv_gate_b0'})
      for key in old:
        for a,b in zip(jax.tree.leaves(old[key]),jax.tree.leaves(new[key])):
          np.testing.assert_array_equal(a,b)
      self.assertEqual(sum(z.size for z in jax.tree.leaves(new))-sum(z.size for z in jax.tree.leaves(old)),258)
      for a,b in zip(parent.apply({'params':old},*args,**kw),child.apply({'params':new},*args,**kw)):
        np.testing.assert_array_equal(a,b)
      leaf = new['W_R']['kernel']
      direction = .1*jax.random.normal(jax.random.key(174),leaf.value.shape,leaf.value.dtype)
      old['W_R']['kernel'] = leaf.replace(value=direction)
      new['W_R']['kernel'] = leaf.replace(value=direction)
      def capture(params):
        result, collections = child.apply({'params':params}, *args, **kw,
            capture_intermediates=lambda mod,method: method in ('_independent_local_vo','_read_fetched_m'),
            mutable=['intermediates'])
        c = collections['intermediates']
        self.assertLen(c['_read_fetched_m'],1)
        return result,c['_independent_local_vo'][0],c
      result, (v,o), metrics = capture(new)
      np.testing.assert_array_equal(v,o)
      self.assertGreater(float(jnp.linalg.norm(v.astype(jnp.float32))),0.)
      baseline = parent.apply({'params':old},*args,**kw)
      for a,b in zip(baseline,result):
        relative = jnp.linalg.norm((a.astype(jnp.float32)-b.astype(jnp.float32)))/jnp.linalg.norm(a.astype(jnp.float32))
        self.assertLess(float(relative),.02)
      for name,index in [('W_lv_gate_b0',0),('W_R_gate_b0',1)]:
        changed = copy.deepcopy(new)
        b = changed[name];changed[name] = b.replace(value=b.value+1.)
        _,pair,c = capture(changed)
        np.testing.assert_array_equal(pair[1-index],(v,o)[1-index])
        self.assertGreater(float(jnp.linalg.norm((pair[index]-(v,o)[index]).astype(jnp.float32))),0.)
        self.assertGreater(float(c['concat_vo_gate_pair'][0][0]),0.)
      grad = jax.grad(lambda p: sum(jnp.mean(z.astype(jnp.float32)**2)
          for z in child.apply({'params':p},*args,**kw)))(new)
      for name in ('W_lv_gate','W_R_gate'):
        g = grad[name]['kernel'].value
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))
        self.assertGreater(float(jnp.linalg.norm(g.astype(jnp.float32))),0.)
      self.assertFalse(bool(jnp.allclose(grad['W_lv_gate']['kernel'].value.reshape(-1),
                                        grad['W_R_gate']['kernel'].value.reshape(-1))))
      fp = init(child.clone(layer_mode='local_qk+full'))
      self.assertNotIn('W_lv_gate',fp)

  def test_concat_shapes_write_seed_and_qk_learning(self):
    import max_utils
    import pyconfig
    from flax.core import unfreeze

    for qk, static in ((False, False), (True, False), (False, True), (True, True)):
      name = (('BamMediumIndependentLLFColOnlyQKConcatSharedRank4StaticMLPPerLayer' if qk
               else 'BamMediumIndependentLLFColOnlyVConcatStaticVOWriteMixMLPPerLayer') if static else
              ('BamMediumIndependentLLFColOnlyQKConcatSharedRank4MLPPerLayer' if qk
               else 'BamMediumIndependentLLFColOnlyVConcatMLPPerLayer'))
      with self.subTest(exp=name), tempfile.TemporaryDirectory() as out:
        Path(out, 'concat').mkdir()
        cfg = pyconfig.initialize(
            [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
            exp_class=name, run_name='concat', enable_checkpointing=False,
            base_output_directory=out+'/', jax_cache_dir='', log_config=False,
            dataset_type='synthetic', base_emb_dim=128, base_num_query_heads=2,
            base_num_kv_heads=2, head_dim=64, max_target_length=8,
            max_prefill_predict_length=8, query_chunk_size=4, per_device_batch_size=1.)
        cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        module = BamAttention(
            config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
            max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
            attention_kernel='dot_product_chunk', dtype=cfg.dtype,
            layer_mode='local_qk+local_o', read_side='col', attention_type=cfg.attention_type)
        x = jax.random.normal(jax.random.key(101), (1,8,128), dtype=cfg.dtype)
        m = jax.random.normal(jax.random.key(104), (1,8,32,32), dtype=cfg.dtype)
        args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
        kw = dict(M_in=m, deterministic=True, layer_index=1)
        variables = module.init({'params':jax.random.key(102)}, *args, **kw)
        params = unfreeze(variables['params'])
        _, updates = module.apply(variables, *args, **kw, mutable=['intermediates'])
        health = updates['intermediates']
        for arm in ('local_q', 'local_k', 'local_v', 'local_o'):
          self.assertAlmostEqual(float(health[f'concat_{arm}_gate'][0][0]), .05, delta=.001)
        if qk:
          self.assertGreater(float(health['concat_qk_scores'][0][0]), 0.)
        self.assertEqual(params['query']['kernel'].value.shape, (128,2,32 if qk else 64))
        self.assertEqual(params['key']['kernel'].value.shape, (128,2,32 if qk else 64))
        self.assertEqual(params['value']['kernel'].value.shape, (128,2,64 if qk else 32))
        # Fresh M must be seeded even when every BAM read is initially zero.
        y, m0 = module.apply(variables, *args, M_in=jnp.zeros_like(m), deterministic=True, layer_index=0)
        self.assertEqual(y.shape, x.shape)
        self.assertGreater(float(jnp.linalg.norm(m0.astype(jnp.float32))), 0.)
        grad = jax.grad(lambda p: jnp.sum(module.apply(
            {'params':p}, *args, **kw)[0].astype(jnp.float32)**2))(params)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(z))) for z in jax.tree.leaves(grad)))
        if static:
          for arm in (('q', 'k') if qk else ('vo',)):
            leaf = params['static_' + arm + '_key'].value
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))
            self.assertGreater(float(jnp.linalg.norm(grad['static_' + arm + '_key'].value.astype(jnp.float32))), 0.)
          if not qk:
            # Mixing is applied before normalization and must receive gradients.
            o = jnp.concatenate((jnp.ones((1,8,2,32)), jnp.full((1,8,2,32), 3.)), -1).astype(cfg.dtype)
            data = module.apply(variables, o, x, method=module._write_data)
            np.testing.assert_allclose(data.astype('float32'), 1.1, atol=.015)
            mix_grad = jax.grad(lambda p: jnp.sum(module.apply(
                {'params': p}, o, x, method=module._write_data).astype(jnp.float32)))(params)
            self.assertGreater(float(jnp.linalg.norm(mix_grad['write_mix']['kernel'].value.astype(jnp.float32))), 0.)
            self.assertAlmostEqual(float(health['concat_write_mix_gate'][0][0]), .05, delta=.001)
        if qk:
          # The shared basis is initialized nonzero and receives a genuine attention-loss gradient.
          basis_grad = grad['W_local_packed']['kernel'].value[:, :128]
          self.assertGreater(float(jnp.linalg.norm(basis_grad.astype(jnp.float32))), 0.)
          self.assertNotIn('W_lk_bias', params)
          def probe(bound):
            inputs = bound._local_inputs(x)
            cache = bound._shared_qk_basis(m, inputs)
            qc = bound._read_local('q', m, x, inputs, cache)
            kc = bound._read_local('k', m, x, inputs, cache)
            qref = bound._read_local('q', m, x, inputs)
            kref = bound._read_local('k', m, x, inputs)
            rot = jnp.ones((1,8,2,32), cfg.dtype)
            joined, _ = bound._add_local_qk(rot, rot, qc, kc)
            return qc, kc, qref, kref, joined, rot
          qc,kc,qr,kr,joined,rot = module.apply(variables, method=probe)
          np.testing.assert_allclose(qc.astype('float32'), qr.astype('float32'), rtol=.02, atol=2e-4)
          np.testing.assert_allclose(kc.astype('float32'), kr.astype('float32'), rtol=.02, atol=2e-4)
          np.testing.assert_array_equal(joined[..., :32], qc[..., :32])
          np.testing.assert_array_equal(joined[..., 32:], rot)
        # F layers retain full V in both experiments.
        fetched = module.clone(layer_mode='local_qk+full')
        fp = fetched.init({'params':jax.random.key(102)}, *args, **kw)['params']
        self.assertEqual(fp['value']['kernel'].value.shape, (128,2,64))

  def test_local_o_static_col_zero_equivalence_gradient_and_ungated_output(self):
    import copy
    import max_utils
    import pyconfig
    from flax.core import unfreeze

    with tempfile.TemporaryDirectory() as out:
      Path(out, 'static-col').mkdir()
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
          exp_class='BamMediumIndependentLLFMLPPerLayerColOnlyLocalOStaticCol',
          run_name='static-col', enable_checkpointing=False,
          base_output_directory=out+'/', jax_cache_dir='', log_config=False,
          dataset_type='synthetic', base_emb_dim=128, base_num_query_heads=2,
          base_num_kv_heads=2, head_dim=64, max_target_length=8,
          max_prefill_predict_length=8, query_chunk_size=4, per_device_batch_size=1.)
      cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      module = BamAttention(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
          max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode='local_qk+local_o', read_side='col', attention_type=cfg.attention_type)
      x = jax.random.normal(jax.random.key(1), (1,8,128), dtype=cfg.dtype)
      m = jax.random.normal(jax.random.key(4), (1,8,32,32), dtype=cfg.dtype)
      args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
      kw = dict(M_in=m, deterministic=True, layer_index=1)
      variables = module.init({'params':jax.random.key(2)}, *args, **kw)
      params = unfreeze(variables['params'])
      static = params['local_o_static_col_key']
      np.testing.assert_array_equal(static.value, jnp.zeros((32,2)))
      def output(p):
        return module.apply({'params':p}, *args, **kw)[0].astype(jnp.float32)
      y = output(params)
      # Disable just the new branch, retaining identical historical parameters.
      cfg.get_keys()['bam_local_o_static_col'] = False
      np.testing.assert_array_equal(y, output(params))
      cfg.get_keys()['bam_local_o_static_col'] = True
      grad = jax.grad(lambda p:jnp.sum(output(p)))(params)
      self.assertGreater(float(jnp.linalg.norm(grad['local_o_static_col_key'].value)), 0.)
      # With the dynamic O gate shut, a nonzero static key still changes output.
      off = copy.deepcopy(params)
      gate = off['W_R_gate_b0']
      off['W_R_gate_b0'] = gate.replace(value=jnp.full_like(gate.value,-100.))
      zero = output(off)
      off['local_o_static_col_key'] = static.replace(value=jnp.full_like(static.value,.01))
      self.assertGreater(float(jnp.linalg.norm(output(off)-zero)), 0.)
      fetched = module.clone(layer_mode='local_qk+full')
      fvars = fetched.init({'params':jax.random.key(2)}, *args, **kw)
      self.assertNotIn('local_o_static_col_key', fvars['params'])

  def test_scale_only_matches_legacy_initial_weights_and_has_scale_gradient(self):
    logits = jax.random.normal(jax.random.key(23), (2, 8, 16))
    for dtype in (jnp.float32, jnp.bfloat16):
      legacy = _dynamic_bam_fetch_mix_weights(
          logits, dtype, rms_epsilon=_RMS_EPSILON)
      scaled = lambda s: _dynamic_bam_fetch_mix_weights(
          logits, dtype, rms_epsilon=_RMS_EPSILON, scale=s)
      np.testing.assert_array_equal(legacy, scaled(jnp.asarray(.25)))
      gradient = jax.grad(lambda s: jnp.sum(scaled(s).astype(jnp.float32) ** 2))(.25)
      self.assertTrue(bool(jnp.isfinite(gradient)))
      self.assertGreater(float(gradient), 0.)

  def test_gelu_fetch_values_gradients_and_raw_negative_statistics(self):
    alpha = jnp.asarray([[[[.8, .2], [.3, .7]], [[.2, .8], [.7, .3]]]])
    weights = jnp.asarray([[[-.5, .2], [.4, -.7]]])
    matrix = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 2, 2)
    diagonal = jnp.eye(2, dtype=bool)
    def reference(w):
      raw = jnp.einsum('bnts,btn->bts', alpha, w)
      route = jnp.where(diagonal[None], 1, nn.gelu(raw))
      return jnp.einsum('bts,bskv->btkv', route, matrix)
    for implementation in ('dot', 'mul_reduce'):
      actual = lambda w: _bam_fetch_op(
          alpha, matrix, w, diagonal, diagonal_one=True,
          gelu_alpha=True, mix_implementation=implementation)
      np.testing.assert_allclose(actual(weights), reference(weights), rtol=1e-6)
      np.testing.assert_allclose(
          jax.grad(lambda w: jnp.sum(actual(w)))(weights),
          jax.grad(lambda w: jnp.sum(reference(w)))(weights), rtol=1e-6)
    _, raw, route = _bam_fetch_op(
        alpha, matrix, weights, diagonal, diagonal_one=True,
        gelu_alpha=True, return_route=True)
    stats = _bam_fetch_route_sums(raw, route, jnp.ones_like(route, bool), diagonal)
    self.assertEqual(float(stats[0]), 1.)
    self.assertEqual(float(stats[4]), 2.)

  def test_mha_control_initializes_without_bam_health_state(self):
    import max_utils
    import pyconfig
    from flax.traverse_util import flatten_dict

    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    (Path(output.name) / 'test-mha-control').mkdir()
    cfg = pyconfig.initialize(
        [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
        exp_class='BamMHALlama2MediumC256ScanAotCleanControl',
        run_name='test-mha-control', enable_checkpointing=False,
        base_output_directory=output.name + '/',
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        base_emb_dim=64, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=2, base_mlp_dim=128, head_dim=32,
        max_target_length=8, max_prefill_predict_length=8,
        query_chunk_size=4, per_device_batch_size=1.0)
    cfg.get_keys()['bam_record_fetched_read_amplitude_metrics'] = True
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x = jax.random.normal(jax.random.key(1), (1, 8, 64), dtype=cfg.dtype)
    positions = jnp.arange(8)[None]
    segments = jnp.ones((1, 8), dtype=jnp.int32)
    for kernel in ('dot_product', 'dot_product_chunk'):
      with self.subTest(kernel=kernel):
        attention = BamAttention(
            config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=32,
            max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
            attention_kernel=kernel, dtype=cfg.dtype, layer_mode='none',
            attention_type=cfg.attention_type)
        variables = attention.init(
            {'params': jax.random.key(2), 'aqt': jax.random.key(3)},
            x, x, positions, segments, deterministic=True)
        (y, matrix), updates = attention.apply(
            variables, x, x, positions, segments, deterministic=True,
            mutable=['intermediates'])
        self.assertEqual(y.shape, x.shape)
        self.assertIsNone(matrix)
        self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
        paths = ['/'.join(p) for p in flatten_dict(variables['params'])]
        self.assertFalse(any('W_R' in p or 'P_loc' in p for p in paths))
        self.assertNotIn('fetched_read_amplitude', updates.get('intermediates', {}))

  def test_fetched_read_gate_bins_cover_distribution_and_read_energy(self):
    probabilities = jnp.asarray((0.1, 0.3, 0.5, 0.7, 0.9))
    logits = jnp.log(probabilities / (1.0 - probabilities))
    gate_logits = jnp.stack((logits, logits), axis=-1)[None, :, None, :]
    y_bam = jnp.ones((1, 5, 1, 4), jnp.float32)
    y_std = 2.0 * jnp.ones_like(y_bam)
    stats = _fetched_read_gate_bin_stats(
        gate_logits, y_bam, y_std, 2, 2, 1, 4)
    self.assertEqual(stats.shape, (2, 5, 3))
    np.testing.assert_allclose(stats[..., 0], 0.2, rtol=1e-6)
    np.testing.assert_allclose(stats[..., 1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(stats[..., 2], 0.2, rtol=1e-6)

  def test_external_amplitude_preserves_gate_and_controls_total_energy(self):
    gate_opening = 0.005
    gate_logits = jnp.full(
        (1, 1, 1, 2), jnp.log(gate_opening / (1.0 - gate_opening)))
    projected = jnp.ones((1, 1, 1, 16))

    _, _, v2_row, v2_col = _kw_split_keys(
        8, jnp.zeros((1, 1, 1)), lambda _x: projected,
        rms_epsilon=_RMS_EPSILON, key_mode='rms_gate', key_scale=2.0,
        key_gate_logits=gate_logits)
    _, _, amplitude_row, amplitude_col = _kw_split_keys(
        8, jnp.zeros((1, 1, 1)), lambda _x: projected,
        rms_epsilon=_RMS_EPSILON, key_mode='rms_gate',
        key_scale=5.65685 / np.sqrt(8.0),
        key_gate_logits=gate_logits)
    np.testing.assert_allclose(amplitude_row, v2_row, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(amplitude_col, v2_col, rtol=2e-6, atol=2e-6)

    c8_total = (2.5 / np.sqrt(8.0)) * gate_opening * np.sqrt(8.0)
    c32_total = (2.5 / np.sqrt(32.0)) * gate_opening * np.sqrt(32.0)
    self.assertAlmostEqual(c8_total, 0.0125)
    self.assertAlmostEqual(c32_total, 0.0125)

  def test_clean_gate050_fixed_amplitude_matches_initial_read_and_key_jacobian(self):
    from exp import (
        BamLlama2MediumV2C256ScanAotCleanControl as Control,
        BamLlama2MediumV2C256ScanAotCleanGate050FixedAmplitude as Gate050,
    )

    self.assertEqual(Control.bam_read_key_scale, 2.0)
    self.assertEqual(Gate050.bam_read_gate_init, Control.bam_read_gate_init)
    self.assertEqual(Gate050.wd_mults, Control.wd_mults)
    self.assertTrue(Gate050.scan_layers)
    self.assertEqual(Gate050.checkpoint_period, 200)
    width = Gate050.bam_abs_v_compression_dim
    # Historical amplitude experiment is ledger-only; verify its scalar identity.
    scale = Gate050.bam_fetched_read_amplitude_init / np.sqrt(width)
    self.assertAlmostEqual(scale * Gate050.bam_fetched_read_gate_init,
                           2.0 * Control.bam_read_gate_init)

    def read_keys(projected, p, amplitude):
      logits = jnp.full((1, 1, 1, 2), np.log(p / (1 - p)))
      _, _, row, col = _kw_split_keys(
          32, jnp.zeros((1, 1, 1)), lambda _x: projected,
          rms_epsilon=Control.bam_read_key_epsilon, key_mode='rms_gate',
          key_scale=amplitude,
          key_gate_logits=logits)
      return jnp.concatenate((row, col), axis=-1)

    old = lambda x: read_keys(x, Control.bam_read_gate_init, 2.0)
    new = lambda x: read_keys(x, Gate050.bam_fetched_read_gate_init, scale)
    projected = jnp.linspace(-1, 1, 32 + width).reshape((1, 1, 1, -1))
    np.testing.assert_allclose(new(projected), old(projected), rtol=2e-6, atol=1e-8)
    zero = jnp.zeros_like(projected)
    np.testing.assert_allclose(jax.jacfwd(new)(zero), jax.jacfwd(old)(zero),
                               rtol=2e-6, atol=1e-8)

  def test_read_key_sides_use_common_scale(self):
    projected = jnp.asarray([[[[3.0, 4.0, 5.0, 12.0]]]])
    gate_logits = jnp.zeros((1, 1, 1, 2))
    _, _, row_key, col_key = _kw_split_keys(
        2, jnp.zeros((1, 1, 1)), lambda _x: projected,
        rms_epsilon=_RMS_EPSILON, key_mode='rms_gate', key_scale=0.02,
        key_gate_logits=gate_logits)
    expected_row = normalizations.rms_norm(
        projected[..., :2], dtype=projected.dtype, epsilon=_RMS_EPSILON)
    expected_col = normalizations.rms_norm(
        projected[..., 2:], dtype=projected.dtype, epsilon=_RMS_EPSILON)
    np.testing.assert_allclose(
        row_key, 0.01 * expected_row, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        col_key, 0.01 * expected_col, rtol=1e-6, atol=1e-6)

  def test_attention_op_matches_dense_and_chunk_values_and_gradients(self):
    b, t, n, d, chunk_size = 2, 6, 3, 4, 2
    keys = jax.random.split(jax.random.PRNGKey(79), 6)
    args = (
        jax.random.normal(keys[0], (b, t, n, d)),
        jax.random.normal(keys[1], (b, t, n, d)),
        jax.random.normal(keys[2], (b, t, n, d)),
    )
    segment_ids = jnp.asarray([[1, 1, 1, 2, 2, 2], [3, 3, 4, 4, 4, 4]])
    causal = jnp.arange(t)[None, :] <= jnp.arange(t)[:, None]
    valid = causal[None] & (
        segment_ids[:, :, None] == segment_ids[:, None, :])
    output_weight = jax.random.normal(keys[3], (b, t, n, d))
    alpha_weight = jax.random.normal(keys[4], (b, n, t, t))

    def dense(values):
      return _attention_op(
          *values, valid, attn_logits_soft_cap=3.0, float32_logits=True,
      )

    def reference(values):
      query, key, value = values
      logits = jnp.einsum('btnd,bsnd->bnts', query, key)
      logits = jnp.tanh(logits / 3.0) * 3.0
      logits = jnp.where(valid[:, None], logits, -1e30).astype(jnp.float32)
      alpha = jax.nn.softmax(logits, axis=-1)
      return jnp.einsum('bnts,bsnd->btnd', alpha, value), alpha

    expected_y, expected_alpha = reference(args)
    actual_y, actual_alpha = dense(args)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_alpha, expected_alpha, rtol=1e-6, atol=1e-6)
    chunk_outputs = []
    chunk_alphas = []
    for q0 in range(0, t, chunk_size):
      q1 = q0 + chunk_size
      chunk_y, chunk_alpha = _attention_op(
          args[0][:, q0:q1], args[1][:, :q1], args[2][:, :q1],
          valid[:, q0:q1, :q1], attn_logits_soft_cap=3.0,
          float32_logits=True)
      chunk_outputs.append(chunk_y)
      chunk_alphas.append(jnp.pad(
          chunk_alpha, ((0, 0), (0, 0), (0, 0), (0, t - q1))))
    np.testing.assert_allclose(
        jnp.concatenate(chunk_outputs, axis=1), expected_y,
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.concatenate(chunk_alphas, axis=2), expected_alpha,
        rtol=1e-6, atol=1e-6)

    def objective(function, values):
      y, alpha = function(values)
      return jnp.sum(y * output_weight) + jnp.sum(alpha * alpha_weight)

    expected_value, expected_grad = jax.value_and_grad(
        lambda values: objective(reference, values))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda values: objective(dense, values))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
    for got, expected in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  def test_bam_fetch_op_respects_diagonal_one_for_values_and_gradients(self):
    b, n, t, k, v = 2, 3, 5, 4, 6
    keys = jax.random.split(jax.random.PRNGKey(83), 4)
    args = (
        jax.nn.softmax(jax.random.normal(keys[0], (b, n, t, t)), axis=-1),
        jax.random.normal(keys[1], (b, t, n)),
        jax.random.normal(keys[2], (b, t, k, v)),
    )
    upstream = jax.random.normal(keys[3], (b, t, k, v))
    diagonal_mask = jnp.eye(t, dtype=bool)

    def reference(values, diagonal_one):
      alpha, mix_weights, fetch_state = values
      fetch_alpha = jnp.einsum('bnts,btn->bts', alpha, mix_weights)
      if diagonal_one:
        fetch_alpha = jnp.where(diagonal_mask[None], 1, fetch_alpha)
      return jnp.einsum('bts,bskv->btkv', fetch_alpha, fetch_state)

    def actual(values, diagonal_one, implementation):
      alpha, mix_weights, fetch_state = values
      return _bam_fetch_op(
          alpha, fetch_state, mix_weights, diagonal_mask,
          diagonal_one=diagonal_one, mix_implementation=implementation)

    for implementation in ('dot', 'mul_reduce'):
      for diagonal_one in (False, True):
        expected = reference(args, diagonal_one)
        got = actual(args, diagonal_one, implementation)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        fetched, raw_route, route = _bam_fetch_op(
            args[0], args[2], args[1], diagonal_mask,
            diagonal_one=diagonal_one, mix_implementation=implementation,
            return_route=True)
        np.testing.assert_array_equal(fetched, got)
        expected_route = jnp.einsum('bnts,btn->bts', args[0], args[1])
        if diagonal_one:
          expected_route = jnp.where(diagonal_mask[None], 1, expected_route)
        np.testing.assert_allclose(route, expected_route, rtol=1e-6, atol=1e-6)

        expected_value, expected_grad = jax.value_and_grad(
            lambda values: jnp.sum(
                reference(values, diagonal_one) * upstream))(args)
        actual_value, actual_grad = jax.value_and_grad(
            lambda values: jnp.sum(
                actual(values, diagonal_one, implementation) * upstream))(args)
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=1e-6, atol=1e-6)
        for got_item, expected_item in zip(actual_grad, expected_grad):
          np.testing.assert_allclose(
              got_item, expected_item, rtol=1e-6, atol=1e-6)

  def test_bam_read_head_mapping_pads_or_adapts_only_v_side(self):
    direct = jnp.arange(96, dtype=jnp.float32).reshape(1, 1, 1, 96)
    padded = _fit_bam_read_to_head(direct, bam_k=64, head_dim=128)
    np.testing.assert_array_equal(padded[..., :96], direct)
    np.testing.assert_array_equal(padded[..., 96:], 0)

    adapter = jnp.zeros((1, 64, 32), dtype=jnp.float32)
    adapter = adapter.at[0, :32].set(jnp.eye(32))
    wide = jnp.arange(96, dtype=jnp.float32).reshape(1, 1, 1, 96)
    adapted = _fit_bam_read_to_head(
        wide, bam_k=32, head_dim=64, v_adapter=adapter)
    np.testing.assert_array_equal(adapted[..., :32], wide[..., :32])
    np.testing.assert_array_equal(adapted[..., 32:], wide[..., 32:64])

  def test_bam_read_head_mapping_rejects_wide_v_without_adapter(self):
    with self.assertRaisesRegex(ValueError, 'without an adapter'):
      _fit_bam_read_to_head(
          jnp.zeros((1, 1, 1, 96)), bam_k=32, head_dim=64)

  def test_fetched_bam_heads_pack_adjacent_heads_then_pad_tail(self):
    read = jnp.arange(32 * 48, dtype=jnp.float32).reshape(1, 1, 32, 48)
    packed = _pack_fetched_bam_heads(
        read, num_query_heads=16, head_dim=128)
    self.assertEqual(packed.shape, (1, 1, 16, 128))
    np.testing.assert_array_equal(
        packed[..., :96], read.reshape(1, 1, 16, 96))
    np.testing.assert_array_equal(packed[..., 96:], 0)

  def test_fetched_bam_heads_reject_overwide_group(self):
    with self.assertRaisesRegex(ValueError, 'need 144 coordinates'):
      _pack_fetched_bam_heads(
          jnp.zeros((1, 1, 48, 48)),
          num_query_heads=16, head_dim=128)


  def test_dynamic_bam_fetch_rms_mix_weights(self):
    logits = jax.random.normal(jax.random.PRNGKey(13), (2, 4, 3))
    weights = _dynamic_bam_fetch_mix_weights(
        logits, jnp.bfloat16, rms_epsilon=_RMS_EPSILON)
    self.assertEqual(weights.shape, logits.shape)
    self.assertEqual(weights.dtype, jnp.bfloat16)

  def _local_arms(self, names, rank, routing, heads=4, k_dim=5, v_dim=7):
    return [
        _BamReadArm(
            name=name, rank=rank, k_dim=k_dim, v_dim=v_dim,
            rank_routing=routing, num_heads=heads, pre_rms_bias=True,
            read_side='both')
        for name in names]

  def test_packed_local_arms_preserve_segment_initializers(self):
    embed, heads, key_width = 64, 4, 12
    regular_init = initializers.nd_dense_init(
        1.0, 'fan_in', 'truncated_normal')
    arms = self._local_arms(('q', 'k'), rank=1, routing='legacy', heads=heads)
    layout, packed_width = _packed_local_layout(arms)
    self.assertEqual(packed_width, 2 * (key_width + 2 + 2 * heads))
    kernel = _packed_local_arms_init(regular_init, arms)(
        jax.random.PRNGKey(0), (embed, packed_width), jnp.float32)
    (q_key, q_gate, q_mix), (k_key, k_gate, k_mix) = (
        tuple(kernel[:, s] for s in segments) for segments in layout)
    for zero_segment in (q_key, q_gate, k_key, k_gate):
      np.testing.assert_array_equal(zero_segment, 0)
    self.assertGreater(float(jnp.linalg.norm(q_mix)), 0.0)
    self.assertGreater(float(jnp.linalg.norm(k_mix)), 0.0)
    self.assertFalse(np.array_equal(q_mix, k_mix))

  def test_packed_layout_serves_any_group_and_routing_mode(self):
    embed, heads, key_width, rank = 32, 4, 12, 2
    regular_init = initializers.nd_dense_init(
        1.0, 'fan_in', 'truncated_normal')
    for mode, gate_width in (
        ('legacy', 2), ('shared_rank_gate', 2 * rank),
        ('head_gate_n', 2 * heads), ('head_gate_r', 2 * heads),
        ('effective_key', 2 * heads)):
      for names in (('q', 'k'), ('v',), ('q', 'k', 'v')):
        arms = self._local_arms(names, rank=rank, routing=mode, heads=heads)
        layout, packed_width = _packed_local_layout(arms)
        mix_width = 2 * heads * rank
        self.assertEqual(
            packed_width, len(names) * (rank * key_width + gate_width + mix_width))
        kernel = _packed_local_arms_init(regular_init, arms)(
            jax.random.PRNGKey(100 + gate_width), (embed, packed_width), jnp.float32)
        for basis, gate, mix in layout:
          np.testing.assert_array_equal(kernel[:, basis], 0)
          np.testing.assert_array_equal(kernel[:, gate], 0)
          self.assertEqual(kernel[:, gate].shape[-1], gate_width)
          self.assertGreater(float(jnp.linalg.norm(kernel[:, mix])), 0.0)

  def test_paired_row_key_seeds_every_rank_slot_identically_across_arms(self):
    embed, heads, rank, k_dim, v_dim = 16, 2, 2, 5, 7
    regular_init = initializers.nd_dense_init(
        1.0, 'fan_in', 'truncated_normal')
    arms = self._local_arms(
        ('q', 'k'), rank=rank, routing='legacy', heads=heads, k_dim=k_dim, v_dim=v_dim)
    layout, packed_width = _packed_local_layout(arms)
    kernel = _packed_local_arms_init(regular_init, arms, paired_row_width=k_dim)(
        jax.random.PRNGKey(3), (embed, packed_width), jnp.float32)
    bases = [kernel[:, segments[0]].reshape(embed, rank, k_dim + v_dim)
             for segments in layout]
    np.testing.assert_array_equal(bases[0], bases[1])
    np.testing.assert_array_equal(bases[0][..., k_dim:], 0)
    self.assertGreater(float(jnp.linalg.norm(bases[0][:, 0, :k_dim])), 0.0)
    self.assertFalse(np.array_equal(bases[0][:, 0, :k_dim], bases[0][:, 1, :k_dim]))

  def test_paired_parameter_init_matches_single_init_in_both_slices(self):
    regular_init = initializers.nd_dense_init(
        1.0, 'fan_in', 'truncated_normal')
    key = jax.random.PRNGKey(17)
    expected = regular_init(key, (32, 8), jnp.float32)
    paired = _paired_parameter_init(regular_init)(
        key, (2, 32, 8), jnp.float32)
    np.testing.assert_array_equal(paired[0], expected)
    np.testing.assert_array_equal(paired[1], expected)

  def test_rmsnorm_supports_nontrailing_axis(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 2))
    norm = normalizations.RMSNorm(axis=-2)
    variables = norm.init(jax.random.PRNGKey(1), x)
    actual = norm.apply(variables, x)
    expected = normalizations.rms_norm(
        x, dtype=x.dtype, epsilon=_RMS_EPSILON, axis=-2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.mean(actual ** 2, axis=-2), jnp.ones((2, 3, 2)),
        rtol=1e-5, atol=1e-5)

  def test_factorized_head_read_implementations_match_gradients(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(59), 7)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, k + v)),
        jax.random.normal(random[3], (e, n, 2)),
        jax.random.normal(random[4], (b, t, 2)),
    )
    upstream = jax.random.normal(random[5], (b, t, n, k + v))

    def output(values, implementation):
      M, x, key_kernel, mix_kernel, gates = values
      projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)[..., None, :]
      head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)[..., None]
      return _factorized_read_joined(
          M, x, projection, head_projection, key_mode='rms_gate',
          key_scale=2.0, rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation=implementation)

    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'dot_btn') * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'mul_reduce_btn') * upstream))(args)
    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
    for got, expected in zip(actual_grad, reference_grad):
      np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_attention_op_allows_qk_only_width_expansion(self):
    b, t, n, d, extra = 2, 5, 3, 4, 2
    q, k, v = (
        jax.random.normal(key, (b, t, n, d))
        for key in jax.random.split(jax.random.PRNGKey(83), 3))
    valid = jnp.tril(jnp.ones((t, t), dtype=bool))[None]
    reference = _attention_op(q, k, v, valid)
    zeros = jnp.zeros(q.shape[:-1] + (extra,), q.dtype)
    expanded = _attention_op(
        jnp.concatenate((q, zeros), axis=-1),
        jnp.concatenate((k, zeros), axis=-1), v, valid)
    for got, expected in zip(expanded, reference):
      np.testing.assert_array_equal(got, expected)

  def test_factorized_head_read_projects_v_before_head_expansion(self):
    b, t, n, k, v, c, e = 2, 3, 4, 3, 5, 2, 7
    random = jax.random.split(jax.random.PRNGKey(89), 6)
    M = jax.random.normal(random[0], (b, t, k, v))
    x = jax.random.normal(random[1], (b, t, e))
    key_kernel = jax.random.normal(random[2], (e, k + v))
    mix_kernel = jax.random.normal(random[3], (e, n, 2))
    gates = jax.random.normal(random[4], (b, t, 2))
    projection = jax.random.normal(random[5], (v, c))
    key_fn = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)[..., None, :]
    mix_fn = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)[..., None]
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gates, implementation='mul_reduce_btn')
    full = _factorized_read_joined(M, x, key_fn, mix_fn, **kwargs)
    expected_u, expected_v = jnp.split(full, [k], axis=-1)
    expected = jnp.concatenate((
        expected_u, jnp.einsum('btnv,vc->btnc', expected_v, projection)),
        axis=-1)
    actual = _factorized_read_joined(
        M, x, key_fn, mix_fn, v_projection=projection, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

  def test_bam_read_implementations_match_values_and_gradients(self):
    b, t, n, f, k, v, e = 2, 3, 4, 2, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(41), 7)

    for fetched in (False, True):
      M_shape = (b, f, t, k, v) if fetched else (b, t, k, v)
      key_shape = (e, n, f, k + v) if fetched else (e, n, k + v)
      gate_shape = (b, t, n, f, 2) if fetched else (b, t, n, 2)
      args = (
          jax.random.normal(random[0], M_shape),
          jax.random.normal(random[1], (b, t, e)),
          jax.random.normal(random[2], key_shape),
          jax.random.normal(random[3], gate_shape),
      )
      upstream = jax.random.normal(random[4], (b, t, n, k + v))

      def output(values, implementation):
        M, x, kernel, gates = values
        projection = lambda z: jnp.einsum(
            'bte,enfD->btnfD', z, kernel) if fetched else jnp.einsum(
                'bte,enD->btnD', z, kernel)
        y = _kw_bam_read(
            M, x, projection, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
            implementation=implementation, return_sides=False)
        return y

      reference = output(args, 'dot_btn')
      reference_value, reference_grad = jax.value_and_grad(
          lambda z: jnp.sum(output(z, 'dot_btn') * upstream))(args)
      for implementation in ('mul_reduce_btn',):
        actual = output(args, implementation)
        actual_value, actual_grad = jax.value_and_grad(
            lambda z: jnp.sum(output(z, implementation) * upstream))(args)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
        for got, expected in zip(actual_grad, reference_grad):
          np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_bam_read_return_sides_matches_joined_output(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(73), 6)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, n, k + v)),
        jax.random.normal(random[3], (b, t, n, 2)),
    )
    upstream = jax.random.normal(random[4], (b, t, n, k + v))

    def output(values, return_sides):
      M, x, kernel, gates = values
      projection = lambda z: jnp.einsum('bte,end->btnd', z, kernel)
      y = _kw_bam_read(
          M, x, projection, key_mode='rms_gate', key_scale=2.0,
          rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation='mul_reduce_btn', return_sides=return_sides)
      return jnp.concatenate(y, axis=-1) if return_sides else y

    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, False) * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, True) * upstream))(args)
    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-6, atol=1e-6)
    for got, expected in zip(actual_grad, reference_grad):
      np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  def test_one_sided_bam_reads_keep_selected_half(self):
    b, t, n, f, k, v, e = 2, 3, 4, 2, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(67), 6)
    x = jax.random.normal(random[0], (b, t, e))
    gates = jax.random.normal(random[1], (b, t, n, f, 2))
    M = jax.random.normal(random[2], (b, f, t, k, v))
    kernel = jax.random.normal(random[3], (e, n, f, k + v))
    projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)

    for implementation in ('dot_btn', 'mul_reduce_btn'):
      outputs = {}
      for read_side in ('both', 'row', 'col'):
        y = _kw_bam_read(
            M, x, projection, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
            implementation=implementation, read_side=read_side, return_sides=False)
        outputs[read_side] = y
      np.testing.assert_array_equal(outputs['row'][..., :k], 0)
      np.testing.assert_allclose(
          outputs['row'][..., k:], outputs['both'][..., k:], rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(
          outputs['col'][..., :k], outputs['both'][..., :k], rtol=1e-5, atol=1e-5)
      np.testing.assert_array_equal(outputs['col'][..., k:], 0)

    local_M = jax.random.normal(random[4], (b, t, k, v))
    key_kernel = jax.random.normal(random[3], (e, k + v))
    mix_kernel = jax.random.normal(random[5], (e, n, 2))
    key_projection = lambda z: jnp.einsum('bte,eD->btD', z, key_kernel)[..., None, :]
    mix_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)[..., None]
    local_gates = gates[:, :, 0, 0]
    both = _factorized_read_joined(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn')
    row = _factorized_read_joined(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn', read_side='row')
    col = _factorized_read_joined(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn', read_side='col')
    np.testing.assert_array_equal(row[..., :k], 0)
    np.testing.assert_allclose(row[..., k:], both[..., k:], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(col[..., :k], both[..., :k], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(col[..., k:], 0)

  def test_single_fetch_axis_squeeze_matches_values_and_gradients(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(47), 5)
    args = (
        jax.random.normal(random[0], (b, 1, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, n, 1, k + v)),
        jax.random.normal(random[3], (b, t, n, 1, 2)),
    )
    upstream = jax.random.normal(random[4], (b, t, n, k + v))

    def output(values, squeeze):
      M, x, kernel, gates = values
      if squeeze:
        projection = lambda z: jnp.einsum('bte,enD->btnD', z, kernel[:, :, 0])
        return _kw_bam_read(
            M[:, 0], x, projection, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates[..., 0, :],
            implementation='dot_btn', return_sides=False)
      projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)
      return _kw_bam_read(
          M, x, projection, key_mode='rms_gate', key_scale=2.0,
          rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation='dot_btn', return_sides=False)

    expected = output(args, False)
    actual = output(args, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    expected_value, expected_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, False) * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, True) * upstream))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    for got, expected_grad_item in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got, expected_grad_item, rtol=2e-5, atol=2e-5)

  def test_factorized_head_read_matches_explicit_rank_one_keys(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    keys = jax.random.split(jax.random.PRNGKey(29), 6)
    M = jax.random.normal(keys[0], (b, t, k, v))
    x = jax.random.normal(keys[1], (b, t, e))
    key_kernel = jax.random.normal(keys[2], (e, k + v))
    mix_kernel = jax.random.normal(keys[3], (e, n, 2))
    gate_kernel = jax.random.normal(keys[4], (e, 2))
    gate_bias = jax.random.normal(keys[5], (2,))
    projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)[..., None, :]
    head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)[..., None]
    gate_logits = jnp.einsum('bte,er->btr', x, gate_kernel) + gate_bias
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gate_logits)

    actual = _factorized_read_joined(
        M, x, projection, head_projection, **kwargs)
    actual_mul = _factorized_read_joined(
        M, x, projection, head_projection, **kwargs,
        implementation='mul_reduce_btn')
    raw_row, raw_col = jnp.split(projection(x)[..., 0, :], [k], axis=-1)
    row_gate, col_gate = jnp.split(gate_logits, 2, axis=-1)
    row = _kw_transform(
        raw_row, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=row_gate)
    col = _kw_transform(
        raw_col, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=col_gate)
    raw_mix = head_projection(x)[..., 0]
    mix = normalizations.rms_norm(
        raw_mix, dtype=raw_mix.dtype, epsilon=_RMS_EPSILON, axis=-2)
    explicit_row = row[:, :, None, :] * mix[..., 0, None]
    explicit_col = col[:, :, None, :] * mix[..., 1, None]
    expected = jnp.concatenate([
        jnp.einsum('btkv,btnv->btnk', M, explicit_col),
        jnp.einsum('btkv,btnk->btnv', M, explicit_row),
    ], axis=-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_mul, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        jnp.mean(mix ** 2, axis=-2), jnp.ones((b, t, 2)),
        rtol=2e-3, atol=2e-3)

  def test_factorized_head_rank_two_dot_and_mul_reduce_match(self):
    b, t, n, rank, k, v, e = 2, 3, 4, 2, 3, 5, 7
    keys = jax.random.split(jax.random.PRNGKey(30), 6)
    M = jax.random.normal(keys[0], (b, t, k, v))
    x = jax.random.normal(keys[1], (b, t, e))
    key_kernel = jax.random.normal(keys[2], (e, rank, k + v))
    mix_kernel = jax.random.normal(keys[3], (e, n, 2, rank))
    gate_kernel = jax.random.normal(keys[4], (e, 2))
    gate_bias = jax.random.normal(keys[5], (2,))
    projection = lambda z: jnp.einsum('bte,erd->btrd', z, key_kernel)
    head_projection = lambda z: jnp.einsum(
        'bte,ensr->btnsr', z, mix_kernel)
    gate_logits = jnp.einsum('bte,es->bts', x, gate_kernel) + gate_bias
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gate_logits, implementation='mul_reduce_btn', rank=rank)

    dot = _factorized_read_joined(
        M, x, projection, head_projection,
        second_implementation='dot', **kwargs)
    mul = _factorized_read_joined(
        M, x, projection, head_projection,
        second_implementation='mul_reduce', **kwargs)
    np.testing.assert_allclose(dot, mul, rtol=2e-5, atol=2e-5)

    upstream = jax.random.normal(jax.random.PRNGKey(35), dot.shape)
    args = (M, x, key_kernel, mix_kernel, gate_kernel, gate_bias)

    def output(values, second_implementation):
      matrix, hidden, key_w, mix_w, gate_w, gate_b = values
      return _factorized_read_joined(
          matrix, hidden,
          lambda z: jnp.einsum('bte,erd->btrd', z, key_w),
          lambda z: jnp.einsum('bte,ensr->btnsr', z, mix_w),
          key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=jnp.einsum('bte,es->bts', hidden, gate_w) + gate_b,
          implementation='mul_reduce_btn', rank=rank,
          second_implementation=second_implementation)

    dot_value, dot_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'dot') * upstream))(args)
    mul_value, mul_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'mul_reduce') * upstream))(args)
    np.testing.assert_allclose(dot_value, mul_value, rtol=2e-5, atol=2e-5)
    for got, expected in zip(dot_grad, mul_grad):
      np.testing.assert_allclose(got, expected, rtol=3e-5, atol=3e-5)

  def test_factorized_head_shared_rank_gate_matches_explicit_read(self):
    b, t, n, rank, k, v, e = 2, 3, 4, 2, 3, 5, 7
    keys = jax.random.split(jax.random.PRNGKey(130), 6)
    M = jax.random.normal(keys[0], (b, t, k, v))
    x = jax.random.normal(keys[1], (b, t, e))
    key_kernel = jax.random.normal(keys[2], (e, rank, k + v))
    mix_kernel = jax.random.normal(keys[3], (e, n, 2, rank))
    gate_kernel = jax.random.normal(keys[4], (e, rank, 2))
    gate_bias = jax.random.normal(keys[5], (rank, 2))
    projection = lambda z: jnp.einsum('bte,erd->btrd', z, key_kernel)
    head_projection = lambda z: jnp.einsum(
        'bte,ensr->btnsr', z, mix_kernel)
    gate_logits = jnp.einsum('bte,ers->btrs', x, gate_kernel) + gate_bias
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gate_logits, implementation='mul_reduce_btn', rank=rank,
        rank_routing='shared_rank_gate', return_rank_gate=True)

    dot, dot_gate = _factorized_read_joined(
        M, x, projection, head_projection,
        second_implementation='dot', **kwargs)
    mul, mul_gate = _factorized_read_joined(
        M, x, projection, head_projection,
        second_implementation='mul_reduce', **kwargs)
    np.testing.assert_allclose(dot, mul, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(dot_gate, mul_gate, rtol=0, atol=0)
    expected_gate = jnp.broadcast_to(
        jnp.swapaxes(jax.nn.sigmoid(gate_logits), -1, -2)[..., None, :, :],
        (b, t, n, 2, rank))
    np.testing.assert_allclose(dot_gate, expected_gate, rtol=1e-6, atol=1e-6)


  def test_factorized_head_read_zero_key_starts_dormant_but_has_key_gradient(self):
    b, t, n, k, v, e = 1, 3, 4, 3, 5, 7
    M = jax.random.normal(jax.random.PRNGKey(31), (b, t, k, v))
    x = jax.random.normal(jax.random.PRNGKey(32), (b, t, e))
    mix_kernel = jax.random.normal(jax.random.PRNGKey(33), (e, n, 2))
    upstream = jax.random.normal(jax.random.PRNGKey(34), (b, t, n, k + v))
    head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)[..., None]
    gate_init = np.sqrt(_RMS_EPSILON) / 2.0
    gate_logits = jnp.full((b, t, 2), np.log(gate_init / (1.0 - gate_init)))

    def objective(key_kernel):
      projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)[..., None, :]
      y = _factorized_read_joined(
          M, x, projection, head_projection, key_mode='rms_gate',
          key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gate_logits)
      return jnp.sum(y * upstream), y

    (value, y), grad = jax.value_and_grad(objective, has_aux=True)(
        jnp.zeros((e, k + v)))
    self.assertEqual(float(value), 0.0)
    np.testing.assert_array_equal(y, jnp.zeros_like(y))
    self.assertGreater(float(jnp.linalg.norm(grad)), 0.0)

  def test_combined_shared_read_matches_separate_value_and_gradients(self):
    """Read(F, r) + Read(L, r) == Read(F + L, r), including shared-key grads."""
    b, t, n, k, v, e = 2, 5, 3, 4, 4, 7
    keys = jax.random.split(jax.random.PRNGKey(17), 7)

    def make_inputs(dtype):
      return (
          jax.random.normal(keys[0], (b, t, k, v), dtype=dtype),
          jax.random.normal(keys[1], (b, 1, t, t), dtype=jnp.float32),
          jax.random.normal(keys[2], (b, t, e), dtype=dtype),
          jax.random.normal(keys[3], (e, n, 1, k + v), dtype=jnp.float32),
          jax.random.normal(keys[4], (e, n, 1, 2), dtype=jnp.float32),
      )

    upstream = jax.random.normal(
        keys[5], (b, t, n, k + v), dtype=jnp.bfloat16).astype(jnp.float32)
    gate_bias = jax.random.normal(keys[6], (n, 1, 2), dtype=jnp.float32)

    def read_output(args, combine):
      Mh, fetch_alpha, x, read_kernel, gate_kernel = args
      compute_dtype = x.dtype
      projection = lambda z: jnp.einsum(
          'bte,enfD->btnfD', z.astype(compute_dtype), read_kernel.astype(compute_dtype))
      gate_logits = jnp.einsum(
          'bte,enfg->btnfg', x, gate_kernel.astype(compute_dtype)) + gate_bias
      eye = jnp.eye(t, dtype=fetch_alpha.dtype)[None, None]
      offdiag_alpha = fetch_alpha * (1 - eye)
      routed_alpha = offdiag_alpha + eye if combine == 'diag_one' else offdiag_alpha
      Mbar = jnp.einsum('bfts,bskv->bftkv', routed_alpha, Mh)
      kwargs = dict(
          key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gate_logits, return_sides=False)
      if combine == 'diag_one':
        y = _kw_bam_read(Mbar, x, projection, **kwargs)
      elif combine:
        y = _kw_bam_read(Mbar + Mh[:, None], x, projection, **kwargs)
      else:
        y = _kw_bam_read(Mbar, x, projection, **kwargs)
        y += _kw_bam_read(
            Mh, x, lambda z: jnp.squeeze(projection(z), axis=-2),
            key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON,
            key_gate_logits=jnp.squeeze(gate_logits, axis=-2), return_sides=False)
      return y

    def objective(args, combine):
      return jnp.sum(jnp.asarray(read_output(args, combine), jnp.float32) * upstream)

    backend = jax.default_backend()
    for dtype, cpu_relative_limit in (
        (jnp.float32, 2e-6),
        (jnp.bfloat16, 2e-2),
    ):
      # TPU's default dot precision may use reduced-precision products even for
      # float32 operands.  Keep this test diagnostic on TPU while retaining a
      # strict algebraic check on CPU.
      relative_limit = cpu_relative_limit if backend == 'cpu' else 2e-2
      args = make_inputs(dtype)
      old_output = np.asarray(jax.jit(lambda values: read_output(values, False))(args), np.float32)
      new_output = np.asarray(jax.jit(lambda values: read_output(values, True))(args), np.float32)
      output_diff = new_output - old_output
      output_relative_l2 = np.linalg.norm(output_diff) / max(np.linalg.norm(old_output), 1e-12)
      print(
          f'combined_read backend={backend} dtype={dtype} '
          f'output_rel_l2={output_relative_l2:.3e} '
          f'output_max_abs={np.max(np.abs(output_diff)):.3e}')
      self.assertLess(output_relative_l2, relative_limit)
      separate = jax.jit(jax.value_and_grad(lambda values: objective(values, False)))
      combined = jax.jit(jax.value_and_grad(lambda values: objective(values, True)))
      old_value, old_grads = separate(args)
      new_value, new_grads = combined(args)
      old_value_f = float(old_value)
      value_rel = abs(float(new_value) - old_value_f) / max(abs(old_value_f), 1e-12)
      print(f'combined_read dtype={dtype} value_rel={value_rel:.3e}')
      self.assertLess(value_rel, relative_limit)
      for index, (new_grad, old_grad) in enumerate(zip(new_grads, old_grads)):
        new_grad = np.asarray(new_grad, dtype=np.float32)
        old_grad = np.asarray(old_grad, dtype=np.float32)
        diff = new_grad - old_grad
        relative_l2 = np.linalg.norm(diff) / max(np.linalg.norm(old_grad), 1e-12)
        print(
            f'combined_read dtype={dtype} grad={index} '
            f'rel_l2={relative_l2:.3e} max_abs={np.max(np.abs(diff)):.3e}')
        self.assertLess(relative_l2, relative_limit)

      diag_output = np.asarray(
          jax.jit(lambda values: read_output(values, 'diag_one'))(args), np.float32)
      diag_output_diff = diag_output - new_output
      diag_output_relative_l2 = np.linalg.norm(diag_output_diff) / max(
          np.linalg.norm(new_output), 1e-12)
      print(
          f'diag_one backend={backend} dtype={dtype} '
          f'output_rel_l2_vs_add_local={diag_output_relative_l2:.3e} '
          f'output_max_abs={np.max(np.abs(diag_output_diff)):.3e}')
      self.assertLess(diag_output_relative_l2, relative_limit)
      diag_value, diag_grads = jax.jit(
          jax.value_and_grad(lambda values: objective(values, 'diag_one')))(args)
      diag_value_rel = abs(float(diag_value) - float(new_value)) / max(abs(float(new_value)), 1e-12)
      print(f'diag_one dtype={dtype} value_rel_vs_add_local={diag_value_rel:.3e}')
      self.assertLess(diag_value_rel, relative_limit)
      for index, (diag_grad, combined_grad) in enumerate(zip(diag_grads, new_grads)):
        diag_grad = np.asarray(diag_grad, dtype=np.float32)
        combined_grad = np.asarray(combined_grad, dtype=np.float32)
        diff = diag_grad - combined_grad
        relative_l2 = np.linalg.norm(diff) / max(np.linalg.norm(combined_grad), 1e-12)
        print(
            f'diag_one dtype={dtype} grad={index} '
            f'rel_l2_vs_add_local={relative_l2:.3e} max_abs={np.max(np.abs(diff)):.3e}')
        self.assertLess(relative_l2, relative_limit)

  def test_grouped_rmsnorm_has_independent_group_scales(self):
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)
    norm = GroupedRMSNorm(
        scale_shape=(3, 4), epsilon=_RMS_EPSILON, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None))
    variables = norm.init(jax.random.PRNGKey(0), x)
    scale_param = variables['params']['scale']
    self.assertEqual(scale_param.value.shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    scale = jnp.zeros((3, 4)).at[1].set(1.0)
    scaled = norm.apply({'params': {'scale': scale}}, x)
    np.testing.assert_allclose(scaled[:, 0], expected[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 1], 2.0 * expected[:, 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 2], expected[:, 2], rtol=1e-6, atol=1e-6)

  def test_grouped_rmsnorm_supports_independent_group_biases(self):
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)
    norm = GroupedRMSNorm(
        scale_shape=(3, 4), epsilon=_RMS_EPSILON, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None), use_bias=True)
    variables = norm.init(jax.random.PRNGKey(0), x)
    self.assertEqual(variables['params']['bias'].value.shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    bias = jnp.zeros((3, 4)).at[1].set(jnp.arange(4, dtype=jnp.float32))
    biased = norm.apply(
        {'params': {'scale': jnp.zeros((3, 4)), 'bias': bias}}, x)
    np.testing.assert_allclose(biased[:, 0], expected[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        biased[:, 1], expected[:, 1] + bias[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(biased[:, 2], expected[:, 2], rtol=1e-6, atol=1e-6)

  def test_grouped_rmsnorm_supports_bias_without_learned_scale(self):
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)
    norm = GroupedRMSNorm(
        scale_shape=(3, 4), epsilon=_RMS_EPSILON, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None),
        scale_init=None, use_bias=True)
    variables = norm.init(jax.random.PRNGKey(0), x)
    self.assertNotIn('scale', variables['params'])
    self.assertEqual(variables['params']['bias'].value.shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    bias = jnp.zeros((3, 4)).at[1].set(jnp.arange(4, dtype=jnp.float32))
    biased = norm.apply({'params': {'bias': bias}}, x)
    np.testing.assert_allclose(biased, expected + bias, rtol=1e-6, atol=1e-6)

  def test_constant_matrix_update_matches_existing_decay(self):
    M_in = jnp.arange(12, dtype=jnp.float32).reshape(1, 1, 3, 4)
    dM = jnp.ones_like(M_in)
    M_out = _update_bam_matrix(M_in, dM, 0.75)
    np.testing.assert_array_equal(M_out, 0.75 * M_in + dM)


  def test_rms_gate_bias_calibration_preserves_zero_jacobian(self):
    scale = 2.0
    initial_gate = np.sqrt(_RMS_EPSILON) / scale
    gate_logits = jnp.full((1,), np.log(initial_gate / (1.0 - initial_gate)))
    jacobian = jax.jacfwd(
        lambda z: _kw_transform(
            z, 'rms_gate', scale, rms_epsilon=_RMS_EPSILON,
            gate_logits=gate_logits))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-5, atol=1e-5)

  def test_rms_gate_has_requested_rms(self):
    scale = 2.0
    gate = 0.25
    gate_logits = jnp.full((1,), np.log(gate / (1.0 - gate)))
    transformed = _kw_transform(
        jnp.array([3.0, 4.0]), 'rms_gate', scale,
        rms_epsilon=_RMS_EPSILON, gate_logits=gate_logits)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    np.testing.assert_allclose(transformed_rms, scale * gate, rtol=1e-6, atol=1e-6)

  def test_rms_direction_omits_only_the_scalar_gate(self):
    r = jnp.array([3.0, 4.0], dtype=jnp.float32)
    direction = _kw_transform(
        r, 'rms', 2.0, rms_epsilon=_RMS_EPSILON)
    expected = normalizations.rms_norm(
        r, dtype=r.dtype, epsilon=_RMS_EPSILON)
    np.testing.assert_array_equal(direction, expected)



if __name__ == '__main__':
  absltest.main()

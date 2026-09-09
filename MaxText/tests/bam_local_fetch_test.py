"""LocalO/fetch experiments: module gradients and static two-layer scan semantics."""
from pathlib import Path
import tempfile
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
from layers.attentions import BamAttention, _transform_bam_read_key
from layers.fusion import BamLayerPair
import train
import train_compile


class LocalFetchTest(absltest.TestCase):
  def config(self, exp):
    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    (Path(output.name) / 'test').mkdir()
    cfg = pyconfig.initialize(
        [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
        exp_class=exp, run_name='test', enable_checkpointing=False,
        base_output_directory=output.name + '/', jax_cache_dir='',
        log_config=False, dataset_type='synthetic',
        base_emb_dim=128, base_num_query_heads=2, base_num_kv_heads=2,
        base_num_decoder_layers=4, base_mlp_dim=256, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8,
        query_chunk_size=4, per_device_batch_size=1.)
    cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
    cfg.get_keys()['bam_layer_modes'] = ['local_qk+local_o', 'local_qk+full'] * 2
    return cfg

  def test_local_modules_forward_and_gradients(self):
    for suffix in ('C8', 'C8LocalV', 'Full', 'FullLocalV', 'C8SharedRead', 'FullSharedRead', 'C8LocalVSharedRankGate'):
      with self.subTest(suffix=suffix):
        layout = 'Scan' if suffix in ('FullSharedRead', 'C8LocalVSharedRankGate') else 'NonScan'
        cfg = self.config('BamLlama2MediumV2C256LocalFetch' + suffix + layout)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        module = BamAttention(
            config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
            max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
            attention_kernel='dot_product_chunk', dtype=cfg.dtype,
            layer_mode='local_qk+local_o', attention_type=cfg.attention_type)
        x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
        m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), dtype=cfg.dtype)
        args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
        variables = module.init(
            {'params': jax.random.key(3), 'aqt': jax.random.key(4)},
            *args, M_in=m, deterministic=True, layer_index=2)
        def loss(params):
          y, next_m = module.apply({'params': params}, *args,
                                  M_in=m, deterministic=True, layer_index=2)
          return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(next_m.astype(jnp.float32)**2)
        grads = jax.grad(loss)(variables['params'])
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(grads)))
        paths = ['/'.join(p) for p in flatten_dict(variables['params'])]
        if suffix == 'C8LocalVSharedRankGate':
          self.assertEqual(variables['params']['W_lv_gate_b0'].value.shape, (4,))
        self.assertFalse(any('fetch_head_mix' in p for p in paths))
        self.assertEqual(any('W_local_v_packed' in p for p in paths), 'LocalV' in suffix)
        self.assertEqual(any('abs_v_cache_projection' in p for p in paths), suffix.startswith('C8'))
        wr = grads['W_R']['kernel']
        wr = wr.value if hasattr(wr, 'value') else wr
        self.assertGreater(float(jnp.linalg.norm(wr.astype(jnp.float32))), 0.)

  def test_pair_scan_is_two_static_layers_with_independent_parameters(self):
    cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalVScan')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    module = nn.scan(
        BamLayerPair,
        variable_axes={'params': cfg.param_scan_axis},
        split_rngs={'params': True, 'dropout': False},
        in_axes=(nn.broadcast,) * 10 + (0,), length=2,
        metadata_params={nn.PARTITION_NAME: 'layers'})(
            cfg, mesh, 8, all_global_attention=True)
    h = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
    m = jnp.zeros((1, 8, 32, 32), cfg.dtype)
    args = ((h, m), jnp.ones((1, 8), jnp.int32), jnp.arange(8)[None],
            jnp.ones((1, 8), jnp.int32), None, True, 'train', None,
            None, None, None, jnp.arange(2))
    variables = module.init({'params': jax.random.key(8), 'aqt': jax.random.key(9)}, *args)
    (y, final_m), _ = module.apply(variables, *args)
    self.assertEqual(y.shape, h.shape)
    self.assertEqual(final_m.shape, m.shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
    params = variables['params']
    self.assertEqual(set(params), {'local_0', 'fetch_1'})
    self.assertIn('W_local_v_packed', params['local_0']['block']['self_attention'])
    self.assertNotIn('W_local_v_packed', params['fetch_1']['block']['self_attention'])
    jaxpr = str(jax.make_jaxpr(lambda hh, mm: module.apply(
        variables, (hh, mm), *args[1:]))(h, m))
    self.assertNotIn('cond[', jaxpr)
    self.assertIn('length=2', jaxpr)

  def test_training_signature_has_loss_but_no_health_metrics(self):
    for layout in ('Scan', 'NonScan'):
      with self.subTest(layout=layout):
        cfg = self.config('BamLlama2MediumV2C256LocalFetchC8LocalV' + layout)
        cfg.get_keys()['vocab_size'] = 128
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        args, _, shardings, model = train_compile.get_shaped_inputs(mesh, cfg)
        with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
          _, metrics = jax.eval_shape(
              lambda state, data, rng: train.train_step(model, cfg, shardings, state, data, rng),
              *args)
        self.assertIn('learning/loss', metrics['scalar'])
        self.assertFalse(any('norm' in name or 'bam/' in name for name in metrics['scalar']))
        self.assertEqual(cfg.steps, 13500)
        self.assertEqual(cfg.checkpoint_period, 200)

  def test_llf_scan_has_two_independent_local_layers(self):
    for suffix, block_size in (('C8LocalVLLFScan', 3), ('C8SharedReadLLFScan', 3),
                               ('C8SharedReadLLLFScan', 4),
                               ('C8SharedIndependentSharedLLLFScan', 4)):
      cfg = self.config('BamLlama2MediumV2C256LocalFetch' + suffix)
      cfg.get_keys()['num_decoder_layers'] = block_size * 2
      cfg.get_keys()['bam_layer_modes'] = ['local_qk+local_o'] * (block_size - 1) + ['local_qk+full']
      cfg.get_keys()['bam_layer_modes'] *= 2
      if isinstance(cfg.bam_local_o_v_mode, list):
        cfg.get_keys()['bam_local_o_v_mode'] = cfg.bam_local_o_v_mode[:block_size] * 2
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      module = nn.scan(
          BamLayerPair, variable_axes={'params': cfg.param_scan_axis},
          split_rngs={'params': True, 'dropout': False},
          in_axes=(nn.broadcast,) * 10 + (0,), length=2,
          metadata_params={nn.PARTITION_NAME: 'layers'})(
              cfg, mesh, 8, all_global_attention=True)
      h = jnp.ones((1, 8, 128), cfg.dtype)
      m = jnp.zeros((1, 8, 32, 32), cfg.dtype)
      args = ((h, m), jnp.ones((1, 8), jnp.int32), jnp.arange(8)[None],
              jnp.ones((1, 8), jnp.int32), None, True, 'train', None,
              None, None, None, jnp.arange(2))
      variables = module.init({'params': jax.random.key(8), 'aqt': jax.random.key(9)}, *args)
      (y, final_m), _ = module.apply(variables, *args)
      self.assertEqual(set(variables['params']),
                       {f'local_{i}' for i in range(block_size - 1)} | {f'fetch_{block_size - 1}'})
      if suffix == 'C8SharedIndependentSharedLLLFScan':
        for i in range(3):
          attention = variables['params'][f'local_{i}']['block']['self_attention']
          self.assertEqual('W_local_v_packed' in attention, i == 1)
          self.assertEqual('W_lv_gate' in attention, i != 1)
      self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
      self.assertEqual(final_m.shape, m.shape)
      jaxpr = str(jax.make_jaxpr(lambda hh, mm: module.apply(
          variables, (hh, mm), *args[1:]))(h, m))
      self.assertNotIn('cond[', jaxpr)

  def test_shared_output_gate_preserves_key_gate_scale(self):
    m = jax.random.normal(jax.random.key(10), (1, 3, 32, 8))
    row = jax.random.normal(jax.random.key(11), (1, 3, 2, 32))
    col = jax.random.normal(jax.random.key(12), (1, 3, 2, 8))
    logits = jax.random.normal(jax.random.key(13), (1, 3, 2, 2))
    def read(mode):
      r = _transform_bam_read_key(row, mode, 2., rms_epsilon=1e-4,
                                 gate_logits=logits[..., :1])
      c = _transform_bam_read_key(col, mode, 2., rms_epsilon=1e-4,
                                 gate_logits=logits[..., 1:])
      return jnp.pad(jnp.concatenate((jnp.einsum('btkv,btnv->btnk', m, c),
                                     jnp.einsum('btkv,btnk->btnv', m, r)), -1),
                     [(0, 0)] * 3 + [(0, 24)])
    cfg = SimpleNamespace(bam_k=32, bam_v=32, _abs_v_dim=8, _read_key_scale=2.,
                          _abs_k_dim=None, _abs_v_row_output='direct',
                          num_query_heads=2, head_dim=64)
    cfg._expand_full_read = lambda sides: BamAttention._expand_full_read.__wrapped__(cfg, sides)
    # Invoke the pure arithmetic with an attribute-only receiver.
    ungated = read('rms')
    sides = (ungated[..., :32], ungated[..., 32:40])
    got = BamAttention._gate_local_output.__wrapped__(cfg, sides, logits)
    np.testing.assert_allclose(got, read('rms_gate'), rtol=2e-5, atol=2e-5)
    def old_gate(col, row, gate_logits):
      expanded = cfg._expand_full_read((col, row))
      u, v, tail = jnp.split(expanded, [32, 40], axis=-1)
      gates = 2. * jax.nn.sigmoid(gate_logits)
      return jnp.concatenate((u * gates[..., 1:2], v * gates[..., :1], tail), -1)
    def new_gate(col, row, gate_logits):
      return BamAttention._gate_local_output.__wrapped__(cfg, (col, row), gate_logits)
    for dtype in (jnp.float32, jnp.bfloat16):
      args = tuple(x.astype(dtype) for x in (*sides, logits))
      np.testing.assert_array_equal(new_gate(*args), old_gate(*args))
      old_grad = jax.grad(lambda *a: old_gate(*a).astype(jnp.float32).sum(), (0, 1, 2))(*args)
      new_grad = jax.grad(lambda *a: new_gate(*a).astype(jnp.float32).sum(), (0, 1, 2))(*args)
      for old, new in zip(old_grad, new_grad):
        np.testing.assert_array_equal(old, new)

  def test_llf_native_diagonal_changes_only_fetch_override(self):
    import exp
    baseline = exp.BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan
    variant = exp.BamLlama2MediumV2C256LocalFetchC8SharedReadLLFNativeDiagonalScan
    differences = {name for name in dir(baseline) if not name.startswith('_')
                   and getattr(baseline, name) != getattr(variant, name)}
    self.assertEqual(differences, {'model_name', 'bam_fetch_diagonal_one'})
    self.assertFalse(variant.bam_fetch_diagonal_one)
    self.assertTrue(variant.scan_layers)
    self.assertEqual(variant.checkpoint_period, 200)


if __name__ == '__main__':
  absltest.main()

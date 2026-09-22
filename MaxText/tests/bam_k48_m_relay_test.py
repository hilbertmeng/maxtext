"""M anchor capture, zero identity and unchanged persistent-state write semantics."""
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
from layers.fusion import BamLayerPair, FusionDecoderLayer
from layers.attentions import BamAttention
from layers.models import Transformer
from layers.bam_m_relay_init import map_m_relay_params
from flax.linen import partitioning as nn_partitioning
from types import SimpleNamespace
import train_compile
import train


class MRelayTest(unittest.TestCase):
  def test_model_mapping_and_train_signature(self):
    for name in ('BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3LearnedScale', 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3VOOnly'):
      cfg = self.config(name)
      cfg.get_keys().update(vocab_size=128, dtype=jnp.float32,
          bam_local_o_v_mode=['rank2', 'rank2', 'none'] * 2)
      parent_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(), bam_m_relay_anchor=0)))
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model, parent = Transformer(cfg, mesh, quant=None), Transformer(parent_cfg, mesh, quant=None)
      key = jax.random.key(19)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        state = jax.jit(lambda: max_utils.init_initial_state(model, None, cfg, False, key))()
        old = jax.jit(lambda: max_utils.init_initial_state(parent, None, parent_cfg, False, key))()
        expected = map_m_relay_params(state.params['params'], old.params['params'], cfg.param_scan_axis)
        for actual, wanted in zip(jax.tree.leaves(state.params['params']), jax.tree.leaves(expected)):
          np.testing.assert_array_equal(actual, wanted)
        tokens = jnp.arange(4)[None]
        args = (tokens, tokens, jnp.ones_like(tokens), tokens)
        actual = jax.jit(lambda: model.apply(state.params, *args))()
        expected = jax.jit(lambda: parent.apply(old.params, *args))()
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
          np.testing.assert_allclose(actual_leaf, expected_leaf, atol=2e-5, rtol=2e-5)
          self.assertTrue(bool(jnp.all(jnp.isfinite(actual_leaf))))
      shaped, _, shardings, model = train_compile.get_shaped_inputs(mesh, cfg)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        _, metrics = jax.eval_shape(lambda st, data, rng: train.train_step(
            model, cfg, shardings, st, data, rng), *shaped)
        self.assertIn('learning/raw_grad_norm', metrics['scalar'])
        arms = {'all': ('all',), 'qk_vo': ('qk', 'vo'), 'vo_only': ('vo',)}[cfg.bam_m_relay_reads]
        for arm in arms:
          prefix = 'bam/m_relay' if arm == 'all' else f'bam/m_relay/{arm}'
          self.assertIn(f'{prefix}/layer_003/scale_mean', metrics['scalar'])
          self.assertIn(f'{prefix}/layer_005/scale_mean', metrics['scalar'])
        self.assertIn('bam/concat/local_q_gate/layer_000/mean', metrics['scalar'])
        self.assertIn('bam/concat/local_q_gate/layer_005/mean', metrics['scalar'])
        flat = flatten_dict(state.params['params'])
        gate_count = sum((v.value if hasattr(v, 'unbox') else v).size for path,v in flat.items() if 'm_relay_scale' in path)
        self.assertEqual(gate_count, 3*(128+1)*len(arms))
        self.assertFalse(any('first_block' in path and 'm_relay_scale' in path for path in flat))
        if getattr(cfg, 'bam_m_relay_learned_scale', False):
          amplitudes = [v.value if hasattr(v, 'unbox') else v for path,v in flat.items() if 'm_relay_amplitude_scale' in path]
          self.assertEqual(sum(v.size for v in amplitudes), 3)
          for v in amplitudes: np.testing.assert_array_equal(v, jnp.ones_like(v))
          self.assertIn('bam/m_relay/layer_003/amplitude_scale', metrics['scalar'])
          self.assertIn('bam/m_relay/layer_005/effective_scale_abs_mean', metrics['scalar'])

  def config(self, name):
    output = tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    (Path(output.name) / 'test').mkdir()
    cfg = pyconfig.initialize(
        [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
        exp_class=name, run_name='test', enable_checkpointing=False,
        base_output_directory=output.name + '/', jax_cache_dir='', log_config=False,
        dataset_type='synthetic', base_emb_dim=128, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=6, base_mlp_dim=256,
        head_dim=64, max_target_length=4, max_prefill_predict_length=4,
        query_chunk_size=2, per_device_batch_size=1.)
    cfg.get_keys()['bam_write_v_bottleneck_dim'] = 16
    cfg.get_keys()['mlp_dim_by_block'] = [128, 128, 128]
    cfg.get_keys()['bam_layer_modes'] = cfg.bam_layer_modes[:3] * 2
    return cfg

  def test_nonzero_routing_keeps_write_source_unmodified(self):
    class Probe(BamAttention):
      def _read_local(self, name, M, *args, **kwargs):
        self.sow('probe', name, M)
        return super()._read_local(name, M, *args, **kwargs)
      def _independent_local_vo(self, M, *args, **kwargs):
        self.sow('probe', 'vo', M)
        return super()._independent_local_vo(M, *args, **kwargs)
      def _compress_m(self, M):
        self.sow('probe', 'compression', M)
        return super()._compress_m(M)
      def _write(self, o, x, M):
        self.sow('probe', 'write_source', M)
        return super()._write(o, x, M)
    base = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayer'
    for suffix in ('MRelayM3', 'MRelayM3QKVO', 'MRelayM3LearnedScale', 'MRelayM3VOOnly'):
      cfg = self.config(base+suffix)
      cfg.get_keys().update(bam_m_read_norm='none', dtype=jnp.float32)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      for mode in ('local_qk+local_o', 'local_qk+full'):
        module = Probe(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
            bam_k=48, bam_v=32, max_target_length=4, max_prefill_predict_length=4,
            mesh=mesh, attention_kernel='dot_product_chunk', dtype=cfg.dtype,
            layer_mode=mode, local_v_mode='rank2' if 'local_o' in mode else 'none',
            read_side='col', attention_type=cfg.attention_type)
        x = jax.random.normal(jax.random.key(21), (1,4,128))
        m = jax.random.normal(jax.random.key(22), (1,4,48,32))
        a = jax.random.normal(jax.random.key(23), m.shape)
        args = (x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
        kw = dict(M_in=m, anchor_m=a, deterministic=True, layer_index=3)
        params = module.init({'params':jax.random.key(24)}, *args, **kw)['params']
        scales = jnp.array([.25, -.5] if suffix.endswith('QKVO') else [.25])
        bias = params['m_relay_scale']['bias']
        vals = jnp.arctanh(scales)
        params['m_relay_scale']['bias'] = bias.replace(value=vals) if hasattr(bias,'unbox') else vals
        amplitude = 1.
        if suffix.endswith('LearnedScale'):
          scalar = params['m_relay_amplitude_scale']
          np.testing.assert_array_equal(scalar.value if hasattr(scalar, 'unbox') else scalar, jnp.ones((1,)))
          amplitude = 2.
          val = jnp.array([amplitude], jnp.float32)
          params['m_relay_amplitude_scale'] = scalar.replace(value=val) if hasattr(scalar,'unbox') else val
        (_,m_out), out = module.apply({'params':params}, *args, **kw, mutable=['probe'])
        probe = out['probe']
        qk_scale = 0. if suffix.endswith('VOOnly') else amplitude*.25
        np.testing.assert_allclose(probe['q'][0], m+qk_scale*a, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(probe['k'][0], m+qk_scale*a, rtol=1e-5, atol=1e-5)
        vo_scale = -.5 if suffix.endswith('QKVO') else amplitude*.25
        for v in probe['compression']:
          np.testing.assert_allclose(v, m+vo_scale*a, rtol=1e-5, atol=1e-5)
        if 'local_o' in mode:
          np.testing.assert_allclose(probe['vo'][0], m+vo_scale*a, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(probe['write_source'][0], m)
        self.assertTrue(bool(jnp.all(jnp.isfinite(m_out))))

  def test_capture_and_zero_relay(self):
    for width, anchor in ((48, 3),):
      with self.subTest(width=width, anchor=anchor):
        name = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerMRelayM3QKVO'
        cfg = self.config(name)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        block = BamLayerPair(cfg, mesh, 4, all_global_attention=True, capture_anchor=True)
        h = jax.random.normal(jax.random.key(1), (1, 4, 128), dtype=cfg.dtype)
        m = jnp.zeros((1, 4, width, 32), cfg.dtype)
        tail = (jnp.ones((1, 4), jnp.int32), jnp.arange(4)[None],
                jnp.ones((1, 4), jnp.int32), None, True, 'train', None, None, None, None, 0)
        var = block.init({'params': jax.random.key(2)}, (h, m), *tail)
        carry, _ = block.apply(var, (h, m), *tail)
        if anchor == 3:
          np.testing.assert_array_equal(carry[2], carry[1])
        else:
          first = FusionDecoderLayer(cfg, mesh, 4, all_global_attention=True, static_layer_index=0)
          first_carry, _ = first.apply({'params': var['params']['local_0']}, (h, m), *tail)
          np.testing.assert_array_equal(carry[2], first_carry[1])
        regular = BamLayerPair(cfg, mesh, 4, all_global_attention=True)
        params = regular.init({'params': jax.random.key(3)}, carry, *tail)['params']
        y_relay, _ = regular.apply({'params': params}, carry, *tail)
        y_plain, _ = regular.apply({'params': params}, carry[:2], *tail)
        for x, y in zip(y_relay[:2], y_plain):
          np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(y_relay[2], carry[2])
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in y_relay))


if __name__ == '__main__':
  unittest.main()

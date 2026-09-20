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
    for name in ('BamMediumColOnlyK32MRelayM1', 'BamMediumColOnlyK64TruncateMRelayM3',
                 'BamMediumColOnlyK32MRelayM3Linear', 'BamMediumColOnlyK32MRelayM3Interpolate',
                 'BamMediumColOnlyK32PartialMRelayM3', 'BamMediumColOnlyK64MRelayM3QKOnly',
                 'BamMediumColOnlyK64MRelayM3VOnly', 'BamMediumColOnlyK64MRelayM3OOnly'):
      cfg = self.config(name)
      cfg.get_keys().update(vocab_size=128, dtype=jnp.float32,
          bam_local_o_v_mode=['rank2', 'rank2', 'none'] * 2)
      parent_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(), bam_m_relay_anchor=0)))
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model, parent = Transformer(cfg, mesh, quant=None), Transformer(parent_cfg, mesh, quant=None)
      key = jax.random.key(19)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        state = jax.jit(lambda: max_utils.init_initial_state(model, None, cfg, False, key))()
        if cfg.bam_m_relay_mixing == 'sigmoid_interpolate':
          gates = [v for p, v in flatten_dict(state.params['params']).items() if p[-1] == 'm_relay_gate_b0']
          self.assertTrue(gates)
          for gate in gates:
            gate = gate.value if hasattr(gate, 'unbox') else gate
            np.testing.assert_allclose(jax.nn.sigmoid(gate), .01, rtol=1e-5)
        old = jax.jit(lambda: max_utils.init_initial_state(parent, None, parent_cfg, False, key))()
        expected = map_m_relay_params(state.params['params'], old.params['params'], cfg.param_scan_axis)
        for actual, wanted in zip(jax.tree.leaves(state.params['params']), jax.tree.leaves(expected)):
          np.testing.assert_array_equal(actual, wanted)
        tokens = jnp.arange(4)[None]
        args = (tokens, tokens, jnp.ones_like(tokens), tokens)
        actual = jax.jit(lambda: model.apply(state.params, *args))()
        expected = jax.jit(lambda: parent.apply(old.params, *args))()
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
          if cfg.bam_m_relay_mixing != 'sigmoid_interpolate':
            np.testing.assert_allclose(actual_leaf, expected_leaf, atol=2e-5, rtol=2e-5)
          self.assertTrue(bool(jnp.all(jnp.isfinite(actual_leaf))))
      shaped, _, shardings, model = train_compile.get_shaped_inputs(mesh, cfg)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        _, metrics = jax.eval_shape(lambda st, data, rng: train.train_step(
            model, cfg, shardings, st, data, rng), *shaped)
        self.assertIn('learning/raw_grad_norm', metrics['scalar'])

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

  def test_capture_and_zero_relay(self):
    for width, anchor in ((32, 1), (32, 3), (64, 1), (64, 3)):
      with self.subTest(width=width, anchor=anchor):
        name = f'BamMediumColOnlyK{width}' + ('Truncate' if width == 64 else '') + f'MRelayM{anchor}'
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

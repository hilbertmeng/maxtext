"""Official Pythia-2.8B dimensions, BAM budget, initialization and scan tails."""
import contextlib
import copy
import io
import math
from types import SimpleNamespace
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from pathlib import Path
import pyconfig
import pythia_test as reference
from layers import initializers, models


class Pythia2p8BTest(unittest.TestCase):
  setUp = reference.PythiaTest.setUp
  tearDown = reference.PythiaTest.tearDown
  model = reference.PythiaTest.model
  inputs = reference.PythiaTest.inputs
  init = reference.PythiaTest.init

  def config(self, **kwargs):
    args = dict(exp_class='BamPythia2p8BSota', run_name='test',
                base_output_directory=self.tmp.name+'/', enable_checkpointing=False,
                jax_cache_dir='', log_config=False, dataset_type='synthetic')
    args.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
      return pyconfig.initialize([None, 'MaxText/configs/base.yml'], **args)

  def small_config(self, **kwargs):
    args = dict(base_emb_dim=320, base_num_decoder_layers=5,
                base_num_query_heads=4, base_num_kv_heads=4, vocab_size=32,
                max_target_length=8, max_prefill_predict_length=4,
                target_train_batch_size=None, per_device_batch_size=1.,
                attention='dot_product', dtype='float32', weight_dtype='float32',
                loss_chunk_size=4, bam_layer_modes=['local_qk+local_v+local_o']*5,
                emb_bam_num_head=4, mlp_dim_by_block=[256, 224, 256], bam_final_local_mlp_dim=256,
                bam_local_full_m_read_bottleneck_dim=64, bam_write_v_bottleneck_dim=64,
                emb_bam_v_bottleneck_dim=64, bam_mlp_write_address_rank=64,
                bam_splash_attention=False, remat_policy='full')
    args.update(kwargs)
    declared = yaml.safe_load(Path('MaxText/configs/base.yml').read_text())
    patch = {key: args.pop(key) for key in list(args) if key not in declared}
    c = self.config(**args)
    return pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), **patch)))

  def test_official_recipe_and_device_independent_batch(self):
    for arm in ('Pythia2p8B', 'BamPythia2p8BSota'):
      c = self.config(exp_class=arm)
      self.assertEqual((c.emb_dim, c.num_decoder_layers, c.num_query_heads, c.head_dim), (2560, 32, 32, 80))
      self.assertEqual((c.mlp_dim, c.vocab_size, c.max_target_length), (10240, 50304, 2048))
      self.assertEqual(c.mlp_activations, ['gelu_exact'])
      self.assertEqual(c.pythia_rotary_dim, 20)
      self.assertEqual(c.normalization_layer_epsilon, 1e-5)
      self.assertEqual(c.dtype, jnp.bfloat16)
      self.assertEqual(jnp.dtype(c.weight_dtype), jnp.dtype(jnp.float32))
      self.assertEqual(c.global_batch_size_to_train_on, 1024)
      self.assertEqual((c.steps, c.learning_rate_schedule_steps), (143000, 143000))
      self.assertEqual(c.learning_rate, 1.6e-4)
      self.assertEqual(c.pad_id, -1)
      self.assertFalse(c.qk_norm)
      self.assertFalse(c.record_training_health_metrics)
      if arm.startswith('Bam'):
        self.assertFalse(c.pythia_parallel_residual)
        self.assertEqual((c.bam_k, c.bam_v), (80, 40))
        self.assertEqual((c.bam_partial_rope_nope_dim, c.bam_standard_qk_dim), (60, 20))
        self.assertTrue(c.bam_local_full_m_read_share_down)
        self.assertFalse(c.bam_share_read_attention_write_down)
        self.assertFalse(c.bam_share_mlp_write_down)
        self.assertEqual(c.bam_m_shift_init, .5)
        self.assertEqual(c.bam_m_shift_mode, 'read')
        self.assertEqual(initializers.get_bam_write_epsilon(c), 1e-6)
      else:
        self.assertTrue(c.pythia_parallel_residual)
    # The actual Pile converter must supervise EOS0 while masking only padding.
    import tensorflow as tf
    from input_pipeline._pile_data_processing import PileDatasets
    reader = object.__new__(PileDatasets)
    reader.seq_len, reader.batch_size, reader.mix_attn, reader.pad_id = 3, 1, False, c.pad_id
    batch = reader.convert({'input_ids': tf.constant([[7, 0, 8, -1]], tf.int32)})
    np.testing.assert_array_equal(batch['targets_segmentation'], [[1, 1, 0]])
    from pythia_config import validate_pythia_config
    for flag in ('bam_share_read_attention_write_down', 'bam_share_mlp_write_down'):
      with self.assertRaisesRegex(ValueError, 'recorded runtime'):
        validate_pythia_config(dict(c.get_keys(), **{flag: True}))
    for devices in (1, 8, 32, 64, 128):
      for accumulation in (1, 4, 16):
        per_device = pyconfig.resolve_per_device_batch_size(3., devices, accumulation, 1024)
        _, total, _ = pyconfig.calculate_global_batch_sizes(per_device, -1, devices, accumulation)
        self.assertEqual(total, 1024)
    self.assertEqual(pyconfig.resolve_per_device_batch_size(3., 8, 1), 3.)
    for target in (True, 0, -1, 1024.5):
      with self.assertRaises(ValueError):
        pyconfig.resolve_per_device_batch_size(1., 8, 1, target)
    with self.assertRaises(ValueError):
      pyconfig.resolve_per_device_batch_size(1., 8, 3, 1024)

  def test_full_parameter_budget(self):
    counts = {}
    for arm in ('Pythia2p8B', 'BamPythia2p8BSota'):
      c = self.config(exp_class=arm, max_target_length=8, max_prefill_predict_length=4,
                      attention='dot_product', loss_chunk_size=4, bam_splash_attention=False)
      m, mesh = self.model(c)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), contextlib.redirect_stdout(io.StringIO()):
        p = jax.eval_shape(lambda key: m.init(key, *self.inputs(), enable_dropout=False)['params'], jax.random.key(19))
      p = nn.unbox(p)
      flat = flatten_dict(p)
      counts[arm] = sum(math.prod(a.shape) for a in flat.values())
      if c.bam_enabled:
        self.assertEqual(p['decoder']['layers']['local_0']['block']['mlp']['wi']['kernel'].shape, (2560, 10, 12427))
        self.assertEqual(p['decoder']['final_local_layer_0']['block']['mlp']['wi']['kernel'].shape, (2560, 12427))
        self.assertEqual(p['decoder']['final_local_layer_1']['block']['mlp']['wi']['kernel'].shape, (2560, 12427))
        self.assertNotIn('mlp_address_down', p['decoder']['final_local_layer_1']['block'])
        attn = p['decoder']['layers']['local_0']['block']['self_attention']
        self.assertIn('full_m_read_down', attn)
        self.assertIn('P_loc_down', attn)  # Read and write downs are independent.
    self.assertEqual(counts, {'Pythia2p8B': 2775208960, 'BamPythia2p8BSota': 2775208602})

  def test_initialization_roles_and_seed_epsilon(self):
    c = self.small_config(scan_layers=False)
    _, _, p = self.init(c)
    flat = flatten_dict(nn.unbox(p))
    small = math.sqrt(2/(5*c.emb_dim))
    wang = 2/c.num_decoder_layers/math.sqrt(c.emb_dim)
    checked = set()
    for path, value in flat.items():
      core = path[-1] == 'logits_dense' or (path[-1] == 'kernel' and path[-2] in ('query', 'key', 'value', 'out', 'wi', 'wo'))
      embedding = path[-1] == 'embedding'
      bam_kernel = path[-1] == 'kernel' and path[-2] in ('full_m_read_down', 'P_loc_down', 'P_loc_up', 'mlp_address_down', 'mlp_address_up', 'W_emb_u', 'W_emb_v_down', 'W_emb_v_up', 'W_emb_g')
      if core or embedding or bam_kernel:
        target = (wang if path[-2] in ('out', 'wo') else small) if core else .006
        self.assertAlmostEqual(float(value.std())/target, 1., delta=.12, msg=str(path))
        checked.add('core' if core else 'embedding' if embedding else 'bam')
      if path[-2:] == ('M_shift_gate', 'kernel'):
        np.testing.assert_array_equal(value, 0.)
      if path[-1] == 'M_shift_gate_b0':
        np.testing.assert_array_equal(value, 0.)  # sigmoid(0) = .5
    self.assertEqual(checked, {'core', 'embedding', 'bam'})
    x = jax.random.normal(jax.random.key(7), (1, 8, 320))*.006
    legacy = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), normalization_layer_epsilon=1e-6, bam_write_rms_epsilon=None)))
    seed = lambda cfg: models.EmbeddingBamWrite(cfg, 4, cfg.dtype, cfg.weight_dtype, None, initializers.get_bam_init(cfg))
    m, old = seed(c), seed(legacy)
    with nn.partitioning.axis_rules(c.logical_axis_rules):
      sp = m.init(jax.random.key(8), x)
      np.testing.assert_array_equal(m.apply(sp, x), old.apply(sp, x))
    c = self.config(exp_class='Pythia2p8B')
    for fn in (initializers.get_model_init, initializers.get_token_embedding_init):
      a = fn(c)(jax.random.key(3), (1024, 1024), jnp.float32)
      self.assertAlmostEqual(float(a.std())/.0125, 1., delta=.005)

  def test_two_tail_scan_parity_and_bfloat16_gradients(self):
    c = self.small_config()
    m, mesh, p = self.init(c)
    p = nn.unbox(p)
    up = copy.deepcopy(p)
    layers = up['decoder'].pop('layers')
    for index, name in enumerate(('local_0', 'local_1', 'fetch_2')):
      up['decoder'][f'layers_{index}'] = jax.tree.map(lambda a: jnp.take(a, 0, axis=c.param_scan_axis), layers[name])
    for offset in range(2):
      up['decoder'][f'layers_{3+offset}'] = up['decoder'].pop(f'final_local_layer_{offset}')
    uc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), scan_layers=False)))
    um, _ = self.model(uc)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      expected = m.apply({'params': p}, *self.inputs(), enable_dropout=False)[0]
      actual = um.apply({'params': up}, *self.inputs(), enable_dropout=False)[0]
      np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    # Exercise the requested precision, rematerialization, both tails, and an update.
    bc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), dtype=jnp.bfloat16)))
    bm, _ = self.model(bc)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      loss, g = jax.jit(jax.value_and_grad(lambda p: jnp.mean(bm.apply({'params': p}, *self.inputs(), enable_dropout=False)[0])))(p)
      self.assertTrue(np.isfinite(loss))
      grads = flatten_dict(g)
      for value in grads.values():
        self.assertTrue(np.isfinite(value).all())
      for name in ('final_local_layer_0', 'final_local_layer_1'):
        self.assertGreater(float(jnp.linalg.norm(grads[('decoder', name, 'block', 'mlp', 'wi', 'kernel')])), 0.)
      self.assertGreater(float(jnp.linalg.norm(grads[('decoder', 'layers', 'local_1', 'block', 'mlp_address_up', 'kernel')])), 0.)
      optimizer = optax.adamw(1e-4)
      updates, _ = optimizer.update(g, optimizer.init(p), p)
      updated = optax.apply_updates(p, updates)
      self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(updated))))


if __name__ == '__main__':
  unittest.main()

"""Raw-content frequency configs: full budgets/health and scanned gradients."""
import functools
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import pyconfig
import train
import train_compile
from layers import quantizations
from layers.models import Transformer

PREFIX = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependent'
ARMS = [
    ('EverySecondBlockFirst', 2, 432098528, 9),
    ('EveryLayer', 1, 432105728, 18),
]


class RawFrequencyTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    Path(self.tmp.name, 'audit').mkdir()

  def tearDown(self):
    self.tmp.cleanup()

  def config(self, name, **kwargs):
    return pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=name, run_name='audit',
        enable_checkpointing=False, base_output_directory=self.tmp.name + '/',
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
        per_device_batch_size=1., **kwargs)

  def test_full_budget_tree_and_writer_health(self):
    for suffix, period, expected, writers in ARMS:
      signatures = []
      for raw in (False, True):
        name = PREFIX + suffix + ('RawContent' if raw else '') + 'TruePile'
        c = self.config(name)
        self.assertEqual(c.bam_mlp_write_content_rms, not raw)
        self.assertTrue(c.bam_write_data_rms)
        self.assertTrue(c.bam_mlp_write_dynamic_address)
        self.assertEqual(c.bam_mlp_write_address_rank, 256)
        self.assertEqual(c.DATASET_VARIANT, 'truepile4096')
        self.assertEqual(c.bam_pair_scan, period == 2)
        self.assertEqual(c.mlp_dim_by_block, [3774, 3901] if period == 2 else None)
        if period == 1:
          self.assertEqual(c.mlp_dim, 3774)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        args, kwargs, shardings, model = train_compile.get_shaped_inputs(mesh, c)
        flat = flatten_dict(args[0].params)
        self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()), expected)
        address_count = sum(int(np.prod(v.shape)) for p, v in flat.items()
                            if any(k in p for k in ('mlp_address_down', 'mlp_address_up')))
        self.assertEqual(address_count, 438784 * writers)
        signatures.append({p: v.shape for p, v in flat.items()})
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
          metrics = jax.eval_shape(
              functools.partial(train.train_step, model, c, shardings), *args, **kwargs)[1]
        for layer in range(18):
          for group, stat in [('mlp_write_gate', 'mean'),
                              ('mlp_write_raw_output_amplitude', 'bam_rms'),
                              ('mlp_write_content_amplitude', 'bam_rms'),
                              ('mlp_address_overlap', 'rho_cross')]:
            self.assertEqual(f'bam/concat/{group}/layer_{layer:03d}/{stat}' in metrics['scalar'],
                             layer % period == 0)
      self.assertEqual(*signatures)
      print('RAW_FREQUENCY_BUDGET_HEALTH_OK', suffix, expected, writers, flush=True)

  def test_identical_initial_parameters_and_finite_scanned_gradients(self):
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    mask = jnp.ones_like(tokens)
    call = (tokens, jnp.arange(4)[None], tokens, mask, mask)
    rngs = {'params': jax.random.key(3), 'dropout': jax.random.key(4), 'aqt': jax.random.key(5)}
    for suffix, period, _, _ in ARMS:
      params = []
      for raw in (False, True):
        c = self.config(PREFIX + suffix + ('RawContent' if raw else '') + 'TruePile',
                        dtype='float32', weight_dtype='float32')
        c.get_keys().update(
            emb_dim=150, num_query_heads=2, num_kv_heads=2,
            base_num_decoder_layers=4, num_decoder_layers=4, mlp_dim=64,
            mlp_dim_by_block=[64] * period if period == 2 else None,
            vocab_size=32, bam_layer_modes=['local_qk+local_v+local_o'] * 4,
            bam_write_v_bottleneck_dim=16, emb_bam_num_head=2,
            emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        model = Transformer(c, mesh, quantizations.configure_quantization(c))
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
          p = nn.unbox(model.init(rngs, *call, enable_dropout=False)['params'])
          params.append(flatten_dict(p))
          if raw:
            def loss(par):
              logits = model.apply({'params': par}, *call, enable_dropout=False,
                                   rngs={'aqt': rngs['aqt']})[0]
              return jnp.mean(logits ** 2)
            val, grads = jax.jit(jax.value_and_grad(loss))(p)
            self.assertTrue(np.isfinite(float(val)))
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads)))
            flat = flatten_dict(grads)
            for key in ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate'):
              self.assertGreater(sum(float(jnp.sum(v * v)) for path, v in flat.items() if key in path), 0)
        self.assertEqual(params[0].keys(), params[-1].keys())
      for path in params[0]:
        np.testing.assert_array_equal(params[0][path], params[1][path], err_msg='/'.join(path))
      print('RAW_FREQUENCY_INITIAL_PARITY_GRAD_OK', suffix, flush=True)


if __name__ == '__main__':
  unittest.main()

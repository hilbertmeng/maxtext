"""record_internal_nn_metrics on scanned stacks without the fusion 'sub_0' wrapper (Pythia layers, BAM pair scan)."""
import os
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp

sys.path.insert(0, 'MaxText')
import max_utils  # pylint: disable=wrong-import-position
import pyconfig  # pylint: disable=wrong-import-position
import train  # pylint: disable=wrong-import-position
from layers.models import Transformer  # pylint: disable=wrong-import-position

TINY_MHA = ['base_emb_dim=64', 'base_num_query_heads=2', 'base_num_kv_heads=2', 'head_dim=32',
            'base_num_decoder_layers=4', 'base_mlp_dim=128', 'vocab_size=1024', 'max_target_length=256',
            'max_prefill_predict_length=256', 'pythia_rotary_dim=8', 'attention=dot_product', 'dtype=float32',
            'per_device_batch_size=1']


def collect(exp, extra):
  out = tempfile.mkdtemp()
  os.makedirs(os.path.join(out, exp), exist_ok=True)
  cfg = pyconfig.initialize(['x', 'MaxText/configs/base.yml', f'exp_class={exp}', f'run_name={exp}',
                             f'base_output_directory={out}', 'dataset_type=synthetic', 'enable_checkpointing=False',
                             'jax_cache_dir=', 'record_internal_nn_metrics=True'] + extra)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  b, t = int(cfg.global_batch_size_to_train_on), cfg.max_target_length
  tokens = jax.random.randint(jax.random.PRNGKey(0), (b, t), 0, 1000)
  positions = jnp.broadcast_to(jnp.arange(t), (b, t))
  segments = jnp.ones((b, t), jnp.int32)
  rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2), 'aqt': jax.random.PRNGKey(3)}
  params = model.init(rngs, tokens, positions, segments, tokens)
  _, state = model.apply(params, tokens, positions, segments, tokens, enable_dropout=False, rngs=rngs,
                         mutable=['intermediates'])
  metrics = {'scalar': {}}
  train.record_activation_metrics(metrics, state, cfg)
  return cfg, metrics['scalar']


class ScannedActivationMetricsTest(unittest.TestCase):

  def test_bam_pair_scan(self):
    cfg, scalars = collect('BamPallasPythiaSotaTinyXlaTest', [])
    want = {f'block/{k}/layer_{i:03d}' for i in range(cfg.num_decoder_layers)
            for k in ('attn_lnx/l2norm', 'mlp_lnx/l2norm')}
    self.assertEqual(want, {k for k in scalars if k.startswith('block/')})

  def test_pythia_scanned_layers(self):
    cfg, scalars = collect('Pythia2p8BV64Profile', TINY_MHA)
    want = {f'block/{k}/layer_{i:03d}' for i in range(cfg.num_decoder_layers)
            for k in ('activation_mean', 'activation_stdev')}
    self.assertEqual(want, {k for k in scalars if k.startswith('block/')})


if __name__ == '__main__':
  unittest.main()

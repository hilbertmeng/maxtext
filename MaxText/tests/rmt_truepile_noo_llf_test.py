"""Focused CPU check for the TruePile NoO -> LLF fetched-O change."""

import contextlib
import io
import math
from pathlib import Path
import tempfile

from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from layers import models, rmt


class NoOLLFTest(absltest.TestCase):

  def _config(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    Path(directory.name, 'test').mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
      cfg = pyconfig.initialize(
          [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
          exp_class='RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF',
          run_name='test', enable_checkpointing=False,
          base_output_directory=directory.name + '/', jax_cache_dir='',
          log_config=False, dataset_type='synthetic', base_emb_dim=16 * 20,
          base_num_query_heads=16, base_num_kv_heads=16,
          base_num_decoder_layers=3, base_mlp_dim=128, head_dim=20,
          max_target_length=4, max_prefill_predict_length=4,
          query_chunk_size=2, per_device_batch_size=1.)
    return cfg

  def test_fetch_only_in_f_layer_and_finite_gradients(self):
    cfg = self._config()
    cfg.get_keys().update(dtype=jnp.float32, rmt_mlp_dim_by_block=[128, 128, 127])
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None)
    args = dict(decoder_input_tokens=jnp.array([[1, 2, 3, 4]], jnp.int32),
                decoder_positions=jnp.arange(4)[None],
                decoder_target_tokens=jnp.array([[2, 3, 4, 5]], jnp.int32),
                decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                decoder_segment_ids=jnp.ones((1, 4), jnp.int32),
                enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(701), **args)['params'])
    layers = params['decoder']['layers']
    for offset in (0, 1):
      layer = layers[f'layer_{offset}']
      self.assertNotIn('fetch_head_mix_kernel', layer)
      self.assertEqual(layer['dynamic_vo']['gate_kernel'].shape[-1], cfg.num_query_heads)
    fetched = layers['layer_2']
    self.assertEqual(fetched['dynamic_vo']['gate_kernel'].shape[-1], 2 * cfg.num_query_heads)
    self.assertEqual(fetched['fetch_head_mix_kernel'].shape[-1], cfg.num_query_heads)
    self.assertNotIn('o_key_kernel', fetched['dynamic_vo'])

    # Wake the zero-initialized dynamic key to test the actual temporal route.
    key = fetched['dynamic_vo']['key_kernel']
    fetched['dynamic_vo']['key_kernel'] = key + .01 * jax.random.normal(
        jax.random.key(702), key.shape)
    def forward(p):
      output, aux = model.apply({'params': p}, **args, mutable=['intermediates'])
      return jnp.mean(output[0]), aux
    with contextlib.redirect_stdout(io.StringIO()):
      (loss, aux), grads = jax.value_and_grad(forward, has_aux=True)(params)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads)))
    fetch_grad = grads['decoder']['layers']['layer_2']['fetch_head_mix_kernel']
    self.assertGreater(float(jnp.linalg.norm(fetch_grad)), 0.)
    self.assertIn('rmt_fetch_route_sums', aux['intermediates']['decoder']['layers']['layer_2'])
    health = aux['intermediates']['decoder']['layers']['rmt_dynamic_health'][0]
    self.assertTrue(bool(jnp.all(jnp.isfinite(health))))

  def test_local_no_o_stays_identical(self):
    cfg = self._config()
    cfg.get_keys().update(dtype=jnp.float32, rmt_block_scan=False)
    matrix = jax.random.normal(jax.random.key(703), (1, 4, 48, cfg.head_dim))
    args = (jnp.ones((1, 4), jnp.int32), jnp.arange(4)[None], True, 0)
    layer = rmt.RMTLayer(cfg, mlp_dim=128, is_fetch=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(layer.init(jax.random.key(704), matrix, *args)['params'])
      actual = layer.apply({'params': params}, matrix, *args)[0]
      cfg.get_keys()['rmt_llf_enabled'] = False
      baseline = layer.apply({'params': params}, matrix, *args)[0]
    np.testing.assert_array_equal(actual, baseline)

  def test_full_parameter_budget(self):
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    Path(directory.name, 'audit').mkdir()
    counts = []
    for name in ('RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO',
                 'RMTMediumPropT4096TruePileK48EmbedUnembedDirect32NoOLLF'):
      with contextlib.redirect_stdout(io.StringIO()):
        cfg = pyconfig.initialize(
            [None, str(Path(__file__).parents[1] / 'configs/base.yml')],
            exp_class=name, run_name='audit', enable_checkpointing=False,
            base_output_directory=directory.name + '/', jax_cache_dir='',
            log_config=False, dataset_type='synthetic', max_target_length=4,
            max_prefill_predict_length=4, query_chunk_size=2,
            per_device_batch_size=1.)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = models.Transformer(config=cfg, mesh=mesh, quant=None)
      args = dict(decoder_input_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_positions=jnp.arange(4)[None],
                  decoder_target_tokens=jnp.ones((1, 4), jnp.int32),
                  decoder_target_mask=jnp.ones((1, 4), jnp.float32),
                  decoder_segment_ids=jnp.ones((1, 4), jnp.int32),
                  enable_dropout=False)
      shapes = jax.eval_shape(lambda key: model.init(key, **args)['params'],
                              jax.random.key(1))
      counts.append(sum(math.prod(x.shape) for x in jax.tree.leaves(shapes)))
    self.assertEqual(counts, [431773472, 431766464])


if __name__ == '__main__':
  absltest.main()

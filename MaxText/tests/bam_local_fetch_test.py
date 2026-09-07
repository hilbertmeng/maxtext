"""LocalO/fetch experiments: module gradients and static two-layer scan semantics."""
from pathlib import Path
import tempfile
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
from layers.attentions import BamAttention
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

  def test_five_local_modules_forward_and_gradients(self):
    for suffix in ('C8', 'C8LocalV', 'Full', 'FullLocalV', 'C8SharedRead'):
      with self.subTest(suffix=suffix):
        cfg = self.config('BamLlama2MediumV2C256LocalFetch' + suffix + 'NonScan')
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


if __name__ == '__main__':
  absltest.main()

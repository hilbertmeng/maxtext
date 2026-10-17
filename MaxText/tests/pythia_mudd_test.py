"""Serial Pythia/Mudd integration, full-history routing and parameter accounting."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import pyconfig
import pythia_test as reference


class PythiaSerialTest(reference.PythiaTest):
  """Run the NeoX oracle, cache, scan, remat and optimizer tests in serial mode."""

  def config(self, **kwargs):
    kwargs.setdefault('pythia_parallel_residual', False)
    return super().config(**kwargs)


class PythiaMuddTest(unittest.TestCase):
  setUp = reference.PythiaTest.setUp
  tearDown = reference.PythiaTest.tearDown
  config = reference.PythiaTest.config
  model = reference.PythiaTest.model
  inputs = reference.PythiaTest.inputs
  init = reference.PythiaTest.init

  def test_paxml_3b_config_and_router_shapes(self):
    # Trace the real 3B parameter tree without allocating its multi-GB weights.
    c = self.config(exp_class='MuddPythia3BSerialMLP', base_emb_dim=2560,
                    base_num_decoder_layers=32, base_num_query_heads=32,
                    base_num_kv_heads=32, head_dim=80, vocab_size=50432)
    self.assertEqual(c.model_name, 'MuddPythia3BSerialMLP')
    self.assertEqual(c.mlp_dim, 10240)
    self.assertEqual(c.global_batch_size_to_train_on, 1024)
    self.assertEqual(c.pythia_rotary_dim, 20)
    self.assertEqual(c.mlp_activations, ['gelu'])
    self.assertTrue(c.qk_norm)
    self.assertFalse(c.pythia_parallel_residual)
    self.assertEqual(c.steps, 143000)
    self.assertEqual(c.learning_rate_schedule_steps, 143000)
    self.assertEqual(int(c.steps * c.warmup_steps_fraction), 1430)
    self.assertEqual(c.learning_rate, 1.6e-4)
    self.assertEqual(c.adam_weight_decay, .1)
    self.assertEqual(c.init_weights_seed, 1234)
    self.assertEqual(c.pile_loss_mask, 'nonnegative')
    self.assertEqual(c.rms_norm_epsilon, 1e-6)
    self.assertTrue(c.record_training_health_metrics)
    m, mesh = self.model(c)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      p = nn.unbox(jax.eval_shape(lambda key: m.init(key, *self.inputs(), enable_dropout=False)['params'], jax.random.key(0)))
    for i in range(32):
      width = round(10240 * (i / 31 + .5) / 128) * 128
      self.assertEqual(p['decoder'][f'layers_{i}']['block']['mlp']['wi']['kernel'].shape, (2560, width))
      if i:
        router = p['decoder'][f'layers_{i}']['compose']['mlp']
        hidden = (4 * (i + 1) // 64 + 1) * 64
        self.assertEqual(router['dynamic_dense_conn1']['kernel'].shape, (2560, hidden))
        self.assertEqual(router['dynamic_dense_conn2']['kernel'].shape, (hidden, 4, i + 1))
    final = p['decoder']['compose_final']['mlp']
    self.assertEqual(final['dynamic_dense_conn1']['kernel'].shape, (2560, 192))
    self.assertEqual(final['dynamic_dense_conn2']['kernel'].shape, (192, 1, 33))

  def test_3b_batch_follows_topology(self):
    for devices, accumulation, per_device in [(128, 1, 8.), (256, 1, 4.), (128, 2, 4.)]:
      with self.subTest(devices=devices, accumulation=accumulation):
        with mock.patch.object(pyconfig, 'get_num_target_devices', return_value=devices):
          c = self.config(exp_class='MuddPythia3BSerialMLP',
                          mudd_router_hidden_expand=[1, 4], pythia_rotary_dim=4,
                          gradient_accumulation_steps=accumulation)
        self.assertEqual(c.global_batch_size_to_train_on, 1024)
        self.assertEqual(c.per_device_batch_size, per_device)

  def test_3b_remat_profiles_preserve_logits_and_gradients(self):
    c = self.config(exp_class='MuddPythia3BSerialMLP', target_global_batch_size=0,
                    base_mlp_dim=256, mudd_router_hidden_expand=[1, 4],
                    pythia_rotary_dim=4, remat_policy='full')
    model, mesh, params = self.init(c)
    args = self.inputs()
    results = []
    for policy in ('full', 'save_all', 'minimal'):
      config = pyconfig.HyperParameters(SimpleNamespace(
          keys=dict(c.get_keys(), remat_policy=policy)))
      model, _ = self.model(config)
      def loss(p):
        logits = model.apply({'params': p}, *args, enable_dropout=False)[0]
        return jnp.mean(jnp.square(logits)), logits
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        results.append(jax.jit(jax.value_and_grad(loss, has_aux=True))(params))
    for value in results[1:]:
      for control, candidate in zip(jax.tree.leaves(results[0]), jax.tree.leaves(value)):
        np.testing.assert_allclose(candidate, control, rtol=5e-5, atol=1e-6)

  def test_3b_health_metrics_report_unclipped_norm_without_changing_update(self):
    from flax.training.train_state import TrainState
    import optax
    # External cloud adapters are unrelated to the train-step numerical contract.
    with mock.patch.dict('sys.modules', {
        'vertex_tensorboard': SimpleNamespace(VertexTensorboardManager=object),
        'gcp_workload_monitor': SimpleNamespace(GCPWorkloadMonitor=object),
        'ml_goodput_measurement': SimpleNamespace(monitoring=object),
    }):
      import train
    c = self.config(exp_class='MuddPythia3BSerialMLP', target_global_batch_size=0,
                    base_num_decoder_layers=2, base_mlp_dim=256,
                    mudd_router_hidden_expand=[1, 4], pythia_rotary_dim=4,
                    gradient_clipping_threshold=.01)
    m, mesh, p = self.init(c)
    state = TrainState.create(apply_fn=m.apply, params={'params': nn.unbox(p)},
                              tx=optax.sgd(.01))
    tokens, positions, targets, mask, segments = self.inputs()
    batch = dict(inputs=tokens, inputs_position=positions, targets=targets,
                 targets_segmentation=mask, inputs_segmentation=segments)
    off = pyconfig.HyperParameters(SimpleNamespace(
        keys=dict(c.get_keys(), record_training_health_metrics=False)))
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      on_state, metrics = jax.jit(lambda s: train.train_step(
          m, c, None, s, dict(batch), jax.random.PRNGKey(9)))(state)
      off_state, off_metrics = jax.jit(lambda s: train.train_step(
          m, off, None, s, dict(batch), jax.random.PRNGKey(9)))(state)
    raw = float(metrics['scalar']['learning/raw_grad_norm'])
    clipped = float(metrics['scalar']['learning/grad_norm'])
    self.assertTrue(np.isfinite(raw))
    self.assertGreater(raw, clipped)
    self.assertLessEqual(clipped, .010001)
    self.assertNotIn('learning/raw_grad_norm', off_metrics['scalar'])
    for a, b in zip(jax.tree.leaves(on_state.params), jax.tree.leaves(off_state.params)):
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

  def test_pile_zero_token_mask_is_opt_in(self):
    import tensorflow as tf
    from input_pipeline._pile_data_processing import PileDatasets
    loader = PileDatasets.__new__(PileDatasets)
    loader.seq_len = 3
    loader.batch_size = 1
    loader.pad_id = 0
    loader.mix_attn = False
    data = {'input_ids': tf.constant([[9, 0, 2, 3]]),
            'labels': tf.constant([[9, 0, 2, -100]])}
    loader.loss_mask = 'nonnegative'
    np.testing.assert_array_equal(loader.convert(data)['targets_segmentation'], [[1, 1, 0]])
    loader.loss_mask = 'pad'
    np.testing.assert_array_equal(loader.convert(data)['targets_segmentation'], [[0, 1, 1]])

  def test_fusion_matches_plain_pythia(self):
    # Use identical nonzero affine parameters to expose LayerNorm/bias mistakes.
    for parallel in (False, True):
      with self.subTest(parallel=parallel):
        c = self.config(scan_layers=False, pythia_parallel_residual=parallel)
        m, mesh, p = self.init(c)
        p = nn.unbox(p)
        for path, a in flatten_dict(p).items():
          if path[-1] in ('scale', 'bias'):
            value = jnp.linspace(.8, 1.2, a.size) if path[-1] == 'scale' else jnp.linspace(-.07, .09, a.size)
            target = p
            for key in path[:-1]:
              target = target[key]
            target[path[-1]] = value.reshape(a.shape)
        fc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), pythia_use_fusion=True)))
        fm, _ = self.model(fc)
        fp = copy.deepcopy(p)
        for i in range(c.num_decoder_layers):
          layer = fp['decoder'][f'layers_{i}']
          layer['pre_self_attention_layer_norm'] = layer.pop('input_layernorm')
          layer['post_self_attention_layer_norm'] = layer.pop('post_attention_layernorm')
          fp['decoder'][f'layers_{i}'] = {'block': layer}
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
          def loss(model, params):
            return model.apply({'params': params}, *self.inputs(), enable_dropout=False)[0]
          np.testing.assert_allclose(loss(fm, fp), loss(m, p), atol=2e-6, rtol=2e-6)
          g = jax.grad(lambda p: jnp.mean(loss(m, p)))(p)
          fg = jax.grad(lambda p: jnp.mean(loss(fm, p)))(fp)
        for i in range(c.num_decoder_layers):
          a, b = g['decoder'][f'layers_{i}'], fg['decoder'][f'layers_{i}']['block']
          for name in ('self_attention', 'mlp'):
            for x, y in zip(jax.tree.leaves(a[name]), jax.tree.leaves(b[name])):
              np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-5)

  def test_full_history_remat_gradients(self):
    c = self.config(exp_class='MuddPythiaMediumPropT2048SerialMLP',
                    base_num_decoder_layers=3, base_mlp_dim=256,
                    pythia_rotary_dim=4, remat_policy='full')
    m, mesh, p = self.init(c)
    p = nn.unbox(p)
    flat = flatten_dict(p)
    for i in range(1, 3):
      self.assertEqual(p['decoder'][f'layers_{i}']['compose']['mlp']['dynamic_dense_conn2']['kernel'].shape[-2:], (4, i + 1))
    self.assertFalse(any('compose_break' in path for path in flat))
    self.assertFalse(any('bias' == path[-1] for path in flat if 'pre_dense_proj1_norm' in path))  # Router RMSNorm stays RMSNorm.
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      def loss(p):
        return jnp.mean(m.apply({'params': p}, *self.inputs(), enable_dropout=False)[0])
      step = jax.jit(jax.value_and_grad(loss))
      value, grad = step(p)
      self.assertTrue(np.isfinite(value))
      for a in jax.tree.leaves(grad):
        self.assertTrue(np.isfinite(a).all())
      grads = flatten_dict(grad)
      for i in range(3):
        self.assertGreater(float(jnp.linalg.norm(grads[('decoder', f'layers_{i}', 'block', 'mlp', 'wi', 'kernel')])), 0.)
      for i in range(1, 3):
        self.assertGreater(float(jnp.linalg.norm(grads[('decoder', f'layers_{i}', 'compose', 'mlp', 'dynamic_dense_conn2', 'kernel')])), 0.)
      # Repeated rematerialized calls must not mutate or duplicate the history.
      repeated, repeated_grad = step(p)
      np.testing.assert_array_equal(value, repeated)
      for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(repeated_grad)):
        np.testing.assert_array_equal(a, b)
      np.testing.assert_allclose(value, loss(p), atol=1e-6, rtol=1e-6)

  def test_full_size_budget_and_initialization(self):
    counts = {}
    for serial in (False, True):
      arm = 'MuddPythiaMediumPropT2048' + ('SerialMLP' if serial else '')
      c = self.config(exp_class=arm, base_emb_dim=1200, base_num_decoder_layers=18,
                      base_num_query_heads=16, base_num_kv_heads=16, head_dim=75,
                      vocab_size=50432)
      self.assertEqual(c.pythia_parallel_residual, not serial)
      self.assertFalse(c.scan_layers)
      self.assertTrue(c.dynamic_mlp_dim)
      self.assertEqual(c.init_method, 'pythia')
      m, mesh = self.model(c)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = jax.eval_shape(lambda key: m.init(key, *self.inputs(), enable_dropout=False)['params'], jax.random.key(0))
      p = nn.unbox(p)
      flat = flatten_dict(p)
      counts[serial] = {k: a.shape for k, a in flat.items()}
      self.assertEqual(sum(math.prod(a.shape) for a in flat.values()), 434356347)
      self.assertEqual(len(flat), 414)
      for i in range(18):
        layer = p['decoder'][f'layers_{i}']
        width = round(4800 * (i / 17 + .5) / 128) * 128
        self.assertEqual(layer['block']['mlp']['wi']['kernel'].shape, (1200, width))
        if i:
          self.assertEqual(layer['compose']['mlp']['dynamic_dense_conn2']['kernel'].shape[-2:], (4, i + 1))
    self.assertEqual(counts[False], counts[True])
    # The serial toggle changes the circuit without changing initialized arrays.
    c = self.config(exp_class='MuddPythiaMediumPropT2048SerialMLP',
                    base_num_decoder_layers=3, base_mlp_dim=256, pythia_rotary_dim=4)
    _, _, p = self.init(c)
    pc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), pythia_parallel_residual=True)))
    _, _, pp = self.init(pc)
    a, b = flatten_dict(nn.unbox(p)), flatten_dict(nn.unbox(pp))
    self.assertEqual(set(a), set(b))
    for k in a:
      np.testing.assert_array_equal(a[k], b[k], err_msg=str(k))

  def test_reject_invalid_full_history_scan(self):
    with self.assertRaisesRegex(ValueError, 'scan_layers=False'):
      self.config(exp_class='MuddPythiaMediumPropT2048SerialMLP', scan_layers=True, pythia_rotary_dim=4)


if __name__ == '__main__':
  unittest.main()

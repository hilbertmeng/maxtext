"""O row-key bottleneck placement, initialization, and gradient regression tests."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
from layers.attentions import BamAttention
import bam_local_fetch_test


class ORowLoraTest(absltest.TestCase):
  config = bam_local_fetch_test.LocalFetchTest.config

  def test_placement_and_gradients(self):
    for scope in ('Fetch', 'Local', 'All'):
      cfg = self.config('BamMediumIndependentLLFBAlignedRow' + scope + 'ORowR256Gelu')
      cfg.get_keys()['bam_o_row_bottleneck_dim'] = 16
      self.assertTrue(cfg.scan_layers)
      self.assertTrue(cfg.record_training_health_metrics)
      for local in (True, False):
        with self.subTest(scope=scope, local=local):
          mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
          module = BamAttention(
              config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
              max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
              attention_kernel='dot_product_chunk', dtype=cfg.dtype,
              layer_mode='local_qk+local_o' if local else 'local_qk+full',
              attention_type=cfg.attention_type)
          x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
          m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), dtype=cfg.dtype)
          args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
          variables = module.init({'params': jax.random.key(3), 'aqt': jax.random.key(4)},
                                  *args, M_in=m, deterministic=True, layer_index=2)
          params = variables['params']
          active = scope == 'All' or (scope == 'Local') == local
          self.assertEqual('W_R_row_down' in params, active)
          self.assertEqual(params['W_R']['kernel'].value.shape, (128, 2, 1, 8 if active else 40))
          self.assertNotIn('bias', params['W_R'])
          if active:
            down = params['W_R_row_down']['kernel'].value
            up = params['W_R_row_up']['kernel'].value
            self.assertEqual(down.shape, (128, 16))
            self.assertEqual(up.shape, (16, 2, 1, 32))
            self.assertGreater(float(jnp.linalg.norm(down)), 0.)
            np.testing.assert_array_equal(up, 0)
          def loss(p):
            y, mm = module.apply({'params': p}, *args, M_in=m,
                                 deterministic=True, layer_index=2)
            return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(mm.astype(jnp.float32)**2)
          value, grads = jax.value_and_grad(loss)(params)
          self.assertTrue(bool(jnp.isfinite(value)))
          self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads)))
          if active:
            self.assertGreater(float(jnp.linalg.norm(grads['W_R_row_up']['kernel'].value)), 0.)
            np.testing.assert_array_equal(grads['W_R_row_down']['kernel'].value, 0)
          if scope == 'All' and local:
            # New named projections must not perturb unrelated random initializers.
            cfg.get_keys()['bam_o_row_bottleneck_dim'] = 0
            baseline = module.init({'params': jax.random.key(3), 'aqt': jax.random.key(4)},
                                   *args, M_in=m, deterministic=True, layer_index=2)
            expected = module.apply(baseline, *args, M_in=m,
                                    deterministic=True, layer_index=2)
            cfg.get_keys()['bam_o_row_bottleneck_dim'] = 16
            for name in baseline['params']:
              if name != 'W_R':
                for before, after in zip(jax.tree.leaves(baseline['params'][name]),
                                         jax.tree.leaves(params[name])):
                  np.testing.assert_array_equal(before, after)
            actual = module.apply(variables, *args, M_in=m,
                                  deterministic=True, layer_index=2)
            for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
              np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
  absltest.main()

"""Pure-y_std address projection: paired initialization, WD, gradients and capture."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from layers.attentions import BamAttention


class StdTailWriteTest(absltest.TestCase):
  def test_exporter_has_every_local_and_fetch_layer(self):
    from types import SimpleNamespace
    import train
    tree = {'intermediates': {'decoder': {'layers': {}}}}
    slots = tree['intermediates']['decoder']['layers']
    for name in ('local_0', 'local_1', 'fetch_2'):
      slots[name] = {'block': {'self_attention': {
          'std_tail_write_' + metric: (jnp.array([1., 2.]),)
          for metric in ('source_rms', 'dynamic_rms', 'bias_rms', 'pre_norm_rms',
                         'post_norm_rms', 'epsilon_fraction', 'projection_rms')}}}
    output = {'scalar': {}}
    train.record_std_tail_write_metrics(output, tree,
        SimpleNamespace(bam_local_fetch_block_size=3, base_num_decoder_layers=6))
    self.assertLen(output['scalar'], 48)
    self.assertEqual(output['scalar']['bam/std_tail_write/layer_005/bias_over_dynamic_rms'], 1.)

  def test_modules_pair_and_write_source(self):
    from bam_local_fetch_test import LocalFetchTest
    import max_utils
    import train
    helper = LocalFetchTest()
    self.addCleanup(helper.doCleanups)
    configs = [helper.config('BamMediumIndependentLLFBAlignedRowStdTailWrite' + suffix)
               for suffix in ('Orth', 'Normal')]
    for cfg in configs:
      cfg.get_keys()['bam_write_v_bottleneck_dim'] = None  # helper normally shrinks P_loc
    x = jax.random.normal(jax.random.key(31), (1, 8, 128), configs[0].dtype)
    m = jax.random.normal(jax.random.key(32), (1, 8, 32, 32), configs[0].dtype)
    std = jax.random.normal(jax.random.key(33), (1, 8, 2, 64), configs[0].dtype)
    o = jax.random.normal(jax.random.key(34), std.shape, configs[0].dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    for mode in ('local_qk+local_o', 'local_qk+full'):
      variants = []
      for cfg in configs:
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2,
            head_dim=64, max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
            attention_kernel='dot_product_chunk', dtype=cfg.dtype,
            layer_mode=mode, attention_type=cfg.attention_type)
        variables = module.init({'params':jax.random.key(35), 'aqt':jax.random.key(36)},
            *args, M_in=m, deterministic=True, layer_index=2)
        params = variables['params']
        self.assertFalse(any('P_loc' in '/'.join(k) for k in flatten_dict(params)))
        self.assertEqual(params['write_std_tail_projection'].value.shape, (32, 32))
        self.assertEqual(params['write_std_tail_bias'].value.shape, (2, 32))
        np.testing.assert_array_equal(params['write_std_tail_bias'].value, 0)
        wd = train.get_wd_tree(cfg, params)
        self.assertEqual(wd['write_std_tail_bias'], 0.)
        self.assertEqual(wd['write_std_tail_projection'], cfg.adam_weight_decay)
        variants.append(params)
        matrix = params['write_std_tail_projection'].value
        if cfg.bam_std_tail_projection_init == 'orthogonal':
          np.testing.assert_allclose(matrix.T @ matrix, jnp.eye(32), atol=1e-5)
        else:
          self.assertAlmostEqual(float(jnp.std(matrix)), .006, delta=.0006)

        def write(pp, oo, ss):
          return module.apply({'params': pp}, oo, x, m, y_std=ss, method=module._write)[0]
        # Addresses ignore o_head tail and y_std's first (data) half.
        np.testing.assert_array_equal(write(params, o, std), write(params, o.at[..., 32:].add(7), std))
        np.testing.assert_array_equal(write(params, o, std), write(params, o, std.at[..., :32].add(7)))
        self.assertGreater(float(jnp.linalg.norm((write(params, o, std.at[..., 32:].add(7))
                                                  - write(params, o, std)).astype(jnp.float32))), 0)
        def loss(pp):
          y, mm = module.apply({'params':pp}, *args, M_in=m, deterministic=True, layer_index=2)
          return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(mm.astype(jnp.float32)**2)
        grad = jax.grad(loss)(params)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grad)))
        self.assertGreater(float(jnp.linalg.norm(grad['write_std_tail_projection'].value)), 0.)
        self.assertGreater(float(jnp.linalg.norm(grad['write_std_tail_bias'].value)), 0.)
        _, health = module.apply({'params':params}, o, x, m, y_std=std,
                                method=module._write, mutable=['intermediates'])
        self.assertLen(health['intermediates'], 7)
        frac = health['intermediates']['std_tail_write_epsilon_fraction'][0]
        self.assertTrue(0 <= float(frac) <= 1)
      left, right = map(flatten_dict, variants)
      self.assertEqual(left.keys(), right.keys())
      for path in left:
        if path != ('write_std_tail_projection',):
          for a, b in zip(jax.tree.leaves(left[path]), jax.tree.leaves(right[path])):
            np.testing.assert_array_equal(a, b, err_msg=str(path))


if __name__ == '__main__':
  absltest.main()

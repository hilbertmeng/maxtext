"""Dual LocalV semantic, initialization, health and scan integration checks."""
import unittest
from flax import core, linen as nn
from flax.linen import partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import train
import train_compile
from layers.attentions import BamAttention
from layers.bam_local_v_health import local_v_dual_stats, READ_NAMES
from bam_local_fetch_test import LocalFetchTest


EXP = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRowSharedRead'


class DualLocalVTest(unittest.TestCase):
  config = LocalFetchTest.config

  def test_stats_sign_bins_and_empty_bins(self):
    x = jnp.arange(1, 41, dtype=jnp.float32).reshape(1, 2, 2, 10)
    gate = jnp.ones((1, 2, 2, 2))
    stats = local_v_dual_stats(x, -x, x, (gate, gate, gate), 4, 2)
    self.assertTrue(jnp.all(jnp.isfinite(jnp.concatenate([s.ravel() for s in stats.values()]))))
    r = stats['local_v_dual_read_stats']
    np.testing.assert_allclose(r[:, READ_NAMES.index('cosine')], -1, atol=1e-6)
    np.testing.assert_allclose(r[:, READ_NAMES.index('sum_rms')], 0, atol=1e-6)
    np.testing.assert_allclose(r[:, READ_NAMES.index('interference_fraction')], -1, atol=1e-6)
    np.testing.assert_allclose(stats['local_v_dual_gate_stats'][..., -1], 1)
    np.testing.assert_allclose(stats['local_v_dual_bin_stats'][..., :4, :], 0)

  def test_module_initialization_gradients_and_disabled_shared_reference(self):
    configs = [self.config(EXP.removesuffix('SharedRead')), self.config(EXP)]
    for cfg in configs:
      cfg.get_keys()['dtype'] = jnp.float32
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(configs[0]), configs[0].mesh_axes)
    modules = [BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
        attention_kernel='dot_product_chunk', dtype=jnp.float32,
        layer_mode='local_qk+local_o', attention_type=cfg.attention_type) for cfg in configs]
    x = jax.random.normal(jax.random.key(1), (1, 8, 128))
    m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32))
    args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
    kw = dict(M_in=m, deterministic=True, layer_index=2)
    variables = [mod.init({'params':jax.random.key(3),'aqt':jax.random.key(4)}, *args, **kw) for mod in modules]
    p0, p1 = (v['params'] for v in variables)
    self.assertEqual(set(p1)-set(p0), {'W_lv_shared_gate', 'W_lv_shared_gate_b0'})
    for name, value in flatten_dict(p0).items():
      for a,b in zip(jax.tree.leaves(value), jax.tree.leaves(flatten_dict(p1)[name])):
        np.testing.assert_array_equal(a,b)
    wd = train.get_wd_tree(configs[1], p1)
    self.assertEqual(wd['W_lv_shared_gate_b0'], 0.)
    self.assertEqual(wd['W_lv_gate_b0'], 0.)
    self.assertEqual(wd['W_R_gate_b0'], 0.)
    # W_R starts at zero: probe an active read, not its dormant initialization.
    p0, p1 = core.unfreeze(p0), core.unfreeze(p1)
    kernel = p1['W_R']['kernel']
    active = kernel.replace(value=.006 * jax.random.normal(jax.random.key(9), kernel.value.shape))
    p0['W_R']['kernel'] = active
    p1['W_R']['kernel'] = active
    def loss(params):
      (y,nm), captured = modules[1].apply({'params':params}, *args, **kw, mutable=['intermediates'])
      return jnp.mean(y*y)+jnp.mean(nm*nm), captured
    (_, cap), grad = jax.value_and_grad(loss, has_aux=True)(p1)
    self.assertIn('local_v_dual_gate_stats', cap['intermediates'])
    self.assertGreater(float(jnp.linalg.norm(grad['W_lv_shared_gate_b0'].value)), 0.)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(grad)))
    disabled = core.unfreeze(p1)
    bias = disabled['W_lv_shared_gate_b0']
    disabled['W_lv_shared_gate_b0'] = bias.replace(value=jnp.full_like(bias.value,-1e6))
    expected = modules[0].apply({'params':p0},*args,**kw)
    got = modules[1].apply({'params':disabled},*args,**kw)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(got)):
      np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)

  def test_train_metrics_actual_block_scan_and_non_scan(self):
    for scan in (True,False):
      with self.subTest(scan=scan):
        cfg = self.config(EXP)
        cfg.get_keys().update(base_num_decoder_layers=6, num_decoder_layers=6,
            bam_layer_modes=['local_qk+local_o','local_qk+local_o','local_qk+full']*2,
            scan_layers=scan, bam_pair_scan=scan, vocab_size=128)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        args,_,shardings,model = train_compile.get_shaped_inputs(mesh,cfg)
        with mesh, partitioning.axis_rules(cfg.logical_axis_rules):
          _, metrics = jax.eval_shape(lambda state,data,rng: train.train_step(
              model,cfg,shardings,state,data,rng),*args)
        self.assertIn('learning/raw_grad_norm',metrics['scalar'])
        for layer in (0,1,3,4):
          self.assertIn(f'bam/local_v_dual/col/layer_{layer:03d}/cosine',metrics['scalar'])
        self.assertFalse(any('/layer_002/' in k or '/layer_005/' in k for k in metrics['scalar'] if k.startswith('bam/local_v_dual')))


if __name__ == '__main__':
  unittest.main()

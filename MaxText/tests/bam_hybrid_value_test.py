"""Per-head residual/M value split: real parameter pruning and fused equivalence."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from common_types import MODEL_MODE_TRAIN
from layers import fusion
from layers.models import Transformer

PREFIX = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1HybridV4'


class HybridValueTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=2, keep=True, length=4):
    c = self.config(PREFIX + ('Keep' if keep else 'Drop') + 'LocalOTruePile',
        bam_splash_attention=False, base_emb_dim=150, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=6, base_mlp_dim=32, vocab_size=32,
        dtype='float32', weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False, max_target_length=length,
        max_prefill_predict_length=length, query_chunk_size=length,
        bam_standard_value_heads=1, bam_layer_modes=['local_qk+local_v+local_o']*6,
        bam_attention_write_layers=list(range(5)), bam_mlp_write_layers=list(range(5)),
        mlp_dim_by_block=None, bam_memory_mlp_dim=16,
        bam_memory_mlp_dim_by_layer=None, bam_memory_mlp_last_write_dim=0,
        bam_final_local_layer_count=2, bam_final_local_mlp_dim=None, bam_last_mlp_dim=0,
        emb_bam_num_head=2, bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_formal_budget_and_pruned_parameter_heads(self):
    for keep, total in [(True, 432113632), (False, 432113056)]:
      c = self.config(PREFIX + ('Keep' if keep else 'Drop') + 'LocalOTruePile')
      c.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      actual = sum(math.prod(v.shape) for v in flat.values())
      print('HYBRID_VALUE_PARAMS', keep, actual, flush=True)
      self.assertEqual(actual, total)
      self.assertNotIn('bam_mlp_dim_by_layer', c.get_keys())
      self.assertIsNone(c.mlp_dim_by_block)
      self.assertIsNone(c.bam_final_local_mlp_dim)
      self.assertEqual(c.mlp_dim, 2469 if keep else 2477)
      self.assertEqual(c.bam_local_fetch_block_size, 2)
      self.assertEqual(c.bam_attention_write_layers, list(range(17)))
      self.assertIsNone(c.bam_mlp_write_layers)
      self.assertEqual(c.bam_mlp_write_every, 1)
      self.assertTrue(c.bam_skip_final_mlp_write)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      ratio = 3600*c.mlp_dim/(3600*c.bam_memory_mlp_dim+458000)
      self.assertLess(abs(ratio-2), .003)
      # The last explicit layer has unscanned shapes and no dead write parameters.
      tail = {p[p.index('self_attention')+1:]:v for p,v in flat.items()
              if 'final_local_layer_1' in p and 'self_attention' in p}
      self.assertEqual(tail['value', 'kernel'].shape, (1200, 4, 75))
      self.assertEqual(tail['static_v_key',].shape, (32, 12))
      self.assertEqual(tail['W_lv_gate', 'kernel'].shape, (1200, 12, 1))
      self.assertEqual(tail['general_v_static_gate', 'kernel'].shape, (1200, 12))
      o_heads = 16 if keep else 12
      self.assertEqual(tail['W_R_up', 'kernel'].shape, (256, o_heads, 1, 32))
      self.assertEqual(tail['general_vo_pre_bias',].shape, (o_heads, 1, 32))
      self.assertEqual(tail['static_o_key',].shape, (32, o_heads))
      self.assertEqual(tail['general_o_static_gate', 'kernel'].shape, (1200, o_heads))
      for p in flat:
        if 'final_local_layer_1' in p:
          self.assertFalse(set(p) & {'memory_mlp', 'P_loc_down', 'P_loc_up', 'W_gw', 'mlp_address_up'})

  def test_native_head_sources_and_local_o_mask(self):
    for keep in (True, False):
      c = self.tiny(keep=keep)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      mod = fusion.SubDecoderLayer(c, mesh, quant=None, layer_inx=4)
      x = jax.random.normal(jax.random.key(1), (1, 4, 150))
      m = jax.random.normal(jax.random.key(2), (1, 4, 75, 32))
      pos = jnp.arange(4)[None]; seg = jnp.ones((1, 4), jnp.int32)
      args = (x, seg, pos, seg, None, True, MODEL_MODE_TRAIN, None)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = nn.unbox(mod.init(jax.random.key(3), *args, M_in=m, is_global=True)['params'])
        seen = {'value': [], 'vo': [], 'q': [], 'k': []}
        def capture(next_fun, args, kwargs, ctx):
          out = next_fun(*args, **kwargs)
          if ctx.method_name == 'kv_projection' and kwargs.get('proj_name') == 'value':
            seen['value'].append(out)
          if ctx.method_name == '_independent_local_vo': seen['vo'].append(out)
          if ctx.method_name == '_read_direct_qk_c8': seen[args[0]].append(out)
          return out
        with nn.intercept_methods(capture):
          y, _ = mod.apply({'params': p}, *args, M_in=m, is_global=True)
          z, _ = mod.apply({'params': p}, *args, M_in=-m, is_global=True)
      np.testing.assert_array_equal(*seen['value'])
      self.assertEqual(seen['value'][0].shape, (1, 4, 1, 75))
      v, o = seen['vo'][0]
      np.testing.assert_array_equal(v[..., 0, :], 0)
      self.assertGreater(float(jnp.max(jnp.abs(v[..., 1, :]))), 1e-6)
      if keep: self.assertGreater(float(jnp.max(jnp.abs(o[..., 0, :]))), 1e-6)
      else: np.testing.assert_array_equal(o[..., 0, :], 0)
      for arm in ['q', 'k']:
        for head in range(2):
          self.assertGreater(float(jnp.max(jnp.abs(seen[arm][0][..., head, :]-seen[arm][1][..., head, :]))), 1e-6)
      self.assertGreater(float(jnp.max(jnp.abs(y-z))), 1e-6)

  def test_scan_causality_and_live_preterminal_writes(self):
    for keep in (True, False):
      c = self.tiny(keep=keep)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quant=None)
      tok = jnp.array([[1, 2, 3, 4]], jnp.int32); pos = jnp.arange(4)[None]; seg = jnp.ones_like(tok)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = nn.unbox(model.init(jax.random.key(4), tok, pos, tok, seg, seg, enable_dropout=False)['params'])
        call = lambda q,t: model.apply({'params': q}, t, pos, t, seg, seg, enable_dropout=False)[0]
        np.testing.assert_allclose(call(p,tok)[:,:3], call(p,tok.at[:,3].set(7))[:,:3], rtol=1e-6, atol=1e-6)
        grad = jax.jit(jax.grad(lambda q: jnp.mean(call(q,tok)**2)))(p)
      for path, v in flatten_dict(grad).items():
        self.assertTrue(np.isfinite(np.asarray(v)).all(), str(path))
        if 'final_local_layer_0' in path and ('memory_mlp' in path or 'P_loc_up' in path):
          self.assertGreater(float(jnp.sum(v*v)), 0, str(path))

  check_pallas = StandardValueTest.check_pallas
  def test_pallas_keep_all_gradients(self): self.check_pallas(True)
  def test_pallas_drop_all_gradients(self): self.check_pallas(False)


if __name__ == '__main__': unittest.main()

"""All-in depth budgets, explicit Mudd readout expansion and V5 head geometry."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
from bam_prop_test_utils import MLPWriteTest
from layers.models import Transformer


MHA = 'Llama2MediumPropL{}TotalBudgetTruePile'
MUDD = {
    12: 'MuddLlama2MediumPropL12TotalBudgetRouterFixedTruePile',
    18: 'MuddLlama2MediumPropRouterFixedTruePile',
    24: 'MuddLlama2MediumPropL24TotalBudgetRouterFixedPrePostNormTruePile',
}
BAM = 'BamMediumPropV5SiTU47L{}TotalBudgetTruePile'


class TotalBudgetDepthTest(MLPWriteTest, unittest.TestCase):
  def shaped(self, name, **overrides):
    c = self.config(name, attention='dot_product')
    c.get_keys().update(overrides)
    c.get_keys().update(bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    t = jnp.ones((1, 4), jnp.int32)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      tree = jax.eval_shape(lambda key: Transformer(c, mesh, quant=None).init(
          key, t, jnp.arange(4)[None], t, t, enable_dropout=False)['params'], jax.random.key(9876))
    return c, flatten_dict(nn.unbox(tree))

  def test_mha_all_in_budgets_and_uniform_mlp(self):
    for depth, d, width, total in ((12, 1424, 3728, 432111376), (24, 1056, 2874, 432133152)):
      c, flat = self.shaped(MHA.format(depth))
      self.assertEqual((c.emb_dim, c.head_dim, c.num_query_heads), (d, d // 16, 16))
      embedding = sum(math.prod(x.shape) for p, x in flat.items()
                      if 'token_embedder' in p or 'logits_dense' in p)
      self.assertEqual(embedding, 2 * 50432 * d)
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()), total)
      self.assertEqual(total, 2 * 50432 * d + depth * (4*d*d + 3*d*width + 2*d) + d)
      self.assertLess(abs(total / 432121200 - 1), 3e-5)
      self.assertTrue(c.scan_layers)
      self.assertFalse(c.dynamic_mlp_dim or c.bam_last_mlp_dim)
      self.assertEqual((c.steps, c.learning_rate_schedule_steps), (13500, 13500))

  def test_mudd_router_placement_and_unrepaid_overhead(self):
    for depth, total in ((12, 433257201), (18, 433911307), (24, 434683717)):
      c, flat = self.shaped(MUDD[depth])
      self.assertEqual(c.mudd_router_hidden_expand, [1] * (depth - 1) + [4])
      self.assertEqual((c.mudd_prenorm, c.mudd_postnorm), (depth == 24, depth == 24))
      self.assertFalse(c.scan_layers)
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()), total)
      widths = [flat[('decoder', f'layers_{i}', 'block', 'mlp', 'wi_0', 'kernel')].shape[-1]
                for i in range(depth)]
      self.assertEqual(sum(widths), depth * c.mlp_dim)
      for i in range(1, depth):
        expected = ((4 * (i + 1)) // 64 + 1) * 64
        self.assertEqual(flat[('decoder', f'layers_{i}', 'compose', 'mlp',
                               'dynamic_dense_conn1', 'kernel')].shape, (c.emb_dim, expected))
      final = ((4 * (depth + 1)) // 64 + 1) * 64
      self.assertEqual(flat[('decoder', 'compose_final', 'mlp',
                             'dynamic_dense_conn1', 'kernel')].shape, (c.emb_dim, final))
      if depth == 18:
        # Only the two historically misplaced expansion sites change shape.
        _, old = self.shaped(MUDD[18], mudd_router_hidden_expand=None)
        self.assertEqual(sum(math.prod(x.shape) for x in old.values()), 434076747)
        changed = {p for p in flat if p not in old or flat[p].shape != old[p].shape}
        expected = {
            ('decoder', 'layers_16', 'compose', 'mlp', name, 'kernel')
            for name in ('dynamic_dense_conn1', 'dynamic_dense_conn2')
        } | {
            ('decoder', 'compose_final', 'mlp', name, 'kernel')
            for name in ('dynamic_dense_conn1', 'dynamic_dense_conn2')
        }
        self.assertEqual(flat.keys(), old.keys())
        self.assertEqual(changed, expected)

  def test_bam_budget_branch_ratio_and_terminal_writes(self):
    for depth, total in ((12, 432120776), (24, 432120064)):
      c, flat = self.shaped(BAM.format(depth))
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()), total)
      self.assertLess(abs(total / 432121200 - 1), 3e-6)
      body = {p: x for p, x in flat.items() if p[:2] == ('decoder', 'layers')}
      ordinary = sum(math.prod(x.shape) for p, x in body.items() if 'mlp' in p)
      memory = sum(math.prod(x.shape) for p, x in body.items() if 'memory_mlp' in p)
      control = sum(math.prod(x.shape) for p, x in body.items() if set(p) & {
          'mlp_address_down', 'mlp_address_up', 'mlp_address_bias',
          'mlp_write_gate', 'mlp_write_gate_bias'})
      self.assertLess(abs(ordinary / (memory + control) - 2), .001)
      final = {p for p in flat if 'final_local_layer' in p}
      self.assertTrue(final)
      self.assertFalse(any('memory_mlp' in p or 'mlp_address_down' in p or 'P_loc_down' in p
                           for p in final))
      self.assertEqual(c.bam_attention_write_layers, list(range(depth - 1)))
      self.assertEqual(c.bam_mlp_write_layers, list(range(depth - 1)))
      resolved = self.config(BAM.format(depth))
      self.assertTrue(resolved.bam_pallas_packed_read_keys and resolved.bam_pallas_fused_inputs)
      self.assertTrue(resolved.bam_pallas_core and resolved.bam_pallas_write)
      self.assertEqual((resolved.bam_pallas_body, resolved.bam_pallas_vmem_mib), ('v8t', 48))
      self.assertTrue(resolved.record_training_health_metrics)
      self.assertFalse(resolved.record_internal_nn_metrics or resolved.bam_record_m_shift_health)

  def test_new_head_geometries_native_pallas_all_gradients(self):
    # Reduce heads/MLPs/vocabulary only; preserve M pre-norm, SiTU and both scales.
    for depth, k, rope in ((12, 89, 22), (24, 66, 16)):
      c = self.config(BAM.format(depth), dtype='float32', weight_dtype='float32')
      c.get_keys().update(emb_dim=2*k, num_query_heads=2, num_kv_heads=2,
          num_decoder_layers=2, mlp_dim=24, vocab_size=32, emb_bam_num_head=2,
          bam_layer_modes=['local_qk+local_v+local_o']*2,
          bam_attention_write_layers=[0], bam_mlp_write_layers=[0],
          bam_memory_mlp_dim=12, bam_last_mlp_dim=24,
          bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
          bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8,
          bam_pallas_core=False, bam_pallas_write=False, max_target_length=128,
          max_prefill_predict_length=128, query_chunk_size=64)
      pc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
          bam_pallas_core=True, bam_pallas_write=True, bam_pallas_interpret=True)))
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      native, fused = (Transformer(cfg, mesh, quant=None) for cfg in (c, pc))
      tokens = jax.random.randint(jax.random.key(31), (1, 128), 0, 32)
      pos = (jnp.arange(128) % 64)[None]
      seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
      target, mask = (tokens + 1) % 32, jnp.ones_like(tokens)
      make = splash.make_splash_mha
      with self.subTest(depth=depth), mesh, nn.partitioning.axis_rules(c.logical_axis_rules), mock.patch.object(
          splash, 'make_splash_mha', side_effect=lambda *a, **kw: make(*a, **dict(kw, interpret=True))):
        params = nn.unbox(native.init(jax.random.key(32), tokens, pos, target, mask,
                                     seg, enable_dropout=False))
        def perturb(path, value):
          if path[-1].key == 'common_qk_read_scale': return value * -.65
          if path[-1].key == 'common_vo_read_scale': return value * .8
          return value
        params = jax.tree.map_with_path(perturb, params)
        def loss(model):
          return lambda p: jnp.mean(model.apply(p, tokens, pos, target, mask, seg,
                                                enable_dropout=False)[0].astype(jnp.float32))
        actual, expected = (jax.jit(jax.value_and_grad(loss(m)))(params) for m in (fused, native))
        np.testing.assert_allclose(actual[0], expected[0], rtol=3e-4, atol=2e-6)
        ga, ge = flatten_dict(actual[1]), flatten_dict(expected[1])
        self.assertEqual(ga.keys(), ge.keys())
        errors = []
        for path in ga:
          a, e = np.asarray(ga[path]), np.asarray(ge[path])
          self.assertTrue(np.isfinite(a).all(), path)
          error = np.max(np.abs(a - e)) / (np.max(np.abs(e)) + 1e-12)
          errors.append(float(error))
          self.assertLess(error, 5e-4, str(path))
        for scale in ('common_vo_read_scale', 'common_qk_read_scale'):
          self.assertGreater(sum(float(jnp.sum(v*v)) for p, v in ga.items() if p[-1] == scale), 0.)
        print('TOTAL_BUDGET_DEPTH_PALLAS', depth, k, rope, len(ga), max(errors), flush=True)


if __name__ == '__main__':
  unittest.main()

"""Shared post-attention features with independent generalized write heads."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train_compile
from common_types import MODEL_MODE_TRAIN
from layers import fusion, quantizations
from bam_general_write_vo_test import GeneralWriteVOTest


RUN = ('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
       'LogitDiagNegativeQK75SharedWritePostAttentionGeneralWriteTruePile')


class SharedGeneralWriteTest(GeneralWriteVOTest):
  def tiny(self, length=128):
    c = super().tiny(length)
    c.get_keys().update(
        bam_share_attention_mlp_write_activation=True,
        bam_shared_write_activation_input='post_attention',
        bam_attention_negative_diagonal=True,
        bam_local_qk_col_output_dim=75,
        bam_partial_rope_nope_dim=75)
    return c

  def test_shared_formal_budget_and_terminal(self):
    c = self.config(RUN)
    c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    args, *_ = train_compile.get_shaped_inputs(mesh, c)
    flat = flatten_dict(nn.unbox(args[0].params))
    count = lambda items: sum(math.prod(v.shape) for v in items)
    total = count(flat.values())
    self.assertEqual(total, 432110944)
    self.assertFalse(any('mlp_address_down' in p for p in flat))
    names = {'general_attention_static_address', 'general_attention_static_gate',
             'general_attention_static_gate_bias', 'general_mlp_static_address',
             'general_mlp_static_gate', 'general_mlp_static_gate_bias'}
    added = {p: v for p, v in flat.items() if set(p) & names}
    self.assertEqual(count(added.values()), 670752)
    self.assertFalse(any('final_local_layer' in p for p in added))
    self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim),
                     (2584, 1128, 3910))
    self.assertTrue(c.scan_layers and not c.bam_pair_scan)
    self.assertEqual(c.bam_attention_write_layers, list(range(17)))
    self.assertTrue(c.bam_skip_final_mlp_write)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write)
    self.assertEqual((c.bam_pallas_body, c.bam_pallas_vmem_mib), ('v8t', 48))
    print('SHARED_GENERAL_WRITE_BUDGET_OK', total, 'delta', total-432112192,
          flush=True)

  def test_shared_actual_post_attention_activation(self):
    c = self.tiny(length=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    block = fusion.SubDecoderLayer(c, mesh, quantizations.configure_quantization(c), layer_inx=0)
    x = jax.random.normal(jax.random.key(10), (1, 4, 150))
    matrix = jax.random.normal(jax.random.key(11), (1, 4, 75, 32))
    pos = jnp.arange(4)[None]
    seg = jnp.ones((1, 4), jnp.int32)
    args = (x, seg, pos, seg, None, True, MODEL_MODE_TRAIN, None)
    seen = {}
    def capture(next_fun, args, kwargs, context):
      name = context.module.name
      if context.method_name == '__call__' and name in ('P_loc_down', 'P_loc_up', 'mlp_address_up'):
        seen.setdefault(name, []).append(args[0])
      return next_fun(*args, **kwargs)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      params = block.init(jax.random.key(12), *args, M_in=matrix)
      with nn.intercept_methods(capture):
        block.apply(params, *args, M_in=matrix)
    self.assertEqual(len(seen['P_loc_down']), 1)
    expected = jax.nn.gelu(seen['P_loc_down'][0] @ nn.unbox(params)['params']['self_attention']['P_loc_down']['kernel'])
    np.testing.assert_allclose(seen['P_loc_up'][0], expected, atol=1e-6)
    np.testing.assert_array_equal(seen['P_loc_up'][0], seen['mlp_address_up'][0])

  def test_shared_native_pallas_read_write_logits_and_all_gradients(self):
    super().test_full_pallas_matches_native_logits_and_all_gradients()

  def test_shared_native_pallas_read_only_logits_and_all_gradients(self):
    super().test_read_only_pallas_matches_native_logits_and_all_gradients()


if __name__ == '__main__':
  unittest.main()

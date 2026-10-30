"""H transfer: target budget/scan/health and GELU+QKNorm native/Pallas gradients."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest
from layers.models import Transformer

BASE = ('BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1'
        'AllR640SeparateVOKeysLogitDiagNegativeQK80')
REFERENCE = BASE + 'LocalODynamicNormScale020'
H = BASE + 'CommonReadScaleUnscaledMLPContentEmbeddingFixed'
GAMMA = H + 'LocalODynamicNormScale020'
UNIT = H + 'ReadScaleInitOneLocalODynamicNormScale020'


class PythiaCommonReadRawWriteTest(MLPWriteTest, unittest.TestCase):
  def test_target_config_shapes_budget_and_no_unintended_changes(self):
    parent = self.config(REFERENCE)
    for run, gamma, init in [(H, False, None), (GAMMA, True, None), (UNIT, True, 1.)]:
      c = self.config(run)
      changed = {k for k, v in c.get_keys().items() if v != parent.get_keys()[k]}
      expected = {'exp_class', 'model_name', 'bam_common_vo_read_scale',
                  'bam_embedding_write_content_divisor', 'bam_mlp_write_content_divisor', 'compare_runs'}
      if not gamma:
        expected.add('bam_local_o_dynamic_norm_scale')
      if init is not None:
        expected.add('bam_common_vo_read_scale_init')
      self.assertEqual(changed, expected)
      self.assertEqual(c.compare_runs, [REFERENCE])
      self.assertEqual((c.mlp_dim, c.bam_memory_mlp_dim, c.bam_last_mlp_dim), (7808, 2944, 9984))
      self.assertEqual(c.bam_attention_write_layers, list(range(31)))
      self.assertTrue(c.bam_skip_final_mlp_write and c.scan_layers and not c.bam_pair_scan)
      self.assertEqual(c.bam_final_local_layer_count, 1)
      self.assertEqual(c.bam_common_vo_read_scale_init, init)
      self.assertEqual(c.bam_embedding_write_content_divisor, .0020825863)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertEqual(c.bam_mlp_write_content_divisor, 1.)
      self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat)
      self.assertEqual(c.mlp_activations, ['gelu_exact'])
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_fused_inputs)
      self.assertEqual((c.bam_pallas_body, c.bam_pallas_vmem_mib, c.remat_policy), ('v8t', 60, 'minimal'))
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 2775190880 + 32 + 65536 * gamma)
      self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items() if p[-1] == 'common_vo_read_scale'), 32)
      print('TARGET_CONFIG_BUDGET_OK', run, flush=True)

  def tiny(self, run, length=128, dtype='float32'):
    c = self.config(run, dtype=dtype, weight_dtype='float32')
    c.get_keys().update(emb_dim=160, num_query_heads=2, num_kv_heads=2,
        num_decoder_layers=3, mlp_dim=32, vocab_size=32,
        bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False,
        max_target_length=length, max_prefill_predict_length=length,
        query_chunk_size=64, bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1],
        bam_memory_mlp_dim=16, bam_last_mlp_dim=32, emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_unit_init_scan_values_and_signed_health(self):
    for run in (H, UNIT):
      c = self.tiny(run, length=4)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quant=None)
      tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = nn.unbox(model.init(jax.random.key(1), tok, jnp.arange(4)[None], tok, jnp.ones_like(tok)))
      for path, value in flatten_dict(p).items():
        if path[-1] != 'common_vo_read_scale':
          continue
        expected = np.ones(value.size) if run == UNIT else (
            1 / np.sqrt(1 + 2 * np.arange(2)) if 'layers' in path else [1 / np.sqrt(5)])
        np.testing.assert_allclose(np.asarray(value).reshape(-1), expected, rtol=1e-6)
    c = self.config(UNIT)
    shape = [1]
    shape.insert(c.param_scan_axis, 31)
    tree = {'params': {'decoder': {
        'layers': {'block': {'self_attention': {'common_vo_read_scale': -jnp.ones(shape)}}},
        'final_local_layer': {'block': {'self_attention': {'common_vo_read_scale': -jnp.ones(1)}}}}}}
    signed = {k: v for k, v in train.compute_params_norm(tree, c).items() if k.startswith('parameter_values/')}
    self.assertEqual(len(signed), 32)
    self.assertTrue(all(float(v) == -1. for v in signed.values()))
    self.assertTrue(all(v == 0. for v in flatten_dict(train.get_wd_tree(c, tree)).values()))

  def test_h_native_pallas_logits_and_all_gradients(self):
    CommonAllReadGammaTest.check_model(self, H, qknorm=True)

  def test_gamma_native_pallas_logits_and_all_gradients(self):
    CommonAllReadGammaTest.check_model(self, GAMMA, qknorm=True)

  def test_unit_gamma_native_pallas_logits_and_all_gradients(self):
    CommonAllReadGammaTest.check_model(self, UNIT, qknorm=True)


if __name__ == '__main__':
  unittest.main()

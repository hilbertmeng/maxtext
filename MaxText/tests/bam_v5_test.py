"""Regression suite for the retained V5 circuit, recipes and execution profiles.

Historical experiment suites remain next to this file for use at their recorded
runtime commits; this suite exercises the architecture supported by current main.
"""
import math
from pathlib import Path
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp

import exp
import max_utils
from bam_config import validate_bam_config
import bam_fast_kernel_test as fast_control
from bam_prop_test_utils import MLPWriteTest
import bam_xlprop_v5_test as xl_control
from layers.models import Transformer


PYTHIA = 'BamPythia2p8BHUnit300BFastKernelMPreNormQKScale'


class V5Test(MLPWriteTest, unittest.TestCase):
  # Reuse proven model controls without inheriting archived experimental tests.
  flags = fast_control.FastKernelTest.flags
  tiny = fast_control.FastKernelTest.tiny
  check_model = fast_control.FastKernelTest.check_model

  def test_thin_model_configuration_contracts(self):
    expected = {
        'BamMediumPropV5': (1200, 18, 16, 75, 32, 256, 2510, 1128, 3904),
        'BamXLPropV5': (1920, 28, 20, 96, 40, 400, 4048, 1827, 6277),
        PYTHIA: (2560, 32, 32, 80, 64, 640, 7808, 2944, 9984),
    }
    for name, values in expected.items():
      c = self.config(name)
      self.assertEqual((c.emb_dim, c.num_decoder_layers, c.num_query_heads, c.head_dim,
                        c.bam_v, c.bam_local_full_m_read_bottleneck_dim, c.mlp_dim,
                        c.bam_memory_mlp_dim, c.bam_last_mlp_dim), values)
      self.assertEqual((c.bam_write_v_bottleneck_dim, c.bam_mlp_write_address_rank,
                        c.emb_bam_v_bottleneck_dim), (values[5],)*3)
      self.assertEqual(c.bam_mlp_write_content_divisor, 1.)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertEqual(c.bam_m_shift_mode, 'read')
      self.assertEqual(c.bam_m_shift_init, .5)
      self.assertEqual(c.bam_m_read_norm, 'rms')
      self.assertTrue(c.bam_common_qk_read_scale and c.bam_common_vo_read_scale)
      self.assertFalse(c.bam_share_read_attention_write_down or c.bam_share_mlp_write_down)
      cls = getattr(exp, name)
      self.assertLess(len(cls.mro()), 15)
      self.assertFalse(any(x.__name__.startswith('BamMediumPropMShift') for x in cls.mro()))
    c = self.config(PYTHIA)
    self.assertEqual((c.decoder_block, c.mlp_activations), ('pythia', ['gelu_exact']))
    self.assertFalse(c.pythia_parallel_residual)
    self.assertEqual((c.init_method, c.bam_init_method, c.token_embedding_init_method),
                     ('pythia', '006normal', '006normal'))
    self.assertEqual((c.normalization_layer_epsilon, c.bam_write_rms_epsilon), (1e-5, 1e-6))
    self.assertTrue(c.cast_logits_to_fp32 and c.logits_fp32_output_bf16_backward)
    self.assertEqual((c.steps, c.learning_rate_schedule_steps, c.learning_rate), (143000, 143000, 1.6e-4))
    self.assertEqual(c.checkpoint_keep_steps, list(range(3000, 143001, 10000)))

  def test_hardware_profiles_do_not_change_model_or_plan(self):
    parent = self.config(PYTHIA).get_keys()
    for suffix, allowed in (
        ('V5p128', {'loss_chunk_size'}),
        ('V6eFast', {'remat_policy', 'bam_pallas_vmem_mib', 'bam_pallas_read_tile',
                     'bam_pallas_write_tile', 'bam_pallas_write_reverse_tile',
                     'bam_pallas_read_xhat_double_buffer', 'bam_pallas_read_chunk',
                     'bam_splash_ignore_segments'})):
      cfg = self.config(PYTHIA + suffix).get_keys()
      changed = {k for k, v in cfg.items() if v != parent.get(k)} - {'exp_class', 'model_name'}
      self.assertLessEqual(changed, allowed)
      self.assertTrue(changed)

  def test_nonbam_pythia_recipes_remain_available(self):
    for name in ('Pythia2p8B', 'MuddPythia3BSerialMLPShortTrain'):
      c = self.config(name)
      self.assertFalse(c.bam_enabled)
      self.assertEqual((c.decoder_block, c.mlp_activations, c.init_method),
                       ('pythia', ['gelu_exact'], 'pythia'))
      self.assertEqual(c.normalization_layer_epsilon, 1e-5)
      self.assertEqual(bool(c.dense_conn), name.startswith('Mudd'))

  def test_complete_checkpoint_parameter_trees_and_budgets(self):
    expected = {'BamMediumPropV5': 432112228, 'BamXLPropV5': 1432396996,
                PYTHIA: 2775190944}
    for name, count in expected.items():
      c = self.config(name)
      c.get_keys().update(bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      tok = jnp.ones((1, 4), jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        tree = jax.eval_shape(lambda key: Transformer(c, mesh, quant=None).init(
            key, tok, jnp.arange(4)[None], tok, tok, enable_dropout=False)['params'], jax.random.key(3))
      flat = flatten_dict(nn.unbox(tree))
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()), count)
      self.assertEqual(sum(x.size for p, x in flat.items() if p[-1] == 'common_qk_read_scale'), c.num_decoder_layers)
      tail = {p: x for p, x in flat.items() if 'final_local_layer' in p}
      self.assertTrue(tail)
      self.assertFalse(any('memory_mlp' in p or 'P_loc_down' in p for p in tail))

  def test_retired_options_fail_before_model_creation(self):
    c = dict(self.config('BamMediumPropV5').get_keys())
    for key in ('bam_content_adapter_write', 'bam_content_adapter_v', 'bam_content_adapter_o',
                'bam_mudd_tail_blocks', 'bam_memory_mlp_share_hidden', 'bam_pair_scan',
                'bam_local_o_dynamic_norm_scale', 'rmt_enabled'):
      with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'recorded runtime commit'):
        validate_bam_config(dict(c, **{key: True}))

  def test_mainline_does_not_import_legacy_implementations(self):
    root = Path(__file__).resolve().parents[1]
    for name in ('models.py', 'fusion.py', 'attentions.py', 'bam_pallas_v8.py'):
      source = (root/'layers'/name).read_text()
      for legacy in ('from layers import bam_mudd', 'from layers.rmt import',
                     'from layers.bam_content_adapter import', 'from layers.bam_pallas import',
                     'from layers.bam_pallas_v7 import'):
        self.assertNotIn(legacy, source)

  def test_xl_native_pallas_all_gradients(self):
    xl_control.XLPropV5Test.test_native_pallas_model_loss_and_all_gradients(self)

  def test_xl_packed_native_pallas_all_gradients(self):
    xl_control.XLPropV5Test.test_native_pallas_model_loss_and_all_gradients(self, packed=True)

  def test_pythia_native_pallas_all_gradients(self):
    self.check_model(PYTHIA)

  def test_pythia_bf16_training_objective_and_gradients(self):
    self.bf16_training_objective = True
    self.check_model(PYTHIA, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

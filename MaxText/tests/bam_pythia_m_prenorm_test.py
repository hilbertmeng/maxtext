"""300B and ShortTrain UnitH FastKernel + whole-M pre-norm and V/O / QK scales."""
import math
import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import train
from bam_fast_kernel_test import FastKernelTest
from layers.models import Transformer

PARENT = 'BamPythia2p8BHUnit300BFastKernel'
RUN = PARENT + 'MPreNormQKScale'


class PythiaMPreNormTest(FastKernelTest):
  def test_v6e_fast_config_only_changes_kernel_settings(self):
    v6e, fast = self.config(RUN + 'V6e'), self.config(RUN + 'V6eFast')
    changed = {k for k, v in fast.get_keys().items() if v != v6e.get_keys().get(k)}
    self.assertEqual(changed, {'exp_class', 'model_name', 'bam_pallas_vmem_mib', 'bam_pallas_read_tile',
                               'bam_pallas_write_tile', 'bam_pallas_write_reverse_tile',
                               'bam_pallas_read_xhat_double_buffer', 'bam_pallas_read_chunk',
                               'bam_splash_ignore_segments'})
    self.assertFalse(fast.mix_attn)

  def test_v6e_config_only_changes_remat_and_saves_only_splash(self):
    v5p, v6e = self.config(RUN), self.config(RUN + 'V6e')
    changed = {k for k, v in v6e.get_keys().items() if v != v5p.get_keys().get(k)}
    self.assertEqual(changed, {'exp_class', 'model_name', 'remat_policy'})
    self.assertEqual(v6e.remat_policy, 'full_save_splash')

    from jax._src.ad_checkpoint import saved_residuals
    from jax.ad_checkpoint import checkpoint_name
    from layers.models import get_remat_policy
    def f(x, w):
      h = checkpoint_name(jnp.tanh(x @ w), 'bam_splash_residual')
      return jnp.sum(jnp.sin(h @ w) @ w)
    x, w = jnp.ones((4, 8)), jnp.ones((8, 8))
    def kept(cfg):  # residuals other than the inputs
      return [why for _, why in saved_residuals(jax.checkpoint(f, policy=get_remat_policy(cfg)), x, w)
              if not why.startswith('from the argument')]
    self.assertEqual(len(kept(v6e)), 1)  # only the named Splash residual
    self.assertEqual(len(kept(v5p)), 3)  # minimal also keeps the two weight dots

  def test_formal_recipe_parameter_tree_and_health(self):
    parent, cfg = self.config(PARENT), self.config(RUN)
    changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys().get(k)}
    self.assertEqual(changed, {'exp_class', 'model_name', 'compare_runs',
                               'bam_m_read_norm', 'bam_common_qk_read_scale',
                               'logits_fp32_output_bf16_backward', 'record_logits_health_metrics'})
    self.assertEqual(cfg.compare_runs, ['BamPythia2p8BHUnit300B', 'Pythia_2.8B_2022',
                                      'PileMUDDPythia3BPlusOcdbtv6e',
                                      'MuddPythia3BSerialMLPKeepActivations'])
    self.assertTrue(cfg.cast_logits_to_fp32 and cfg.logits_fp32_output_bf16_backward)
    self.assertTrue(cfg.record_logits_health_metrics)
    self.assertEqual((cfg.steps, cfg.learning_rate_schedule_steps), (143000, 143000))
    self.assertEqual(cfg.bam_qk_logit_scale_dim, 'qk_dim')
    self.assertTrue(cfg.qk_norm and cfg.bam_qk_norm_after_concat)
    self.assertTrue(cfg.bam_common_vo_read_scale and cfg.bam_common_qk_read_scale)
    self.assertEqual(cfg.bam_common_vo_read_scale_init, 1.)
    self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim),
                     (7808, 2944, 9984))
    self.assertTrue(all(x % 128 == 0 for x in (cfg.mlp_dim, cfg.bam_memory_mlp_dim,
                                             cfg.bam_last_mlp_dim)))
    self.assertEqual((cfg.num_decoder_layers, cfg.num_query_heads, cfg.head_dim,
                      cfg.bam_k, cfg.bam_v), (32, 32, 80, 80, 64))
    self.assertTrue(cfg.scan_layers and not cfg.bam_pair_scan)
    self.assertEqual(cfg.bam_final_local_layer_count, 1)
    self.assertEqual((cfg.remat_policy, cfg.bam_splash_residual_name, cfg.bam_pallas_vmem_mib),
                     ('minimal_save_splash', 'bam_splash_residual', 63))
    self.assertTrue(cfg.bam_pallas_packed_read_keys and cfg.bam_pallas_fused_qk_norm)
    self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_write and cfg.bam_pallas_fused_inputs)
    self.assertTrue(cfg.record_training_health_metrics)
    self.assertFalse(cfg.record_internal_nn_metrics)

    def shape_tree(c):
      c.get_keys().update(max_target_length=8, max_prefill_predict_length=8,
                          bam_pallas_core=False, bam_pallas_write=False,
                          bam_splash_attention=False)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      tokens = jnp.ones((1, 8), jnp.int32)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        p = jax.eval_shape(lambda k: Transformer(c, mesh, quant=None).init(
            k, tokens, jnp.arange(8)[None], tokens, tokens, enable_dropout=False),
            jax.random.key(0))
      return flatten_dict({'params': nn.unbox(p['params'])})
    child, base = shape_tree(cfg), shape_tree(parent)
    new = {k: v for k, v in child.items() if k not in base}
    self.assertEqual(len(new), 2)  # Scanned L0..30 and the explicit L31 tail.
    self.assertTrue(all(k[-1] == 'common_qk_read_scale' for k in new))
    self.assertEqual(sum(math.prod(v.shape) for v in new.values()), 32)
    self.assertEqual({k: v.shape for k, v in child.items() if k in base},
                     {k: v.shape for k, v in base.items()})
    self.assertEqual(sum(math.prod(v.shape) for v in child.values()), 2775190944)
    self.assertEqual(sum(math.prod(v.shape) for v in base.values()), 2775190912)

    tree = {}
    for k, v in new.items():
      tree[k] = -jnp.ones(v.shape)
    params = unflatten_dict(tree)
    signed = {k: v for k, v in train.compute_params_norm(params, cfg).items()
              if k.startswith('parameter_values/')}
    self.assertEqual(len(signed), 32)
    self.assertTrue(all(float(v) == -1. for v in signed.values()))
    self.assertTrue(all(float(v) == 0. for v in flatten_dict(train.get_wd_tree(cfg, params)).values()))
    print('PYTHIA300B_M_PRENORM_BUDGET_OK +32 total2775190944, unchanged MLP/scan/plan/FastKernel', flush=True)

  def test_shorttrain_preserves_its_plan_temperature_and_parameter_shapes(self):
    short = 'BamPythia2p8BHUnitShortTrainFastKernel'
    parent, cfg, long = (self.config(n) for n in (short, short + 'MPreNormQKScale', RUN))
    changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys().get(k)}
    self.assertEqual(changed, {'exp_class', 'model_name', 'bam_m_read_norm', 'bam_common_qk_read_scale'})
    self.assertEqual(cfg.compare_runs, parent.compare_runs)
    self.assertEqual((cfg.steps, cfg.learning_rate_schedule_steps), (28000, 28000))
    self.assertIsNone(cfg.bam_qk_logit_scale_dim)
    self.assertTrue(cfg.qk_norm and cfg.bam_qk_norm_after_concat)
    self.assertEqual((cfg.remat_policy, cfg.bam_pallas_vmem_mib), ('minimal_save_splash', 63))
    # The long/short recipe differences are inherited exactly from their parents;
    # no new model shape or initialization difference is introduced by either port.
    structural = ('emb_dim', 'num_decoder_layers', 'num_query_heads', 'num_kv_heads',
                  'head_dim', 'bam_k', 'bam_v', 'mlp_dim', 'bam_memory_mlp_dim',
                  'bam_last_mlp_dim', 'bam_m_read_norm', 'bam_common_vo_read_scale',
                  'bam_common_vo_read_scale_init', 'bam_common_qk_read_scale',
                  'bam_local_full_m_read_bottleneck_dim', 'bam_write_v_bottleneck_dim',
                  'bam_mlp_write_address_rank', 'emb_bam_v_bottleneck_dim',
                  'bam_init_method', 'token_embedding_init_method', 'bam_pallas_fused_qk_norm')
    for k in structural:
      self.assertEqual(getattr(cfg, k), getattr(long, k), k)

  def test_pre_norm_fused_full_head_qknorm_all_gradients(self):
    # Inherited helper perturbs QKNorm vectors, signed Q/K scalars (including
    # zero/negative values), V/O scalars, and checks complete scanned-model gradients.
    self.check_model(RUN)

  def test_pre_norm_fused_full_head_qknorm_bf16_training_loss(self):
    self.bf16_training_objective = True
    self.check_model(RUN, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

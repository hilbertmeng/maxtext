"""Unit H with a Q/K-shared layer scalar before full-head QKNorm."""
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
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from bam_pythia_common_read_raw_write_test import PythiaCommonReadRawWriteTest, H
from layers.models import Transformer

PARENT = H + 'ReadScaleInitOne'
RUN = PARENT + 'CommonQKScale'


class CommonQKReadScaleTest(MLPWriteTest, unittest.TestCase):
  tiny = PythiaCommonReadRawWriteTest.tiny
  check_model = PythiaCommonReadRawWriteTest.check_model

  def test_formal_config_budget_health_and_decay(self):
    parent, cfg = self.config(PARENT), self.config(RUN)
    changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys()[k]}
    self.assertEqual(changed, {'exp_class', 'model_name', 'compare_runs', 'bam_common_qk_read_scale'})
    self.assertEqual(cfg.compare_runs, [PARENT])
    self.assertTrue(cfg.qk_norm and cfg.bam_qk_norm_after_concat)
    self.assertEqual(cfg.bam_common_vo_read_scale_init, 1.)
    self.assertFalse(cfg.bam_local_o_dynamic_norm_scale)
    self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (7808, 2944, 9984))
    self.assertEqual((cfg.steps, cfg.target_train_batch_size, cfg.remat_policy), (28000, 1024, 'minimal'))
    self.assertTrue(cfg.record_training_health_metrics)
    self.assertFalse(cfg.record_internal_nn_metrics)
    self.assertGreater(cfg.tensorboard_gcs_rotate_period, 0)
    cfg.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    args, *_ = train_compile.get_shaped_inputs(mesh, cfg)
    flat = flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 2775190880 + 64)
    self.assertEqual(sum(v.size for p, v in flat.items() if p[-1] == 'common_qk_read_scale'), 32)
    shape = [1]
    shape.insert(cfg.param_scan_axis, 31)
    tree = {'params': {'decoder': {
        'layers': {'block': {'self_attention': {'common_qk_read_scale': -jnp.ones(shape)}}},
        'final_local_layer': {'block': {'self_attention': {'common_qk_read_scale': -jnp.ones(1)}}}}}}
    signed = {k: v for k, v in train.compute_params_norm(tree, cfg).items() if k.startswith('parameter_values/')}
    self.assertEqual(len(signed), 32)
    self.assertTrue(all(float(v) == -1. for v in signed.values()))
    self.assertTrue(all(v == 0. for v in flatten_dict(train.get_wd_tree(cfg, tree)).values()))

  def test_unit_init_preserves_parent_outputs_and_all_old_gradients(self):
    cfg = self.tiny(RUN, length=4)
    pcfg = self.tiny(PARENT, length=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    child, parent = Transformer(cfg, mesh, quant=None), Transformer(pcfg, mesh, quant=None)
    tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
    pos, seg = jnp.arange(4)[None], jnp.ones_like(tok)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      cp, pp = (nn.unbox(m.init(jax.random.key(12), tok, pos, seg, tok)) for m in (child, parent))
      cf, pf = flatten_dict(cp), flatten_dict(pp)
      new = [p for p in cf if p[-1] == 'common_qk_read_scale']
      self.assertEqual(sum(cf[p].size for p in new), 3)
      for p in new:
        np.testing.assert_array_equal(cf[p], np.ones(cf[p].shape))
      for p in pf:
        np.testing.assert_array_equal(cf[p], pf[p])
      def objective(model, params):
        y = model.apply(params, tok, pos, seg, tok, enable_dropout=False)[0]
        return jnp.mean(y ** 2), y
      (cl, cy), cg = jax.value_and_grad(lambda p: objective(child, p), has_aux=True)(cp)
      (pl, py), pg = jax.value_and_grad(lambda p: objective(parent, p), has_aux=True)(pp)
    np.testing.assert_array_equal(cy, py)
    for p, value in flatten_dict(pg).items():
      np.testing.assert_allclose(flatten_dict(cg)[p], value, rtol=2e-5, atol=2e-7)
    self.assertTrue(any(float(jnp.max(jnp.abs(flatten_dict(cg)[p]))) > 0 for p in new))

  def test_requires_concat_and_full_head_qknorm(self):
    cfg = self.config(RUN).get_keys()
    validate_bam_config(cfg)
    for flag in ('bam_concat_qk', 'qk_norm', 'bam_qk_norm_after_concat'):
      with self.assertRaisesRegex(ValueError, 'Common Q/K scale'):
        validate_bam_config(dict(cfg, **{flag: False}))

  def test_native_pallas_logits_and_every_gradient(self):
    self.check_model(RUN)

  def test_bf16_native_pallas_gradients_against_f32(self):
    self.check_model(RUN, 'bfloat16')


if __name__ == '__main__':
  unittest.main()

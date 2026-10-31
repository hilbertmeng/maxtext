"""H memory-MLP biases: zero-init parity, live gradients, budget and no decay."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_pythia_common_read_raw_write_test import PythiaCommonReadRawWriteTest, H
from layers.models import Transformer

BIAS = H + 'MemoryMLPBias'


class PythiaMemoryMLPBiasTest(MLPWriteTest, unittest.TestCase):
  tiny = PythiaCommonReadRawWriteTest.tiny
  check_model = PythiaCommonReadRawWriteTest.check_model

  def test_formal_budget_and_resolved_recipe(self):
    old, new = self.config(H), self.config(BIAS)
    changed = {k for k, value in new.get_keys().items() if value != old.get_keys()[k]}
    self.assertEqual(changed, {'exp_class', 'model_name', 'bam_memory_mlp_use_bias', 'compare_runs'})
    self.assertFalse(old.bam_memory_mlp_use_bias)
    self.assertTrue(new.bam_memory_mlp_use_bias)
    self.assertEqual((new.mlp_dim, new.bam_memory_mlp_dim, new.bam_last_mlp_dim), (7808, 2944, 9984))
    self.assertEqual(new.tensorboard_gcs_rotate_period, 1000)
    self.assertEqual(new.checkpoint_save_timeout_seconds, 1200)
    self.assertTrue(new.record_training_health_metrics and not new.record_internal_nn_metrics)
    self.assertEqual((new.remat_policy, new.bam_pallas_vmem_mib), ('minimal', 60))
    shapes = []
    for cfg in [old, new]:
      cfg.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      shapes.append(flatten_dict(nn.unbox(args[0].params)))
    a, b = shapes
    extra = set(b) - set(a)
    self.assertEqual(len(extra), 2)
    self.assertTrue(all('memory_mlp' in p and p[-1] == 'bias' and p[-2] in ('wi', 'wo') for p in extra))
    self.assertFalse(any('final_local_layer' in p for p in extra))
    self.assertEqual(sum(math.prod(b[p].shape) for p in extra), 170624)
    self.assertEqual(sum(math.prod(v.shape) for v in b.values()), 2775361536)
    for path in a:
      self.assertEqual(a[path].shape, b[path].shape)
    decay = flatten_dict(train.get_wd_tree(new, args[0].params))
    self.assertTrue(all(decay[path] == 0 for path in extra))
    print('H_MEMORY_MLP_BIAS_BUDGET_OK 170624 .02603515625W_Q', flush=True)

  def test_zero_init_preserves_shared_weights_and_logits_then_biases_learn(self):
    cfg_old, cfg_new = self.tiny(H, length=4), self.tiny(BIAS, length=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg_new), cfg_new.mesh_axes)
    old, new = Transformer(cfg_old, mesh, quant=None), Transformer(cfg_new, mesh, quant=None)
    tok = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    pos, seg = jnp.arange(4)[None], jnp.ones_like(tok)
    with mesh, nn.partitioning.axis_rules(cfg_new.logical_axis_rules):
      pa, pb = [nn.unbox(model.init(jax.random.key(71), tok, pos, seg, tok, enable_dropout=False))
                for model in (old, new)]
      a, b = flatten_dict(pa), flatten_dict(pb)
      extra = set(b)-set(a)
      self.assertEqual(len(extra), 2)
      for path in a:
        np.testing.assert_array_equal(a[path], b[path], err_msg=str(path))
      for path in extra:
        np.testing.assert_array_equal(b[path], jnp.zeros_like(b[path]))
      def call(model, params):
        return model.apply(params, tok, pos, seg, tok, enable_dropout=False)[0]
      ya, yb = call(old, pa), call(new, pb)
      np.testing.assert_array_equal(ya, yb)
      target = jax.random.normal(jax.random.key(72), yb.shape)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(call(new, p)*target)))(pb)
      self.assertTrue(np.isfinite(float(loss)))
      flat_grad = flatten_dict(grad)
      for path in extra:
        self.assertTrue(np.isfinite(np.asarray(flat_grad[path])).all())
        self.assertGreater(float(jnp.linalg.norm(flat_grad[path])), 0, str(path))
        changed = dict(b)
        changed[path] = jnp.full_like(b[path], .03)
        self.assertGreater(float(jnp.max(jnp.abs(call(new, unflatten_dict(changed))-yb))), 0, str(path))
    print('H_MEMORY_MLP_BIAS_ZERO_INIT_PARITY_AND_LIVE_GRAD_OK', flush=True)

  def test_native_pallas_forward_and_all_gradients(self):
    self.check_model(BIAS)


if __name__ == '__main__':
  unittest.main()

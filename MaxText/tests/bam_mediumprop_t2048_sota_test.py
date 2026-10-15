"""T2048 transfer keeps the verified shared R256 parameter tree and token budget."""
import unittest

import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import exp
import train_compile
from bam_projection_sharing_test import ProjectionSharingTest


NEW = 'BamMediumPropMShiftReadG050ReadAttentionSharedR256T2048'
PARENT = 'BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile'


class T2048SotaTest(ProjectionSharingTest):
  def test_transfer_budget_and_runtime_settings(self):
    recipe = getattr(exp, NEW)
    self.assertEqual(recipe.max_target_length * recipe.per_device_batch_size * 8, 524288)
    self.assertEqual(recipe.DATASET_VARIANT, 'legacy2048')
    self.assertEqual(recipe.mlp_dim_by_block, [3887, 3762, 3887])
    self.assertEqual(recipe.steps, recipe.learning_rate_schedule_steps)
    self.assertEqual(recipe.steps, 13500)
    self.assertFalse(recipe.qk_norm)
    self.assertTrue(recipe.scan_layers)
    self.assertTrue(recipe.bam_share_read_attention_write_down)
    self.assertFalse(recipe.bam_share_mlp_write_down)
    self.assertEqual(recipe.bam_m_shift_mode, 'read')
    self.assertEqual(recipe.bam_m_shift_init, .5)
    self.assertEqual(recipe.checkpoint_period, 200)
    self.assertEqual(recipe.tensorboard_gcs_rotate_period, 500)
    self.assertEqual(recipe.max_to_keep, 2)
    self.assertEqual(recipe.keep_period, 0)
    self.assertTrue(recipe.force_final_checkpoint)
    trees = {}
    for name in (PARENT, NEW):
      config = self.config(name)
      args, *_ = train_compile.get_shaped_inputs(self.mesh(config), config)
      flat = flatten_dict(nn.unbox(args[0].params))
      trees[name] = {path: value.shape for path, value in flat.items()}
      actual = sum(int(np.prod(value.shape)) for value in flat.values())
      self.assertEqual(actual, 432111680)
      if name == NEW:
        self.assertFalse(config.record_training_health_metrics)
        self.assertFalse(config.record_internal_nn_metrics)
        self.assertTrue(all(not value for key, value in config.get_keys().items()
                            if key.startswith('bam_record_')))
      print('T2048_SOTA_BUDGET_OK', name, actual, flush=True)
    self.assertEqual(trees[PARENT], trees[NEW])


if __name__ == '__main__':
  unittest.main()

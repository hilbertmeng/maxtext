"""Exercise actual Orbax deletion and reopen behavior with the training factory."""
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import orbax.checkpoint as ocp

import checkpointing
import exp


class CheckpointRetentionTest(unittest.TestCase):
  def manager(self, directory, milestones, asynchronous):
    return checkpointing.create_orbax_checkpoint_manager(
        directory, True, asynchronous, 500,
        config=SimpleNamespace(max_to_keep=2, keep_period=0,
                               checkpoint_keep_steps=milestones))

  def save(self, manager, step):
    self.assertTrue(manager.save(step, args=ocp.args.Composite(
        items=ocp.args.PyTreeSave({'weight': np.array([step], dtype=np.int32)}))))
    manager.wait_until_finished()

  def test_milestones_survive_pruning_and_manager_reopen(self):
    for asynchronous in (False, True):
      with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as directory:
        milestones = exp.MuddPythia3BSerialMLPKeepActivations.checkpoint_keep_steps
        with self.manager(directory, milestones, asynchronous) as manager:
          for step in (2500, 3000, 3500, 4000, 12500, 13000, 13500, 14000):
            self.save(manager, step)
          self.assertEqual(sorted(manager.all_steps()), [3000, 13000, 13500, 14000])
        with self.manager(directory, milestones, asynchronous) as manager:
          for step in (14500, 15000, 143000):
            self.save(manager, step)
          self.assertEqual(sorted(manager.all_steps()), [3000, 13000, 15000, 143000])
          restored = manager.restore(3000)
          np.testing.assert_array_equal(restored['items']['weight'], [3000])

  def test_explicit_step_is_saved_even_off_the_periodic_grid(self):
    with tempfile.TemporaryDirectory() as directory, self.manager(directory, [3001], False) as manager:
      for step in (3001, 3500, 4000, 4500):
        self.save(manager, step)
      self.assertEqual(sorted(manager.all_steps()), [3001, 4000, 4500])

  def test_empty_milestones_keep_existing_rolling_policy(self):
    with tempfile.TemporaryDirectory() as directory, self.manager(directory, [], False) as manager:
      for step in (500, 1000, 1500):
        self.save(manager, step)
      self.assertEqual(sorted(manager.all_steps()), [1000, 1500])

  def test_config_and_validation(self):
    self.assertEqual(exp.TrainPythia.checkpoint_keep_steps, list(range(3000, 143001, 10000)))
    for steps in ([-1], [True], [3000.5], ['3000']):
      with self.assertRaises(ValueError):
        checkpointing._checkpoint_retention_options(SimpleNamespace(
            max_to_keep=2, keep_period=0, checkpoint_keep_steps=steps))


if __name__ == '__main__':
  unittest.main()

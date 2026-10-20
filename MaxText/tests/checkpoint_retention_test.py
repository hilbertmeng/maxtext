"""Fixed checkpoint milestones survive rolling cleanup and manager restarts."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import orbax.checkpoint as ocp

import checkpointing


class CheckpointRetentionTest(unittest.TestCase):

  def test_milestones_with_rolling_periodic_retention_and_restart(self):
    for use_async in (False, True):
      for keep_period in (0, 8):
        with self.subTest(async_checkpointing=use_async, keep_period=keep_period), tempfile.TemporaryDirectory() as root:
          config = SimpleNamespace(max_to_keep=2, keep_period=keep_period, checkpoint_keep_steps=[3, 7])

          def manager():
            return checkpointing.create_orbax_checkpoint_manager(
                root, True, use_async, 4, dataset_type='synthetic', config=config)

          def save(mngr, steps):
            for step in steps:
              saved = mngr.save(step, args=ocp.args.Composite(
                  items=ocp.args.PyTreeSave({'value': np.asarray(step, np.int32)})))
              if step in config.checkpoint_keep_steps:
                self.assertTrue(saved)  # Milestones3/7 are outside the ordinary four-step cadence.
              mngr.wait_until_finished()

          with manager() as mngr:
            save(mngr, range(1, 13))
            self.assertEqual(sorted(mngr.all_steps()), [3, 7, 8, 12])
          with manager() as mngr:
            save(mngr, range(13, 21))
            expected = [3, 7, 16, 20] if keep_period == 0 else [3, 7, 8, 16, 20]
            self.assertEqual(sorted(mngr.all_steps()), expected)
            restored = mngr.restore(3, args=ocp.args.Composite(items=ocp.args.PyTreeRestore()))
            self.assertEqual(int(restored['items']['value']), 3)
          actual = sorted(int(p.name) for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit())
          self.assertEqual(actual, expected)  # Check files, not just the manager's cached steps.

  def test_unset_or_empty_milestones_preserve_existing_retention(self):
    for keep_period in (0, 4):
      for extra in ({}, {'checkpoint_keep_steps': []}, {'checkpoint_keep_steps': None}):
        config = SimpleNamespace(max_to_keep=2, keep_period=keep_period, **extra)
        self.assertEqual(checkpointing._checkpoint_retention_options(config),
                         {'max_to_keep': 2, 'keep_period': keep_period or None})
    with tempfile.TemporaryDirectory() as root:
      config = SimpleNamespace(max_to_keep=2, keep_period=0)
      with checkpointing.create_orbax_checkpoint_manager(root, True, False, 1, config=config) as mngr:
        for step in range(1, 5):
          mngr.save(step, args=ocp.args.Composite(items=ocp.args.PyTreeSave({'value': np.asarray(step)})))
        self.assertEqual(sorted(mngr.all_steps()), [3, 4])

  def test_invalid_milestones_are_rejected(self):
    for steps in ([True], [-1], [1.5], ['3']):
      with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, 'nonnegative integer'):
        checkpointing._checkpoint_retention_options(
            SimpleNamespace(max_to_keep=2, keep_period=0, checkpoint_keep_steps=steps))


if __name__ == '__main__':
  unittest.main()

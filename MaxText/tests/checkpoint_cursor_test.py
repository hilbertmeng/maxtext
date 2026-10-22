"""Pile cursor writes cannot pre-create an asynchronously created checkpoint directory."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint._src.path import atomicity

import checkpointing


class CheckpointCursorTest(unittest.TestCase):
  def test_step_directory_is_ready_before_external_cursor_write(self):
    entered = threading.Event()
    release = threading.Event()
    original_create = atomicity.CommitFileTemporaryPath.create
    original_manager = checkpointing.CheckpointManager

    async def delayed_create(path):
      if path.get_final().name == '1':
        entered.set()
        if not await asyncio.to_thread(release.wait, 10):
          raise TimeoutError('test checkpoint directory was never released')
      return await original_create(path)

    def gcs_style_manager(*args, **kwargs):
      # GCS commits in the final directory, unlike local rename-based saves.
      kwargs['options'].temporary_path_class = atomicity.CommitFileTemporaryPath
      return original_manager(*args, **kwargs)

    cfg = SimpleNamespace(max_to_keep=2, keep_period=0, checkpoint_keep_steps=[])
    with tempfile.TemporaryDirectory() as root, \
         mock.patch.object(atomicity.CommitFileTemporaryPath, 'create', delayed_create), \
         mock.patch.object(checkpointing, 'CheckpointManager', gcs_style_manager):
      with checkpointing.create_orbax_checkpoint_manager(
          root, True, True, 1, dataset_type='pile', config=cfg) as manager, \
          ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(manager.save, 1, args=ocp.args.Composite(
            items=ocp.args.PyTreeSave({'value': np.asarray(7, np.int32)})))
        try:
          self.assertTrue(entered.wait(10), 'checkpoint directory creation never started')
          # save must wait for directory creation, while array saving stays asynchronous.
          with self.assertRaises(TimeoutError):
            future.result(timeout=.25)
        finally:
          release.set()
        self.assertTrue(future.result(timeout=20))
        step_dir = Path(root)/'1'
        self.assertTrue(step_dir.is_dir())
        cursor = {'checkpoint_step': 1, 'file_in_data': 3, 'step_in_file': 7}
        (step_dir/'skip_file_and_step.json').write_text(json.dumps(cursor))
        manager.wait_until_finished()
        self.assertTrue((step_dir/'commit_success.txt').exists())
        self.assertEqual(json.loads((step_dir/'skip_file_and_step.json').read_text()), cursor)
        restored = manager.restore(1, args=ocp.args.Composite(items=ocp.args.PyTreeRestore()))
        self.assertEqual(int(restored['items']['value']), 7)


if __name__ == '__main__':
  unittest.main()

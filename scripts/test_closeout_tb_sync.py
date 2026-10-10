import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import closeout_runs_local as closeout
import sync_completed_tensorboards as sync


class CloseoutSyncTest(unittest.TestCase):
  def test_download_is_hidden_until_complete(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "watched"
      root.mkdir()
      def download(*args, **kwargs):
        staging = Path(args[-1])
        self.assertFalse(staging.is_relative_to(root))
        self.assertFalse((root / "Run" / "events.out.tfevents.1").exists())
        (staging / "events.out.tfevents.1").write_bytes(b"complete events")
      with patch.object(sync, "LOCAL_ROOT", root), \
           patch.object(sync, "sources", return_value=["gs://bucket/events/Run"]), \
           patch.object(sync, "run", side_effect=download):
        self.assertTrue(sync.sync_run("Run"))
      self.assertEqual((root / "Run" / "events.out.tfevents.1").read_bytes(), b"complete events")
      self.assertFalse(list(root.rglob("*.gstmp")))

  def test_explicit_batch_refreshes_once_after_success(self):
    with patch("sys.argv", ["sync", "Run1", "Run2"]), \
         patch.object(sync, "sync_run", side_effect=[True, True]), \
         patch.object(sync, "refresh_tensorboard") as refresh:
      self.assertEqual(sync.main(), 0)
      refresh.assert_called_once_with()

  def test_failed_batch_does_not_refresh(self):
    with patch("sys.argv", ["sync", "Run1"]), \
         patch.object(sync, "sync_run", return_value=False), \
         patch.object(sync, "refresh_tensorboard") as refresh:
      self.assertEqual(sync.main(), 1)
      refresh.assert_not_called()

  def test_retry_while_event_changes(self):
    with tempfile.TemporaryDirectory() as directory, patch.object(sync, "LOCAL_ROOT", Path(directory)), \
         patch.object(sync, "sources", return_value=["gs://bucket/events/Run"]), \
         patch.object(sync, "run", side_effect=[subprocess.CalledProcessError(1, "rsync"), None]) as command, \
         patch.object(sync.time, "sleep"):
      self.assertTrue(sync.sync_run("Run"))
      self.assertEqual(command.call_count, 2)

  def test_failure_is_bounded(self):
    with tempfile.TemporaryDirectory() as directory, patch.object(sync, "LOCAL_ROOT", Path(directory)), \
         patch.object(sync, "sources", side_effect=RuntimeError("unavailable")) as discover, \
         patch.object(sync.time, "sleep"):
      self.assertFalse(sync.sync_run("Run"))
      self.assertEqual(discover.call_count, 3)

  def test_discovers_region_without_assuming_checkpoint_location(self):
    target = "gs://newproject-1-llm_projects_europe-west4/log/summaries/train/Run"
    def response(*args, **kwargs):
      if args[0] == "ssh":
        return subprocess.CompletedProcess(args, 0, '{"base_output_directory":"gs://elsewhere/log/"}', "")
      if args[-1] == target + "/events.out.tfevents.*":
        return subprocess.CompletedProcess(args, 0, target + "/events.out.tfevents.1", "")
      return subprocess.CompletedProcess(args, 1, "", "matched no objects")
    with patch.object(sync, "run", side_effect=response):
      self.assertEqual(sync.sources("Run"), [target])

  def test_dry_run_has_no_sync(self):
    with patch("sys.argv", ["closeout", "Run", "--dry-run"]), \
         patch.object(closeout.subprocess, "Popen") as spawn, \
         patch.object(closeout.subprocess, "call", return_value=0) as remote:
      self.assertEqual(closeout.main(), 0)
      spawn.assert_not_called()
      self.assertIn("--dry-run", remote.call_args.args[0][-1])

  def test_sync_detached_and_not_waited(self):
    with tempfile.TemporaryDirectory() as directory, patch.object(closeout.Path, "home", return_value=Path(directory)), \
         patch("sys.argv", ["closeout", "Run", "--reason", "Run=test"]), \
         patch.object(closeout.subprocess, "Popen") as spawn, \
         patch.object(closeout.subprocess, "call", return_value=0):
      self.assertEqual(closeout.main(), 0)
      self.assertTrue(spawn.call_args.kwargs["start_new_session"])
      spawn.return_value.wait.assert_not_called()


if __name__ == "__main__":
  unittest.main()

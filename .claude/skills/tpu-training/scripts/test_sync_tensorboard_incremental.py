"""CPU-only tests: python -m unittest discover -s <this directory> -p test_sync_tensorboard_incremental.py."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import sync_tensorboard_incremental as sync


class TransferRetryTest(unittest.TestCase):
  def test_partial_download_is_truncated_before_retry(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"
      local.write_bytes(b"old")
      attempts = []

      def transfer(command, **kwargs):
        attempts.append(command)
        kwargs["stdout"].write(b"old" if len(attempts) == 1 else b"oldnew")
        kwargs["stdout"].flush()
        if len(attempts) == 1:
          raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

      with mock.patch.object(sync.subprocess, "run", side_effect=transfer), mock.patch.object(sync.time, "sleep"):
        result = sync._sync_one("gsutil", "gs://bucket/events", 6, local)
      self.assertTrue(result.startswith("append"))
      self.assertEqual(local.read_bytes(), b"oldnew")
      self.assertEqual(len(attempts), 2)
      self.assertEqual(list(Path(directory).iterdir()), [local])

  def test_exhausted_retries_preserve_existing_event(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"
      local.write_bytes(b"old")

      def transfer(command, **kwargs):
        kwargs["stdout"].write(b"partial")
        raise subprocess.CalledProcessError(1, command)

      with mock.patch.object(sync.subprocess, "run", side_effect=transfer) as run, mock.patch.object(sync.time, "sleep"):
        with self.assertRaises(subprocess.CalledProcessError):
          sync._sync_one("gsutil", "gs://bucket/events", 6, local)
      self.assertEqual(run.call_count, 3)
      self.assertEqual(local.read_bytes(), b"old")
      self.assertEqual(list(Path(directory).iterdir()), [local])

  def test_listing_retries_with_captured_stdout(self):
    error = subprocess.CalledProcessError(1, ["gsutil", "ls"])
    ok = subprocess.CompletedProcess([], 0, stdout="6 2026-09-06 gs://bucket/events\n")
    with mock.patch.object(sync.subprocess, "run", side_effect=[error, ok]), mock.patch.object(sync.time, "sleep"):
      self.assertEqual(sync._list_events("gsutil", "gs://bucket"), [(6, "gs://bucket/events")])


if __name__ == "__main__":
  unittest.main()

"""CPU-only tests: python -m unittest discover -s <this directory> -p test_sync_tensorboard_incremental.py."""

from pathlib import Path
import json
import subprocess
import tempfile
from types import SimpleNamespace
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


@unittest.skipIf(sync.google_crc32c is None, "optional CRC32C module is unavailable")
class SnapshotCacheTest(unittest.TestCase):
  def snapshot(self, local, generation="1"):
    return dict(uri="gs://bucket/Run/events", size=local.stat().st_size,
                generation=generation, crc32c=sync._file_crc32c(local))

  def test_full_metadata_listing_uses_one_command_and_keeps_versions(self):
    listing = ("gs://bucket/Run/events:\n    Content-Length: 3\n"
               "    Hash (crc32c): abc=\n    Generation: 123\n"
               "    Metageneration: 1\n")
    with mock.patch.object(sync, "_run", return_value=SimpleNamespace(stdout=listing)) as run:
      result = sync._list_snapshots("gsutil", "gs://bucket/Run")
    self.assertEqual(run.call_args.args[0], ["gsutil", "ls", "-L", "gs://bucket/Run/events.out.tfevents.*"])
    self.assertEqual(result, [dict(uri="gs://bucket/Run/events", size=3,
                                  generation="123", crc32c="abc=")])

  def test_bootstrap_existing_files_without_downloading_and_persist_cache(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "Run" / "events"
      local.parent.mkdir(); local.write_bytes(b"old")
      snapshot = self.snapshot(local)
      args = SimpleNamespace(gcs_root="gs://bucket", local_root=Path(directory), gsutil="gsutil")
      with mock.patch.object(sync, "_list_snapshots", return_value=[snapshot]), mock.patch.object(sync, "_sync_one") as transfer:
        first = sync._sync_run(args, "Run")
        with mock.patch.object(sync, "_file_crc32c", side_effect=AssertionError("unchanged file must not be reread")):
          second = sync._sync_run(args, "Run")
      transfer.assert_not_called()
      self.assertIn("transferred=0", first[0]); self.assertIn("unchanged", second[0])
      manifest = json.loads((local.parent / ".sync_versions.json").read_text())
      self.assertEqual(manifest["objects"][snapshot["uri"]]["generation"], "1")

  def test_generation_change_with_identical_crc_needs_no_payload_read(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"; local.write_bytes(b"old")
      snapshot = self.snapshot(local)
      _, previous = sync._sync_snapshot("gsutil", snapshot, local, {})
      changed = dict(snapshot, generation="2")
      with mock.patch.object(sync, "_file_crc32c", side_effect=AssertionError("no local reread")), mock.patch.object(sync, "_sync_one") as transfer:
        outcome, verified = sync._sync_snapshot("gsutil", changed, local, previous)
      transfer.assert_not_called()
      self.assertIn("transferred=0", outcome); self.assertEqual(verified["generation"], "2")

  def test_same_size_remote_rewrite_repairs_entire_file(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"; local.write_bytes(b"old")
      _, previous = sync._sync_snapshot("gsutil", self.snapshot(local), local, {})
      remote = Path(directory) / "remote"; remote.write_bytes(b"new")
      snapshot = dict(self.snapshot(remote, "2"), uri="gs://bucket/Run/events")
      def copy(_gsutil, _uri, dest):
        dest.write_bytes(remote.read_bytes()); return 3
      with mock.patch.object(sync, "_full_copy", side_effect=copy) as full, mock.patch.object(sync, "_sync_one") as tail:
        _, verified = sync._sync_snapshot("gsutil", snapshot, local, previous)
      full.assert_called_once(); tail.assert_not_called()
      self.assertEqual(local.read_bytes(), b"new"); self.assertEqual(verified["generation"], "2")

  def test_same_remote_version_does_not_hide_local_corruption(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"; local.write_bytes(b"old")
      snapshot = self.snapshot(local)
      _, previous = sync._sync_snapshot("gsutil", snapshot, local, {})
      local.write_bytes(b"bad")
      # Do not rely on the filesystem's timestamp resolution in this test.
      previous["local_stamp"]["mtime_ns"] = -1
      def copy(_gsutil, _uri, dest):
        dest.write_bytes(b"old"); return 3
      with mock.patch.object(sync, "_full_copy", side_effect=copy) as full:
        _, verified = sync._sync_snapshot("gsutil", snapshot, local, previous)
      full.assert_called_once(); self.assertEqual(local.read_bytes(), b"old")
      self.assertEqual(verified["crc32c"], snapshot["crc32c"])

  def test_growing_snapshot_caches_only_crc_verified_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"; local.write_bytes(b"old")
      remote = Path(directory) / "remote"; remote.write_bytes(b"oldnew")
      snapshot = self.snapshot(remote, "2")
      def transfer(command, **kwargs):
        kwargs["stdout"].write(b"oldnew"); kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)
      with mock.patch.object(sync.subprocess, "run", side_effect=transfer):
        outcome, verified = sync._sync_snapshot("gsutil", snapshot, local, {})
      self.assertTrue(outcome.startswith("append")); self.assertEqual(local.read_bytes(), b"oldnew")
      self.assertEqual(verified["size"], 6)

  def test_rewritten_prefix_outside_tail_is_repaired(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"; local.write_bytes(b"old")
      remote = Path(directory) / "remote"; remote.write_bytes(b"goodnew")
      snapshot = self.snapshot(remote, "2")
      def tail(*_args):
        local.write_bytes(b"oldnew!"); return "append old=3 new=7 transferred=7"
      def copy(_gsutil, _uri, dest):
        dest.write_bytes(b"goodnew"); return 7
      with mock.patch.object(sync, "_sync_one", side_effect=tail), mock.patch.object(sync, "_full_copy", side_effect=copy) as full:
        _, verified = sync._sync_snapshot("gsutil", snapshot, local, {})
      full.assert_called_once(); self.assertEqual(local.read_bytes(), b"goodnew")
      self.assertEqual(verified["crc32c"], snapshot["crc32c"])

  def test_object_advancing_after_listing_is_not_cached_as_old_generation(self):
    with tempfile.TemporaryDirectory() as directory:
      local = Path(directory) / "events"
      remote = Path(directory) / "remote"; remote.write_bytes(b"old")
      snapshot = self.snapshot(remote, "1")
      def full(*_args):
        local.write_bytes(b"newer"); return "full new=5 transferred=5"
      with mock.patch.object(sync, "_sync_one", side_effect=full):
        _, verified = sync._sync_snapshot("gsutil", snapshot, local, {})
      self.assertIsNone(verified); self.assertEqual(local.read_bytes(), b"newer")


if __name__ == "__main__":
  unittest.main()

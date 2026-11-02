#!/usr/bin/env python3
"""Incrementally mirror TensorBoard event objects from GCS.

One metadata listing supplies object generations and CRC32C hashes. Verified,
unchanged local snapshots require no payload reads; an existing mirror can
bootstrap this cache by hashing local bytes. Growing files retain the original
overlap-plus-tail verification, with atomic full-copy repair on mismatches.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  try:
    import google_crc32c
  except ImportError:
    google_crc32c = None


DEFAULT_GCS_ROOT = (
    "gs://newproject-1-llm_base_models_us-central1/log/summaries/train"
)
DEFAULT_LOCAL_ROOT = Path("/data0/xd/tensorboard_logs")
DEFAULT_GSUTIL = Path.home() / "google-cloud-sdk/bin/gsutil"
OVERLAP_BYTES = 1 << 20


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
  output = kwargs.get("stdout")
  offset = output.tell() if hasattr(output, "seekable") and output.seekable() else None
  for attempt in range(3):
    if offset is not None:
      output.seek(offset)
      output.truncate()
    try:
      return subprocess.run(command, check=True, **kwargs)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
      if attempt == 2:
        raise
      print(f"GCS {command[1]} failed; retry {attempt + 2}/3", file=sys.stderr)
      time.sleep(attempt + 1)


def _list_events(gsutil: str, remote_dir: str) -> list[tuple[int, str]]:
  output = _run(
      [gsutil, "ls", "-l", remote_dir.rstrip("/") + "/events.out.tfevents.*"],
      stdout=subprocess.PIPE,
      text=True,
  ).stdout
  events = []
  for line in output.splitlines():
    match = re.match(r"\s*(\d+)\s+\S+\s+(gs://\S+)\s*$", line)
    if match:
      events.append((int(match.group(1)), match.group(2)))
  if not events:
    raise FileNotFoundError(f"no TensorBoard events under {remote_dir}")
  return events


def _list_snapshots(gsutil: str, remote_dir: str) -> list[dict]:
  """One object listing supplies version/hash metadata without reading payloads."""
  output = _run(
      [gsutil, "ls", "-L", remote_dir.rstrip("/") + "/events.out.tfevents.*"],
      stdout=subprocess.PIPE, text=True,
  ).stdout
  snapshots = []
  current = None
  for line in output.splitlines():
    if line.startswith("gs://") and line.endswith(":"):
      current = {"uri": line[:-1]}
      snapshots.append(current)
    elif current is not None:
      for label, key in (("Content-Length", "size"), ("Generation", "generation"),
                         ("Hash (crc32c)", "crc32c")):
        match = re.match(r"\s*" + re.escape(label) + r":\s+(\S+)\s*$", line)
        if match:
          current[key] = int(match.group(1)) if key == "size" else match.group(1)
  if not snapshots:
    raise FileNotFoundError(f"no TensorBoard events under {remote_dir}")
  for snapshot in snapshots:
    if not all(k in snapshot for k in ("size", "generation", "crc32c")):
      raise ValueError(f"incomplete GCS object metadata: {snapshot}")
  return snapshots


def _local_stamp(local: Path):
  if not local.exists():
    return None
  stat = local.stat()
  return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _file_crc32c(local: Path):
  if google_crc32c is None:
    return None
  checksum = google_crc32c.Checksum()
  with local.open("rb") as source:
    for chunk in iter(lambda: source.read(1 << 20), b""):
      checksum.update(chunk)
  return base64.b64encode(checksum.digest()).decode("ascii")


def _sync_snapshot(gsutil: str, snapshot: dict, local: Path, previous: dict):
  """Skip only verified bytes; generation changes with identical CRC are safe too."""
  size, remote = snapshot["size"], snapshot["uri"]
  stamp = _local_stamp(local)
  if (stamp is not None and stamp["size"] == size and
      previous.get("local_stamp") == stamp and previous.get("size") == size and
      previous.get("crc32c") == snapshot["crc32c"]):
    return f"unchanged generation={snapshot['generation']} transferred=0", dict(snapshot, local_stamp=stamp)

  crc = _file_crc32c(local) if stamp is not None and stamp["size"] == size else None
  if crc is not None and crc == snapshot["crc32c"]:
    return f"verified-local generation={snapshot['generation']} transferred=0", dict(snapshot, local_stamp=stamp)

  if crc is not None:
    # A same-size rewrite or local corruption can lie outside the old tail overlap.
    copied = _full_copy(gsutil, remote, local)
    outcome = f"full crc-mismatch new={copied} transferred={copied}"
  else:
    outcome = _sync_one(gsutil, remote, size, local)
  stamp = _local_stamp(local)
  crc = _file_crc32c(local)
  if crc is not None and stamp["size"] == size and crc == snapshot["crc32c"]:
    return outcome, dict(snapshot, local_stamp=stamp)
  if outcome.startswith("append") and crc is not None:
    # Growth may also rewrite earlier bytes; repair a prefix that passed the tail check.
    copied = _full_copy(gsutil, remote, local)
    outcome += f"; full-after-crc-mismatch transferred={copied}"
    stamp = _local_stamp(local)
    if stamp["size"] == size and _file_crc32c(local) == snapshot["crc32c"]:
      return outcome, dict(snapshot, local_stamp=stamp)
  # A growing object may advance after the listing. Keep the downloaded snapshot,
  # but never cache a generation whose full checksum was not actually verified.
  return outcome, None


def _full_copy(gsutil: str, remote: str, local: Path) -> int:
  local.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=local.parent, delete=False) as handle:
    temporary = Path(handle.name)
  try:
    _run([gsutil, "cp", remote, str(temporary)])
    size = temporary.stat().st_size
    os.replace(temporary, local)
    return size
  finally:
    temporary.unlink(missing_ok=True)


def _sync_one(gsutil: str, remote: str, remote_size: int, local: Path) -> str:
  if not local.exists():
    copied = _full_copy(gsutil, remote, local)
    return f"full new={copied} transferred={copied}"

  local_size = local.stat().st_size
  if remote_size < local_size:
    copied = _full_copy(gsutil, remote, local)
    return f"full old={local_size} new={copied} transferred={copied}"

  overlap = min(local_size, OVERLAP_BYTES)
  start = local_size - overlap
  end = remote_size - 1
  local.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=local.parent, delete=False) as handle:
    segment = Path(handle.name)
  try:
    if remote_size:
      with segment.open("wb") as output:
        _run(
            [gsutil, "cat", "-r", f"{start}-{end}", remote],
            stdout=output,
        )
    expected = remote_size - start
    if segment.stat().st_size != expected:
      copied = _full_copy(gsutil, remote, local)
      return f"full old={local_size} new={copied} transferred={copied}"

    with local.open("rb") as existing, segment.open("rb") as incoming:
      existing.seek(start)
      if existing.read(overlap) != incoming.read(overlap):
        copied = _full_copy(gsutil, remote, local)
        return f"full old={local_size} new={copied} transferred={copied}"
      if remote_size == local_size:
        return f"unchanged size={local_size} checked={overlap}"
      with local.open("ab") as output:
        shutil.copyfileobj(incoming, output)
        output.flush()
        os.fsync(output.fileno())
    return (
        f"append old={local_size} new={remote_size} "
        f"transferred={expected}"
    )
  finally:
    segment.unlink(missing_ok=True)


def _sync_run(args, run: str) -> list[str]:
  remote_dir = f"{args.gcs_root.rstrip('/')}/{run}"
  local_dir = args.local_root / run
  local_dir.mkdir(parents=True, exist_ok=True)
  with (local_dir / ".sync.lock").open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest_path = local_dir / ".sync_versions.json"
    try:
      manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, ValueError):
      manifest = {}
    if not isinstance(manifest, dict):
      manifest = {}
    records = manifest.get("objects", {}) if manifest.get("schema_version") == 1 else {}
    if not isinstance(records, dict):
      records = {}
    outcomes = []
    for snapshot in _list_snapshots(args.gsutil, remote_dir):
      remote = snapshot["uri"]
      local = local_dir / remote.rsplit("/", 1)[-1]
      outcome, verified = _sync_snapshot(args.gsutil, snapshot, local, records.get(remote, {}))
      if verified is None:
        records.pop(remote, None)
      else:
        records[remote] = verified
      temporary = manifest_path.with_suffix(".tmp")
      temporary.write_text(json.dumps({"schema_version": 1, "objects": records}, indent=2) + "\n")
      os.replace(temporary, manifest_path)
      outcomes.append(f"RUN={run} file={local.name} {outcome}")
    return outcomes


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("run", nargs="+")
  parser.add_argument("--gcs-root", default=DEFAULT_GCS_ROOT)
  parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
  parser.add_argument("--gsutil", default=str(DEFAULT_GSUTIL))
  parser.add_argument("--jobs", type=int, default=4)
  args = parser.parse_args()

  with ThreadPoolExecutor(max_workers=min(args.jobs, len(args.run))) as executor:
    for outcomes in executor.map(lambda run: _sync_run(args, run), args.run):
      print(*outcomes, sep="\n")


if __name__ == "__main__":
  main()

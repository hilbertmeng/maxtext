#!/usr/bin/env python3
"""Incrementally mirror TensorBoard event objects from GCS.

GCS objects are immutable snapshots, so ``gsutil rsync`` recopies a growing
event file in full.  This tool fetches one overlap-plus-tail range, verifies
that the remote snapshot extends the local prefix, and appends only new bytes.
It falls back to an atomic full copy whenever the prefix check fails.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


DEFAULT_GCS_ROOT = (
    "gs://newproject-1-llm_base_models_us-central1/log/summaries/train"
)
DEFAULT_LOCAL_ROOT = Path("/data0/xd/tensorboard_logs")
DEFAULT_GSUTIL = Path.home() / "google-cloud-sdk/bin/gsutil"
OVERLAP_BYTES = 1 << 20


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
  return subprocess.run(command, check=True, **kwargs)


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
  outcomes = []
  for remote_size, remote in _list_events(args.gsutil, remote_dir):
    local = local_dir / remote.rsplit("/", 1)[-1]
    outcome = _sync_one(args.gsutil, remote, remote_size, local)
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

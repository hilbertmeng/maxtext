#!/usr/bin/env python3
"""Idempotently sync TensorBoard runs marked complete by auto-train."""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


GSUTIL = "/home/xd/google-cloud-sdk/bin/gsutil"
ROOT = "gs://newproject-1-llm_base_models_us-central1/log"
LOCAL_ROOT = Path("/home/xd/tensorboard_logs")
STATE_DIR = Path("/home/xd/.local/state/maxtext-tensorboard-sync")
MARKER_RE = re.compile(r"^\s*\d+\s+(\S+)\s+(gs://\S+)$")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(args, text=True, capture_output=True, check=check)


def main() -> int:
  STATE_DIR.mkdir(parents=True, exist_ok=True)
  with (STATE_DIR / "lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state_path = STATE_DIR / "synced.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    listed = run(GSUTIL, "ls", "-l", f"{ROOT}/tensorboard_complete/*", check=False)
    if listed.returncode and "matched no objects" not in listed.stderr:
      print(listed.stderr, file=sys.stderr, end="")
      return listed.returncode

    changed = False
    for line in listed.stdout.splitlines():
      match = MARKER_RE.match(line)
      if not match:
        continue
      marker_time, marker_uri = match.groups()
      run_name = marker_uri.rsplit("/", 1)[-1]
      if state.get(run_name) == marker_time:
        continue
      source = f"{ROOT}/summaries/train/{run_name}/"
      destination = LOCAL_ROOT / run_name
      destination.mkdir(parents=True, exist_ok=True)
      print(f"sync {run_name}", flush=True)
      synced = subprocess.run(
          [GSUTIL, "-m", "rsync", "-c", "-r", source, f"{destination}/"],
          check=False)
      if synced.returncode:
        print(f"sync failed: {run_name}", file=sys.stderr)
        continue
      state[run_name] = marker_time
      changed = True

    if changed:
      temporary = state_path.with_suffix(".tmp")
      temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
      os.replace(temporary, state_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

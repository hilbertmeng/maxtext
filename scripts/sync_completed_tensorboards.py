#!/usr/bin/env python3
"""Idempotently sync TensorBoard runs marked complete by auto-train."""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import argparse
import shlex
import time
from concurrent.futures import ThreadPoolExecutor


GSUTIL = "/home/xd/google-cloud-sdk/bin/gsutil"
ROOT = "gs://newproject-1-llm_base_models_us-central1/log"
LOCAL_ROOT = Path("/data0/xd/tensorboard_logs")
STATE_DIR = Path("/home/xd/.local/state/maxtext-tensorboard-sync")
MARKER_RE = re.compile(r"^\s*\d+\s+(\S+)\s+(gs://\S+)$")


def sources(run_name):
  """Discover actual event locations, not checkpoint-region assumptions."""
  if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
    raise ValueError(f"invalid RUN: {run_name}")
  command = "cat " + shlex.quote(
      f"/home/lishengping/xd/projects/run_registry/{run_name}.json")
  meta = json.loads(run("ssh", "-S", "/tmp/ssh-tpu-ag-xd.sock", "tpu-ag", command).stdout)
  candidates = []
  if meta.get("tensorboard_dir"):
    candidates.append(meta["tensorboard_dir"].rstrip("/"))
  roots = [ROOT, "gs://newproject-1-llm_projects_europe-west4/log",
           "gs://newproject-1-llm_projects_us-east5/log"]
  for key in ("base_output_directory", "previous_base_output_directory"):
    if meta.get(key):
      roots.append(meta[key].rstrip("/"))
  for root in dict.fromkeys(roots):
    candidates.extend([f"{root}/summaries/train/{run_name}",
                       f"{root}/{run_name}/tensorboard"])
  found = []
  for source in dict.fromkeys(candidates):
    result = run(GSUTIL, "ls", f"{source}/events.out.tfevents.*", check=False)
    if result.returncode == 0 and result.stdout.strip():
      found.append(source)
    elif result.returncode and not any(s in result.stderr for s in (
        "matched no objects", "matched no URLs")):
      raise RuntimeError(result.stderr)
  if not found:
    raise RuntimeError(f"no TB events found for {run_name}")
  return found


def sync_run(run_name):
  destination = LOCAL_ROOT / run_name
  destination.mkdir(parents=True, exist_ok=True)
  with (destination / ".sync.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for attempt in range(3):
      try:
        for source in sources(run_name):
          print(f"sync {run_name}: {source} -> {destination}", flush=True)
          run(GSUTIL, "-m", "rsync", "-c", "-r", source + "/", str(destination) + "/")
        print(f"SYNC_OK {run_name}", flush=True)
        return True
      except (subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        print(f"SYNC_RETRY {run_name} attempt={attempt + 1}/3: {exc}", file=sys.stderr, flush=True)
        if attempt < 2:
          time.sleep(5 * (attempt + 1))
    print(f"SYNC_FAILED {run_name}", file=sys.stderr, flush=True)
    return False


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(args, text=True, capture_output=True, check=check)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", nargs="*")
  args = parser.parse_args()
  if args.runs:
    with ThreadPoolExecutor(max_workers=min(4, len(args.runs))) as pool:
      return 0 if all(list(pool.map(sync_run, args.runs))) else 1
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
      if not sync_run(run_name):
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

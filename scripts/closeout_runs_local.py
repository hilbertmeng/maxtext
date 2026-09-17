#!/usr/bin/env python3
"""Remote closeout plus independent local TB copy; forwards closeout options."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", nargs="+")
  args, options = parser.parse_known_args()
  remote = ["python3", "/home/lishengping/xd/projects/closeout_runs.py", *args.runs, *options]
  if "--dry-run" not in options:
    logs = Path.home() / ".local/state/maxtext-tensorboard-sync"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = logs / f"closeout-{stamp}.log"
    with log.open("w") as output:
      child = subprocess.Popen(
          [sys.executable, str(Path(__file__).with_name("sync_completed_tensorboards.py")), *args.runs],
          stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
          start_new_session=True)
    print(f"TB background pid={child.pid} log={log}", flush=True)
  return subprocess.call(["ssh", "-S", "/tmp/ssh-tpu-ag-xd.sock", "tpu-ag", shlex.join(remote)])


if __name__ == "__main__":
  raise SystemExit(main())

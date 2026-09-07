#!/usr/bin/env python3
"""Independent scan/non-scan AOT lanes; adopt live preparers and release slots at artifact readiness."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('commit')
  parser.add_argument('--per-lane', type=int, default=2)
  args = parser.parse_args()
  if args.per_lane < 1:
    parser.error('--per-lane must be positive')
  root = Path(os.environ.get('TPU_AG_ROOT', '/home/lishengping/xd/projects'))
  sys.path.insert(0, str(root))
  import prepare_train_aot as aot
  commit = aot.resolve_remote_commit(args.commit)
  aot.verify_remote_commit(commit)
  output = f'gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/local-fetch/{commit}'
  logs = root / 'logs' / f'local-fetch-aot-{commit}'
  logs.mkdir(parents=True, exist_ok=True)
  jobs = {}
  for layout in ('Scan', 'NonScan'):
    for variant in ('Control', 'C8', 'C8LocalV', 'Full', 'FullLocalV', 'C8SharedRead'):
      exp = f'BamLlama2MediumV2C256LocalFetch{variant}{layout}'
      spec = argparse.Namespace(exp=exp, commit=commit, target_topology='v5p-16', steps=13500,
                                output=f'{output}/{exp}.pickle', installer='install_xd_maxtext_jax081.sh',
                                zones=list(aot.DEFAULT_ZONES))
      jobs[exp] = (layout, aot.Preparer(spec))
  processes, started, reported = {}, set(), set()
  ready_states = {'artifact_ready', 'ready'}
  failed_states = {'failed', 'source_failed', 'cleanup_failed', 'interrupted'}
  while True:
    states, active = {}, set()
    for exp, (_, preparer) in jobs.items():
      states[exp] = json.loads(preparer.state_path.read_text()).get('status') if preparer.state_path.exists() else None
      if preparer.lock_path.exists():
        with preparer.lock_path.open('r') as handle:
          try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
          except BlockingIOError:
            active.add(exp)
      process = processes.get(exp)
      if process is not None and process.poll() not in (None, 0):
        raise RuntimeError(f'{exp} preparer exited {process.returncode}; inspect {logs / (exp + ".log")}')
      if states[exp] in failed_states:
        raise RuntimeError(f'{exp}: {states[exp]}; inspect its preparer state/log')
      if exp in started and states[exp] not in ready_states:
        active.add(exp)  # Includes the brief subprocess-start / lock-acquisition interval.
    for layout in ('Scan', 'NonScan'):
      lane = [exp for exp, (kind, _) in jobs.items() if kind == layout]
      if all(states[exp] in ready_states for exp in lane):
        if layout not in reported:
          print(f'AOT_GROUP_READY layout={layout} root={output}', flush=True)
          reported.add(layout)
        continue
      slots = args.per_lane - sum(exp in active and states[exp] not in ready_states for exp in lane)
      for exp in lane:
        if slots <= 0:
          break
        if exp in active or states[exp] in ready_states:
          continue
        if states[exp] is not None:
          raise RuntimeError(f'{exp}: stale preparer state {states[exp]} without a live owner')
        with (logs / f'{exp}.log').open('a') as log:
          processes[exp] = subprocess.Popen(
              [str(root / 'prepare_train_aot.py'), exp, commit, 'v5p-16', '13500',
               '--output', f'{output}/{exp}.pickle'], stdout=log, stderr=subprocess.STDOUT,
              start_new_session=True)
        started.add(exp)
        slots -= 1
        print(f'AOT_START layout={layout} exp={exp} pid={processes[exp].pid}', flush=True)
    if len(reported) == 2:
      print(f'LOCAL_FETCH_AOT_MATRIX_READY root={output} cleanup=pending', flush=True)
      break
    time.sleep(15)
  for exp, process in processes.items():
    if process.wait() != 0:
      raise RuntimeError(f'{exp}: cleanup failed; inspect its preparer log')
  print('OWNED_PREPARERS_CLEANED', flush=True)


if __name__ == '__main__':
  main()

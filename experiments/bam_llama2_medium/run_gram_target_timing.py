#!/usr/bin/env python3
"""tpu-ag: sealed full-shape baseline/MulOutput/MulMix AOT timing on one pod.

Keeps the TPU allocated; caller verifies results and performs resource closeout.
Run this installed orchestrator in tmux, never under auto-train.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time


def main():
  p = argparse.ArgumentParser(__doc__)
  p.add_argument('--tpu', required=True)
  p.add_argument('--zone', required=True)
  p.add_argument('--commit', required=True)
  p.add_argument('--size', choices=('Medium', 'XL'), required=True)
  p.add_argument('--artifact-root', required=True)
  p.add_argument('--output', required=True)
  args = p.parse_args()
  assert args.tpu.startswith('xd-') and re.fullmatch('[0-9a-f]{40}', args.commit)
  root = Path('/home/lishengping/xd/projects')
  repo = str(root / 'maxtext')
  output = Path(args.output)
  result = dict(tpu=args.tpu, zone=args.zone, commit=args.commit, size=args.size, arms=[])
  steps = 13500 if args.size == 'Medium' else 50000
  topology = 'v5p-16' if args.size == 'Medium' else 'v5p-32'
  region = args.zone.rsplit('-', 1)[0]
  dataset_roots = {
      'us-east5': 'newproject-1-common_datasets_us-east5',
      'europe-west4': 'newproject-1-common_datasets_europe-west4',
      'us-central1': 'newproject-1-llm_base_models_us-central1/data'}
  dataset = 'gs://' + dataset_roots[region] + '/pythia_pile_idxmaps_tfrecord'

  def save():
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(output)

  def ssh(command, workers='0'):
    process = subprocess.run([
        'gcloud', '--configuration=xd-tpu', 'compute', 'tpus', 'tpu-vm', 'ssh',
        args.tpu, '--project=newproject-1-451205', '--zone=' + args.zone,
        '--worker=' + workers, '--command=' + command], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    if process.returncode:
      raise RuntimeError(process.stdout[-6000:])
    return process.stdout

  # Validate all artifacts before using the target pod.
  for label in ('Base', 'MulOutput', 'MulMix'):
    exp = f'Bam{args.size}IndependentLLFGram{label}'
    artifact = args.artifact_root.rstrip('/') + '/' + exp + '.pickle'
    manifest = json.loads(subprocess.check_output(['gsutil', 'cat', artifact + '.manifest.json']))
    for key, value in dict(commit=args.commit, exp=exp, target_topology=topology, steps=steps).items():
      if manifest[key] != value:
        raise ValueError(f'{label}: AOT {key} mismatch: {manifest[key]} != {value}')
    result['arms'].append(dict(label=label, exp=exp, artifact=artifact, state='planned'))
  save()
  install_log = root / 'logs' / (args.tpu + '-create.log')
  deadline = time.monotonic() + 7200
  while time.monotonic() < deadline:
    log = install_log.read_text() if install_log.exists() else ''
    if f'INSTALL_OK TPU={args.tpu}' in log:
      break
    if 'ERROR:' in log or 'Traceback (most recent call last)' in log:
      raise RuntimeError('standalone installer failed: ' + log[-2000:])
    time.sleep(15)
  else:
    raise TimeoutError('installation observation limit; inspect existing creator')
  ssh('cd ' + shlex.quote(repo) + ' && test "$(git rev-parse HEAD)" = ' + args.commit
      + " && ! pgrep -f '[M]axText/train.py'", 'all')
  for arm in result['arms']:
    arm['compiled'] = '/tmp/gram-' + args.size + '-' + arm['label'] + '.pickle'
    ssh('gsutil -q cp ' + shlex.quote(arm['artifact']) + ' ' + shlex.quote(arm['compiled']), 'all')
  print('INSTALL_AND_AOT_VERIFIED', flush=True)
  for arm in result['arms']:
    run = 'TimingGram_' + args.size + '_' + arm['label'] + '_' + str(int(time.time()))
    log = '/home/lishengping/train_' + run + '.log'
    pattern = '[M]axText/train.py.*run_name=' + run
    arm.update(run=run, log=log, state='launching')
    save()
    smoke = repo + '/.claude/skills/tpu-diagnostics/scripts/run_train_smoke_compiled.sh'
    command = ('cd ' + shlex.quote(repo) + '; nohup env DATASET_PATH=' + shlex.quote(dataset)
               + ' SMOKE_OUTPUT=' + shlex.quote('/tmp/' + run) + ' bash ' + shlex.quote(smoke)
               + ' ' + arm['exp'] + ' ' + run + ' ' + arm['compiled'] + ' ' + str(steps)
               + ' >' + shlex.quote(log) + ' 2>&1 </dev/null &')
    ssh(command, 'all')
    print('LAUNCHED ' + run, flush=True)
    try:
      deadline, first = time.monotonic() + 3600, False
      while time.monotonic() < deadline:
        text = ssh('tail -50 ' + shlex.quote(log) + '; if pgrep -f ' + shlex.quote(pattern)
                   + ' >/dev/null; then echo PROCESS_ALIVE; fi')
        observed = [int(s) for s in re.findall(r'completed step: (\d+),', text)]
        if observed and not first:
          first = True
          print('FIRST_STEP ' + run + ' observed=' + str(max(observed)), flush=True)
        if observed and max(observed) >= 15:
          break
        if 'PROCESS_ALIVE' not in text:
          raise RuntimeError('train process exited: ' + text[-4000:])
        time.sleep(10)
      else:
        raise TimeoutError('step observation limit; inspect exact process before retry')
      ssh('grep -q "Loaded compiled function!" ' + shlex.quote(log), 'all')
      text = ssh('grep "completed step:" ' + shlex.quote(log))
      records = {int(s): float(v) for s, v in re.findall(
          r'completed step: (\d+), steps/s:\s*([\d.]+)', text)}
      rates = [records[s] for s in range(10, 15)]
      arm.update(state='complete', speeds_10_14=rates,
                 mean_steps_per_second=statistics.mean(rates), all_rates=records)
      output.with_name(run + '.log').write_text(text)
      print(json.dumps(arm), flush=True)
    finally:
      ssh('pkill -KILL -f ' + shlex.quote(pattern) + ' || true', 'all')
      ssh('! pgrep -f ' + shlex.quote(pattern), 'all')
      save()
  result['state'] = 'complete'
  save()
  print('TIMING_COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
  main()

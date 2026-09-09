#!/usr/bin/env python3
"""Run on tpu-ag: wait for a verified installer, then the sealed profile matrix."""
import argparse
from pathlib import Path
import os
import subprocess
import time


def main():
  p = argparse.ArgumentParser(__doc__)
  p.add_argument('--tpu', required=True)
  p.add_argument('--zone', required=True)
  p.add_argument('--commit', required=True)
  p.add_argument('--size', choices=['Medium', 'XL'], required=True)
  p.add_argument('--runner', required=True)
  args = p.parse_args()
  root = Path('/home/lishengping/xd/projects')
  log = root / 'logs' / (args.tpu + '-create.log')
  deadline = time.monotonic() + 3600
  while time.monotonic() < deadline:
    text = log.read_text() if log.exists() else ''
    if f'INSTALL_OK TPU={args.tpu}' in text:
      break
    if 'ERROR:' in text or 'Traceback (most recent call last)' in text:
      raise RuntimeError('Installer failed: ' + text[-2000:])
    time.sleep(20)
  else:
    raise TimeoutError('One-hour resource/install observation limit; inspect creator before retry')
  env = dict(os.environ, PROFILE_STEPS='100', PROFILE_TRACE_COUNT='1',
             PROFILE_LOG_ROOT=str(root / 'logs'),
             PROFILE_SMOKE=str(root / 'maxtext/.claude/skills/tpu-diagnostics/scripts/run_train_smoke.sh'))
  classes = [f'Bam{args.size}IndependentLLFGram{s}SixLayer'
             for s in ('Base', 'DotOutput', 'MulOutput', 'DotMix', 'MulMix')]
  subprocess.run(['bash', args.runner, args.tpu, args.zone, args.commit,
                  'Gram' + args.size, *classes], env=env, check=True)


if __name__ == '__main__':
  main()

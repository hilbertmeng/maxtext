#!/usr/bin/env python3
"""Read compact LocalV dual-branch TB series via the shared incremental cache."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('run')
  parser.add_argument('--steps', required=True, help='Comma-separated milestones')
  parser.add_argument('--reader', default='/home/xd/projects/maxtext/.claude/skills/tpu-training/scripts/report_bam_read_health.py')
  parser.add_argument('--tb-root', default='/data0/xd/tensorboard_logs')
  args = parser.parse_args()
  spec = importlib.util.spec_from_file_location('bam_health_reader', args.reader)
  reader = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(reader)
  steps = [int(s) for s in args.steps.split(',')]
  scalars = reader.Scalars(Path(args.tb_root)/args.run, steps)
  tags = sorted(tag for tag in scalars.tags if tag.startswith('bam/local_v_dual/'))
  if not tags:
    raise ValueError('LocalV dual health tags missing; check capture/export rather than reporting zeros')
  output = {tag: [scalars.at(tag, step) for step in steps] for tag in tags}
  if 'learning/raw_grad_norm' in scalars.tags:
    output['learning/raw_grad_norm'] = [scalars.at('learning/raw_grad_norm',step) for step in steps]
  print(json.dumps(dict(run=args.run, steps=steps, series=output), indent=2))


if __name__ == '__main__':
  main()

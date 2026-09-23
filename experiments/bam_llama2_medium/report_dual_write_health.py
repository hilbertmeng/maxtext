#!/usr/bin/env python3
"""Summarize exact-step dual-write TB snapshots using the incremental scalar cache."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run', default='BamMediumAllLocalDualWriteGates')
  parser.add_argument('--steps', required=True, help='Comma-separated target steps; tolerance 25')
  parser.add_argument('--tb-root', type=Path, default=Path('/data0/xd/tensorboard_logs'))
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  source = Path(__file__).resolve().parents[2] / '.agents/skills/tpu-training/scripts/report_bam_read_health.py'
  spec = importlib.util.spec_from_file_location('bam_health_reader', source)
  reader = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = reader
  spec.loader.exec_module(reader)
  targets = sorted({int(s) for s in args.steps.split(',')})
  points, _ = reader._cached_local_points(args.tb_root / args.run, targets)
  raw_points = {tag: data for tag, data in points.items() if tag.startswith('bam/raw_write/')}
  signed_points = {tag: data for tag, data in points.items() if tag.startswith('bam/signed_write/')}
  address_points = {tag: data for tag, data in points.items() if tag.startswith('bam/feedback_address/')}
  erase_points = {tag: data for tag, data in points.items() if tag.startswith('bam/erase/')}
  if erase_points and len(erase_points) != 5304:
    raise ValueError(f'Expected5304 erase tags, found {len(erase_points)}')
  if address_points and len(address_points) != 3264:
    raise ValueError(f'Expected3264 address-mix tags, found {len(address_points)}')
  if signed_points and len(signed_points) != 1968:
    raise ValueError(f'Expected1968 signed-write tags, found {len(signed_points)}')
  prefix = 'bam/dual_write/'
  points = {tag: data for tag, data in points.items() if tag.startswith(prefix)}
  if len(points) != 3768:
    raise ValueError(f'Expected 3768 dual-write tags, found {len(points)}')
  common = set.intersection(*(set(data) for data in points.values()))
  result = {'run': args.run, 'bands': {'early': [1, 2], 'middle': [3, 16], 'late': [17, 22]}, 'snapshots': []}
  for target in targets:
    step = min(common, key=lambda s: (abs(s-target), -s))
    if abs(step-target) > 25:
      raise ValueError(f'No complete snapshot near {target}; nearest {step}')
    values = {tag.removeprefix(prefix): data[step] for tag, data in points.items()}
    if not all(np.isfinite(v) for v in values.values()):
      raise ValueError(f'Nonfinite dual-write health at step {step}')
    summaries = {}
    metrics = ('read_mean', 'main_mean', 'feedback_mean', 'mean_abs_gate_diff',
               'read_main_corr', 'read_feedback_corr', 'main_feedback_corr', 'feedback_norm_share')
    for band, (lo, hi) in result['bands'].items():
      summaries[band] = {}
      for metric in metrics:
        data = [values[f'layer_{layer:03d}/head_{head:02d}/{metric}']
                for layer in range(lo, hi+1) for head in range(16)]
        summaries[band][metric] = dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                                               map(float, np.quantile(data, [0, .25, .5, .75, 1]))))
    raw_norms = {tag: data[step] for tag, data in raw_points.items()}
    signed_values = {tag: data[step] for tag, data in signed_points.items()}
    address_values = {tag: data[step] for tag, data in address_points.items()}
    erase_values = {tag: data[step] for tag, data in erase_points.items()}
    if not all(np.isfinite(v) for v in erase_values.values()):
      raise ValueError(f'Nonfinite erase health at step {step}')
    if not all(np.isfinite(v) for v in address_values.values()):
      raise ValueError(f'Nonfinite address-mix health at step {step}')
    if not all(np.isfinite(v) for v in signed_values.values()):
      raise ValueError(f'Nonfinite signed-write health at step {step}')
    if not all(np.isfinite(v) for v in raw_norms.values()):
      raise ValueError(f'Nonfinite raw-write norm at step {step}')
    result['snapshots'].append({'target': target, 'step': step, 'head_distributions': summaries, 'values': values, 'raw_write_norms': raw_norms, 'signed_write': signed_values, 'feedback_address': address_values, 'erase': erase_values})
    print(json.dumps({'step': step, 'head_distributions': summaries}, ensure_ascii=False))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
  main()

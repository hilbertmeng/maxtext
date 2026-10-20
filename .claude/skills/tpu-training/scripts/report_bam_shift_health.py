#!/usr/bin/env python3
"""Report cached BAM temporal-shift scalars for any shift variant, without column tables."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('runs', nargs='+')
  parser.add_argument('--steps', nargs='+', type=int, required=True)
  parser.add_argument('--log-root', type=Path, default=Path('/data0/xd/tensorboard_logs'))
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  helper = Path(__file__).with_name('report_bam_read_health.py')
  spec = importlib.util.spec_from_file_location('bam_read_health_cache', helper)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  result = {}
  pattern = re.compile(r'^bam/m_shift/layer_(\d+)/(.+)$')
  for run in args.runs:
    scalars = module.Scalars(args.log_root / run, args.steps)
    tags = {tag: pattern.match(tag) for tag in scalars.tags if pattern.match(tag)}
    summary = {tag: match for tag, match in tags.items() if not match[2].startswith('column_')}
    layer_extent = max((int(match[1]) for match in tags.values()), default=-1) + 1
    points = {}
    nonfinite = []
    for step in sorted(set(args.steps)):
      metrics = {}
      for tag, match in summary.items():
        value = scalars._points.get(tag, {}).get(step)
        if value is None:
          continue
        if not math.isfinite(value):
          nonfinite.append({'tag': tag, 'step': step, 'value': str(value)})
          continue
        metrics.setdefault(match[2], []).append((int(match[1]), value))
      points[step] = {name: {'mean': sum(v for _, v in values) / len(values),
                            'min': min(v for _, v in values),
                            'max': max(v for _, v in values),
                            'min_layer': min(values, key=lambda pair: pair[1])[0],
                            'max_layer': max(values, key=lambda pair: pair[1])[0],
                            'layers': len(values),
                            'band_means': {label: sum(v for _, v in band) / len(band)
                              for index, label in enumerate(('early', 'middle', 'late'))
                              if (band := [(layer, v) for layer, v in values
                                  if min(2, 3 * layer // layer_extent) == index])}}
                      for name, values in sorted(metrics.items())}
    result[run] = {'custom_tags': len(tags), 'summary_tags': len(summary),
                   'layer_count': len({int(match[1]) for match in tags.values()}),
                   'points': points, 'nonfinite': nonfinite}
    print(f'{run}: tags={len(tags)} summaries={len(summary)} nonfinite={len(nonfinite)}')
    for step, row in points.items():
      means = {name: round(values['mean'], 6) for name, values in row.items()
               if name.endswith(('gate_mean', '_mean', 'gate_sum_gt_one_fraction'))}
      print(step, means)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
  if any(row['nonfinite'] for row in result.values()):
    raise SystemExit('nonfinite temporal-shift health metrics')


if __name__ == '__main__':
  main()

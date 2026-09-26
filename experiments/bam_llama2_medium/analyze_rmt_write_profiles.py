#!/usr/bin/env python3
"""Attribute matched RMT train-step traces without counting scan containers twice."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics


def read_trace(path):
  with gzip.open(path, 'rt') as stream:
    events = json.load(stream)['traceEvents']
  device_pids = [e['pid'] for e in events
                 if e.get('name') == 'process_name'
                 and str(e.get('args', {}).get('name', '')).startswith('/device:TPU:')]
  spans = [e for e in events if e.get('pid') in device_pids
           and e.get('ph') == 'X' and e.get('name', '').startswith('jit_train_step(')]
  if not spans:
    raise ValueError(f'No device train step: {path}')
  # Different TPU execution cores execute different subsets of the kernels.
  # Average enclosing wall times, but attribute leaves on one complete core only.
  step = min((e for e in spans if e['pid'] == device_pids[0]), key=lambda e: e['ts'])
  categories = collections.defaultdict(float)
  sources = collections.defaultdict(float)
  ops = collections.defaultdict(float)
  examples = {}
  for event in events:
    if event.get('pid') != step['pid'] or event.get('ph') != 'X':
      continue
    name, args = event.get('name', ''), event.get('args', {})
    if (name.isdigit() or name.startswith(('jit_train_step(', 'while.'))
        or args.get('hlo_category') == 'while'):
      continue
    if not (step['ts'] <= event['ts']
            and event['ts'] + event['dur'] <= step['ts'] + step['dur'] + 1):
      continue
    ms = event['dur'] / 1000
    category = args.get('hlo_category', 'unclassified')
    source = args.get('source', '').split('/MaxText/')[-1]
    op = args.get('tf_op') or f'UNSCOPED/{source}/{category}'
    categories[category] += ms
    sources[source] += ms
    ops[op] += ms
    examples.setdefault(op, dict(name=name, shape=args.get('shape_with_layout'),
                                source=source, category=category))
  wall_ms = step['dur'] / 1000
  leaf_ms = sum(categories.values())
  if abs(leaf_ms - wall_ms) > max(2., wall_ms * .002):
    raise ValueError(f'Leaf attribution does not reconcile: {leaf_ms} vs {wall_ms}: {path}')
  return dict(trace=str(path), device_step_ms=[e['dur']/1000 for e in spans],
              mean_device_step_ms=statistics.mean(e['dur']/1000 for e in spans),
              first_core_step_ms=wall_ms, first_core_leaf_ms=leaf_ms,
              categories_ms=dict(categories), sources_ms=dict(sources),
              ops_ms=dict(ops), examples=examples)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('traces', nargs='+', type=Path, help='Baseline first, then comparison arms')
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  arms = [read_trace(path) for path in args.traces]
  base = arms[0]
  comparisons = []
  for arm in arms[1:]:
    deltas = {}
    for field in ('categories_ms', 'sources_ms', 'ops_ms'):
      a, b = base[field], arm[field]
      deltas[field] = dict(sorted(((k, b.get(k, 0.) - a.get(k, 0.))
                                  for k in a.keys() | b.keys()), key=lambda pair: -abs(pair[1])))
    comparisons.append(dict(trace=arm['trace'],
                            wall_delta_ms=arm['mean_device_step_ms']-base['mean_device_step_ms'],
                            first_core_leaf_delta_ms=arm['first_core_leaf_ms']-base['first_core_leaf_ms'],
                            deltas=deltas))
    print(f"{arm['mean_device_step_ms']:.3f} ms; delta {comparisons[-1]['wall_delta_ms']:+.3f} ms")
    for category, delta in deltas['categories_ms'].items():
      if abs(delta) > .1:
        print(f'  {category}: {delta:+.3f} ms')
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(dict(arms=arms, comparisons=comparisons), indent=2))


if __name__ == '__main__':
  main()

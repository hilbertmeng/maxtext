#!/usr/bin/env python3
"""Compare first complete device-step op attribution, retaining unscoped work."""
import argparse
import collections
import gzip
import json
import re


def read(path):
  with gzip.open(path, 'rt') as stream:
    events = json.load(stream)['traceEvents']
  pid = next(e['pid'] for e in events if e.get('ph') == 'M'
             and e.get('name') == 'process_name'
             and str(e.get('args', {}).get('name', '')).startswith('/device:TPU:'))
  events = [e for e in events if e.get('pid') == pid and e.get('ph') == 'X']
  step = next(e for e in events if e.get('name', '').startswith('jit_train_step('))
  result = collections.defaultdict(lambda: [0., 0., 0])
  for e in events:
    name = e.get('name', '')
    op = e.get('args', {}).get('tf_op', '')
    if name.startswith(('jit_train_step(', 'while.')) or (name.isdigit() and not op):
      continue
    if not (step['ts'] <= e['ts'] and e['ts'] + e['dur'] <= step['ts'] + step['dur'] + 1):
      continue
    key = re.sub(r'/(?:local_\d+|fetch_\d+|layers_\d+)/', '/L/', op)
    key = key or ('UNSCOPED/' + re.sub(r'\.\d+', '.#', name))
    result[key][0] += e['dur'] / 1000
    result[key][1] += float(e.get('args', {}).get('model_flops', 0)) / 1e12
    result[key][2] += 1
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('base')
  parser.add_argument('run')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  base, run = read(args.base), read(args.run)
  rows = []
  for op in base.keys() | run.keys():
    a, b = base.get(op, [0., 0., 0]), run.get(op, [0., 0., 0])
    rows.append(dict(op=op, base=a, run=b, delta_ms=b[0]-a[0]))
  rows.sort(key=lambda x: -abs(x['delta_ms']))
  with open(args.output, 'w') as stream:
    json.dump(dict(base=args.base, run=args.run, rows=rows), stream, indent=2)
  for r in rows[:35]:
    print(f"{r['delta_ms']:+.3f} ms {r['base'][0]:.3f}->{r['run'][0]:.3f} {r['op']}")

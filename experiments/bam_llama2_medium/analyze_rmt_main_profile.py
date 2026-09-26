#!/usr/bin/env python3
"""Partition a sealed MHABudget RMT trace; preserve overlapping subsets explicitly."""
import argparse
import ast
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics


def classifier(source_file):
  tree = ast.parse(source_file.read_text())
  health_lines = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in (
        '_rms', '_read_health', '_write_health', '_gate_health'):
      health_lines.update(range(node.lineno, node.end_lineno + 1))
  for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'RMTLayer':
      method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
      health_start = next(n.lineno for n in method.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'health' for t in n.targets))
      health_lines.update(range(health_start, method.end_lineno))

  def classify(args):
    op = args.get('tf_op', '')
    for scope, label in (
        ('dynamic_qk', 'Dynamic QK'), ('dynamic_vo', 'Dynamic VO C8'),
        ('dynamic_mlp_read', 'Dynamic MLP C8 read'),
        ('dynamic_attn_write', 'Dynamic attention M write'),
        ('dynamic_mlp_write', 'Dynamic MLP M write'),
        ('mlp', 'SwiGLU MLP'), ('attn_vector_norm', 'Vector pre-RMSNorm'),
        ('mlp_vector_norm', 'Vector pre-RMSNorm')):
      if f'/{scope}/' in op:
        return label
    for scope, label in (('qk_logits', 'C256 QK logits'), ('av', 'C256 AV'),
                         ('softmax', 'C256 softmax')):
      if f'/attention/{scope}/' in op:
        return label
    in_layer = bool(re.search(r'/layer_[012]/', op))
    if in_layer and 'btkv,ank->abtnv' in op:
      return 'Static QKV M read'
    if in_layer and 'btkv,kn->btnv' in op:
      return 'Static MLP M read'
    if in_layer and 'btnv,nk->btkv' in op:
      return 'Static attention + MLP M writes'
    if in_layer and ('btd,dr->btr' in op or '/q_rope/' in op or '/k_rope/' in op):
      return 'Independent QK18 projection + RoPE'
    match = re.search(r'/rmt.py:(\d+)$', args.get('source', ''))
    if match and int(match[1]) in health_lines:
      return 'RMT health statistics'
    return 'Scan / residual / LM head / optimizer / other'
  return classify


def analyze(trace, source_file):
  with gzip.open(trace, 'rt') as stream:
    events = json.load(stream)['traceEvents']
  devices = sorted({e['pid'] for e in events if e.get('name') == 'process_name'
                    and str(e.get('args', {}).get('name', '')).startswith('/device:TPU:')})
  classify = classifier(source_file)
  device_rows, walls, counts = [], [], []
  for pid in devices:
    dev = [e for e in events if e.get('pid') == pid and e.get('ph') == 'X']
    leaves = [e for e in dev if not (e['name'].isdigit()
              or e['name'].startswith(('jit_train_step(', 'while.'))
              or e.get('args', {}).get('hlo_category') == 'while')]
    steps = [e for e in dev if e['name'].startswith('jit_train_step(')]
    good, rows = [], defaultdict(lambda: [0., 0., 0.])
    for step in steps:
      inside = [e for e in leaves if step['ts'] <= e['ts']
                and e['ts'] + e['dur'] <= step['ts'] + step['dur'] + 1]
      covered = sum(e['dur'] for e in inside)
      if covered / step['dur'] < .98:
        continue
      if covered > step['dur'] * 1.002:
        raise ValueError('Nested kernels double counted')
      good.append(step['dur'] / 1000)
      for e in inside:
        args = e.get('args', {})
        value = [e['dur'] / 1000, float(args.get('model_flops', 0)) / 1e12,
                 float(args.get('bytes_accessed', 0)) / 1e9]
        labels = [classify(args)]
        op = args.get('tf_op', '')
        if 'dynamic_attn_write/' in op or 'dynamic_mlp_write/' in op:
          sub = 'write transforms / other'
          for equation, name in [('btd,dr->btr', 'write address down'),
                                 ('btr,rd->btd', 'write address up'),
                                 ('btd,dn->btn', 'write gate projection'),
                                 ('btnk,btnv->btkv', 'dynamic outer write')]:
            if equation in op:
              sub = name
              break
          labels.append('subset: ' + sub)
        if '/dynamic_qk/' in op:
          if 'btd,dr->btr' in op or 'btd,dn->btn' in op:
            labels.append('subset: QK basis / mix / gate projections')
          elif 'btvc,btrc->btrv' in op:
            labels.append('subset: QK basis M contraction')
          elif 'btrv,btnr->btnv' in op:
            labels.append('subset: QK rank-to-head expansion')
          else:
            labels.append('subset: QK Gram / RMS / gating / other')
        if e['name'].startswith('copy.') and args.get('hlo_category') == 'data formatting':
          labels.append('subset: all copy kernels')
        for label in labels:
          rows[label] = [a + b for a, b in zip(rows[label], value)]
    if not good:
      raise ValueError(f'No complete step on device {pid}')
    walls.append(statistics.mean(good)); counts.append(len(good))
    device_rows.append({k: [v / len(good) for v in val] for k, val in rows.items()})
  names = set().union(*(r.keys() for r in device_rows))
  result = {k: [statistics.mean(r.get(k, [0, 0, 0])[i] for r in device_rows)
                for i in range(3)] for k in names}
  total = [sum(v[i] for k, v in result.items() if not k.startswith('subset:'))
           for i in range(3)]
  wall = statistics.mean(walls)
  assert abs(total[0] - wall) < wall * .002, (total[0], wall)
  return dict(trace=str(trace), source=str(source_file), devices=len(devices),
              complete_steps_per_device=counts, step_ms=wall,
              attributed_total=total, rows=result)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('trace', type=Path)
  p.add_argument('--source', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  result = analyze(args.trace, args.source)
  args.output.write_text(json.dumps(result, indent=2))
  for name, values in sorted(result['rows'].items(), key=lambda x: -x[1][0]):
    print(name, *(round(v, 4) for v in values))
  print('complete', result['step_ms'], 'devices', result['devices'],
        'steps', result['complete_steps_per_device'])


if __name__ == '__main__':
  main()

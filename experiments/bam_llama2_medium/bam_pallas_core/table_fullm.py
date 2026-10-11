#!/usr/bin/env python3
"""XPlane part table for the pure-JAX full-M generalized column read (XLProp), vs an MHA arm.

usage: table_fullm.py ARM=ops.json [ARM=ops.json ...]   (ops.json from xplane_ops.py)
Rows: exclusive ms per step; the TF/GB columns of the last arm show achieved TFLOP/s and GB/s.
First matching rule wins; 'B' scope = transpose(jvp) (reverse incl. remat recompute is under F/B by XLA naming).
"""
import collections
import json
import sys


def is_bwd(o):
  return 'transpose(' in o


RULES = [
    ('BAM health statistics (sow)', lambda o, h: '_record_' in o),
    ('Attention core (Splash)', lambda o, h: 'splash' in h.lower() or '_bam_splash' in o or 'attention_op' in o
                                             or '_query_chunk_op' in o),
    ('SwiGLU MLP', lambda o, h: '/mlp/' in o and 'mlp_address' not in o and 'mlp_write' not in o),
    ('Standard Q/K(/V) projections', lambda o, h: 'query_projection' in o or 'kv_projection' in o),
    ('QKNorm + RoPE', lambda o, h: 'apply_rotary_embedding' in o or '/qk_norm/' in o),
    ('O projection', lambda o, h: 'out_projection' in o),
    ('Full-M key bottleneck + up projections (+pre-bias)', lambda o, h: any(
        s in o for s in ('full_m_read_down', '_c8_up', 'W_R_up', '_project_column_read_key'))),
    ('Read/static gates projections', lambda o, h: 'read_gate_projection' in o or '_static_gate' in o),
    ('Static Q/K/V/O reads (M · static key)', lambda o, h: '_static_column' in o),
    ('M read normalization', lambda o, h: 'bam/normalize_m' in o or '_matrix_for_read' in o),
    ('LocalQK full-M dynamic read', lambda o, h: 'read_local_m_for_qk' in o),
    ('LocalVO full-M dynamic read + gating', lambda o, h: '_independent_local_vo' in o),
    ('LocalQK concat into heads', lambda o, h: '_add_local_qk' in o),
    ('Attention write (factors + outer product)', lambda o, h: 'bam/write_m' in o or '_deferred_write_factors' in o
                                                         or '/P_loc_' in o or '/W_gw/' in o or 'bam/write_outer' in o),
    ('MLP write (projections + merge)', lambda o, h: 'merge_mlp_write' in o or 'mlp_address_' in o
                                                     or 'mlp_write_gate' in o or 'mlp_independent_write' in o),
    ('Embedding write', lambda o, h: 'initial_bam_matrix' in o or 'embedding_bam_write' in o),
    ('Layer norms', lambda o, h: 'layer_norm' in o),
    ('LM head / loss', lambda o, h: '/lm_head/' in o),
    ('Scan carry / optimizer / unscoped / other', lambda o, h: True),
]
SUBSETS = [
    ('↳ all copy kernels (cross-cutting)', lambda o, h: h.startswith('copy')),
    ('↳ backward/remat scope (cross-cutting)', lambda o, h: is_bwd(o)),
]


def partition(path):
  r = json.load(open(path))
  rows = collections.defaultdict(lambda: [0., 0., 0.])
  subs = collections.defaultdict(lambda: [0., 0., 0.])
  for x in r['ops']:
    o, h, v = x['tf_op'], x['hlo'], (x['ms'], x['tf'], x['gb'])
    for name, pred in RULES:
      if pred(o, h):
        for i in range(3):
          rows[name][i] += v[i]
        break
    for name, pred in SUBSETS:
      if pred(o, h):
        for i in range(3):
          subs[name][i] += v[i]
  step = sum(r['step_ms']) / len(r['step_ms'])
  return rows, subs, step


def main():
  arms = [a.split('=', 1) for a in sys.argv[1:]]
  parts = {name: partition(path) for name, path in arms}
  names = [n for n, _ in arms]
  last = names[-1]
  print('| Part | ' + ' | '.join(f'{n} ms' for n in names) + f' | {last} TF/s | {last} GB/s |')
  print('|---|' + '---:|' * (len(names) + 2))
  for rule, _ in RULES:
    vals = [parts[n][0][rule][0] for n in names]
    if any(v > 0.05 for v in vals):
      ms, tf, gb = parts[last][0][rule]
      rate = lambda q: f'{q / ms * 1000:.0f}' if ms > 0.5 else '-'
      print(f'| {rule} | ' + ' | '.join(f'{v:.1f}' for v in vals) + f' | {rate(tf)} | {rate(gb)} |')
  print('| **Device step** | ' + ' | '.join(f'**{parts[n][2]:.1f}**' for n in names) + ' | | |')
  for rule, _ in SUBSETS:
    print(f'| {rule} | ' + ' | '.join(f'{parts[n][1][rule][0]:.1f}' for n in names) + ' | | |')
  for n in names:
    tot = [sum(v[i] for v in parts[n][0].values()) for i in range(3)]
    print(f'{n}: executed {tot[1]:.1f} TF/step ({tot[1] / tot[0] * 1000:.0f} TF/s avg), HBM {tot[2]:.0f} GB/step')


if __name__ == '__main__':
  main()

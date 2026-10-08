#!/usr/bin/env python3
"""Partition exclusive XPlane op time of paired arms into additive parts (first matching rule wins).

usage: table.py ARM=ops.json [ARM=ops.json ...]   (ops.json from xplane_ops.py)
"""
import collections
import json
import sys


def is_bwd(o):
  return 'transpose(' in o


RULES = [
    ('Fused read kernel (fwd)', lambda o, h: ('bam_core_read' in h or 'bam/pallas_read' in o) and 'backward' not in h and not is_bwd(o)),
    ('Fused read kernel (bwd)', lambda o, h: 'bam_core_read' in h or 'bam/pallas_read' in o),
    ('Fused write kernel (fwd)', lambda o, h: ('bam_core_write' in h or 'bam/pallas_write' in o) and 'backward' not in h and not is_bwd(o)),
    ('Fused write kernel (bwd)', lambda o, h: 'bam_core_write' in h or 'bam/pallas_write' in o),
    ('Concat/BAM health statistics', lambda o, h: '_record_concat' in o),
    ('Attention core (MHA Splash / BAM C256 QChunk)', lambda o, h: 'attention_op' in o or '_query_chunk_op' in o),
    ('SwiGLU MLP', lambda o, h: '/mlp/' in o),
    ('Standard Q/K(/V) projections', lambda o, h: 'query_projection' in o or 'kv_projection' in o),
    ('QKNorm + RoPE', lambda o, h: 'apply_rotary_embedding' in o or '/qk_norm/' in o),
    ('O projection', lambda o, h: 'out_projection' in o),
    ('BAM key/gate projections (C10 keys, W_R, gates)', lambda o, h: any(s in o for s in ('/W_lq_c8/', '/W_lk_c8/', '/W_R/', 'read_gate_projection'))),
    ('LocalQK concat into heads (XLA path)', lambda o, h: '_add_local_qk' in o),
    ('C10 compression (XLA path)', lambda o, h: 'compress_abs_v_cache' in o),
    ('LocalQK direct read + static QK (XLA path)', lambda o, h: 'read_local_m_for_qk' in o),
    ('LocalVO read/gating (XLA path)', lambda o, h: '_independent_local_vo' in o),
    ('Static V/O reads (XLA path)', lambda o, h: '_static_column' in o),
    ('Attention write projections + XLA write', lambda o, h: 'bam/write_m' in o or '_deferred_write_factors' in o or '/P_loc_' in o or '/W_gw/' in o),
    ('MLP write projections + XLA merge', lambda o, h: 'merge_mlp_write' in o or 'mlp_address_' in o or 'mlp_write_gate' in o),
    ('Embedding write', lambda o, h: 'initial_bam_matrix' in o),
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
  return rows, subs, step, r['coverage']


def main():
  arms = [a.split('=', 1) for a in sys.argv[1:]]
  parts = {name: partition(path) for name, path in arms}
  names = [n for n, _ in arms]
  print('| Part | ' + ' | '.join(f'{n} ms' for n in names) + ' |')
  print('|---|' + '---:|' * len(names))
  for rule, _ in RULES:
    vals = [parts[n][0][rule][0] for n in names]
    if any(v > 0.05 for v in vals):
      print(f'| {rule} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
  print('| **Device step** | ' + ' | '.join(f'**{parts[n][2]:.2f}**' for n in names) + ' |')
  for rule, _ in SUBSETS:
    print(f'| {rule} | ' + ' | '.join(f'{parts[n][1][rule][0]:.2f}' for n in names) + ' |')
  print('coverage', {n: [round(c, 4) for c in parts[n][3]] for n in names})


if __name__ == '__main__':
  main()

#!/usr/bin/env python3
"""Partition verified raw-XPlane leaf aggregates by compiler AD scope.

These are additive primary-core source attributions, not isolated execution
experiments. Fused kernels inherit a dominant scope; optimizer work can be
fused into gradient scopes. Remat is separated before testing transpose(jvp).
"""
import argparse
import collections
import json
import math
from pathlib import Path


def phase(op):
  if 'rematted_computation' in op:
    return 'recompute'
  if 'transpose(jvp' in op:
    return 'backward'
  if 'jvp(' in op:
    return 'forward'
  return 'other'


def component(op):
  if '/attention/' in op or '/layers/jit(_where)/' in op:
    return 'attention'
  if '/layers/mlp/' in op:
    return 'MLP'
  if '/dynamic_attn_write/' in op or '/dynamic_mlp_write/' in op:
    return 'layer_writes'
  if '/dynamic_qk/' in op:
    return 'dynamic_QK'
  if '/dynamic_vo/' in op or '/dynamic_mlp_read/' in op:
    return 'dynamic_C8'
  if '/attn_vector_norm/' in op or '/mlp_vector_norm/' in op:
    return 'proxy_norm'
  if '/lm_head/' in op:
    return 'LM_head'
  return 'remaining'


def summarize(arm):
  phases=collections.Counter()
  components=collections.defaultdict(collections.Counter)
  write_kernels=collections.Counter()
  for op,ms in arm['ops_ms'].items():
    p=phase(op)
    phases[p]+=ms
    components[component(op)][p]+=ms
    if ('rmt_token_minor_write' in op or 'rmt_write_reverse_major' in op) and 'pallas_call' in op:
      write_kernels[p]+=ms
  assert math.isclose(sum(phases.values()),arm['first_core_leaf_ms'],abs_tol=1e-6)
  wall=arm['first_core_step_ms']
  return dict(trace=arm['trace'],wall_ms=wall,leaf_ms=arm['first_core_leaf_ms'],
              unattributed_ms=arm['first_core_unattributed_ms'],
              phases_ms=dict(phases),phases_percent={k:100*v/wall for k,v in phases.items()},
              components_ms={k:dict(v) for k,v in components.items()},
              write_pallas_ms=dict(write_kernels))


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('comparisons',nargs='+',type=Path)
  p.add_argument('--output',required=True,type=Path)
  args=p.parse_args()
  rows=[]
  for path in args.comparisons:
    for arm in json.loads(path.read_text())['arms']:
      row=summarize(arm)
      rows.append(row)
      print(Path(arm['trace']).parts[5],row['phases_ms'])
  args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
  main()

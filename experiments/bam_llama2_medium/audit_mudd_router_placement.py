#!/usr/bin/env python3
"""Audit full-history Mudd router widths from downloaded Orbax metadata, without restoring weights.

Use --verify-model with the pinned CPU environment to compare old/new actual parameter trees.
The historical exp module is loaded in isolation so unfinished edits to main exp.py do not affect it.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types


def checkpoint_routers(metadata, layers):
  leaves = {}
  for value in metadata['tree_metadata'].values():
    path = tuple(k['key'] for k in value['key_metadata'])
    if path[:2] == ('params', 'params'):
      leaves[path[2:]] = value['value_metadata'].get('write_shape')
  rows = []
  for path, shape in leaves.items():
    if path[-2:] != ('dynamic_dense_conn1', 'kernel'):
      continue
    module = path[:-2]
    second = leaves[module + ('dynamic_dense_conn2', 'kernel')]
    channels, history = second[-2:]
    final = 'compose_final' in module
    legacy_expand = 4 if history == layers - 1 else 1
    intended_expand = 4 if final else 1
    rounded = lambda expand: (channels * history * expand // 64 + 1) * 64
    rows.append(dict(module='/'.join(module), history_length=history, output_channels=channels,
                     checkpoint_hidden_width=shape[-1], checkpoint_write_shapes=[shape, second],
                     legacy_expand=legacy_expand, legacy_hidden_width=rounded(legacy_expand),
                     intended_expand=intended_expand, intended_hidden_width=rounded(intended_expand)))
  rows.sort(key=lambda r: r['history_length'])
  if len(rows) != layers or [r['history_length'] for r in rows] != list(range(2, layers + 2)):
    raise ValueError('Expected full history: routers with history lengths2 through L+1')
  return rows


def verify_model(args, rows):
  os.environ['JAX_PLATFORMS'] = 'cpu'
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
  root = Path(__file__).resolve().parents[2]
  sys.path[:0] = [str(root / 'MaxText'), str(root / 'MaxText/tests')]
  historical = types.ModuleType('exp')
  source = subprocess.check_output(['git', 'show', f'{args.runtime_commit}:MaxText/exp.py'], cwd=root, text=True)
  exec(compile(source, f'{args.runtime_commit}:MaxText/exp.py', 'exec'), historical.__dict__)
  sys.modules['exp'] = historical
  import jax
  from flax import linen as nn
  from flax.traverse_util import flatten_dict
  from pythia_mudd_test import PythiaMuddTest
  from layers.mudd import Mlp

  case = PythiaMuddTest()
  case.setUp()
  result = {}
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      cfg = case.config(exp_class=args.exp_class, base_emb_dim=1200, base_num_decoder_layers=args.layers,
                        base_num_query_heads=16, base_num_kv_heads=16, head_dim=75,
                        base_mlp_dim=4800, pythia_rotary_dim=18, vocab_size=50432)
    for label, expands in [('historical_default', None), ('final_only_expand4', [1] * (args.layers - 1) + [4])]:
      cfg.get_keys()['mudd_router_hidden_expand'] = expands
      model, mesh = case.model(cfg)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), contextlib.redirect_stdout(io.StringIO()):
        params = jax.eval_shape(lambda key: model.init(key, *case.inputs(), enable_dropout=False)['params'],
                                jax.random.key(0))
      flat = flatten_dict(nn.unbox(params))
      shapes = {'/'.join(path): list(value.shape) for path, value in flat.items()
                if 'dynamic_dense_conn1' in path or 'dynamic_dense_conn2' in path}
      for row in rows:
        expected = row['checkpoint_hidden_width'] if expands is None else row['intended_hidden_width']
        assert shapes[row['module'] + '/dynamic_dense_conn1/kernel'][-1] == expected, row
      result[label] = dict(parameters=sum(math.prod(v.shape) for v in flat.values()), router_shapes=shapes)
    # Check the user's32-layer schedule on the real Mudd module as well.
    cfg.get_keys().update(num_decoder_layers=32, emb_dim=2560, mudd_router_hidden_expand=[1] * 31 + [4])
    result['32layer_schedule'] = []
    for history, channels in ((31, 4), (32, 4), (33, 1)):
      module = Mlp(cfg, None, None, hids_length=history, C=channels)
      with nn.partitioning.axis_rules(cfg.logical_axis_rules), contextlib.redirect_stdout(io.StringIO()):
        params = jax.eval_shape(lambda key, x: module.init(key, x), jax.random.key(0),
                                jax.ShapeDtypeStruct((1, 1, 2560), cfg.dtype))
      shape = nn.unbox(params['params'])['dynamic_dense_conn1']['kernel'].shape
      result['32layer_schedule'].append(dict(history_length=history, list_index=history - 2,
                                             multiplier=cfg.mudd_router_hidden_expand[history - 2],
                                             hidden_width=shape[-1]))
  finally:
    case.tearDown()
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--metadata', type=Path, required=True)
  parser.add_argument('--checkpoint-uri', required=True)
  parser.add_argument('--runtime-commit', default='3538f6c57c6a548436a89310a01d318c6b8e9234')
  parser.add_argument('--exp-class', default='MuddPythiaMediumPropT2048SerialMLP')
  parser.add_argument('--layers', type=int, default=18)
  parser.add_argument('--verify-model', action='store_true')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  rows = checkpoint_routers(json.loads(args.metadata.read_text()), args.layers)
  result = dict(exp_class=args.exp_class, runtime_commit=args.runtime_commit, checkpoint=args.checkpoint_uri,
                metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(), routers=rows,
                legacy_rule_matches_checkpoint=all(r['checkpoint_hidden_width'] == r['legacy_hidden_width'] for r in rows),
                placement_mismatches=[r for r in rows if r['checkpoint_hidden_width'] != r['intended_hidden_width']],
                note='write_shape is per-shard; hidden width is the unsharded second axis of conn1 here.')
  if args.verify_model:
    result['cpu_verification'] = verify_model(args, rows)
  result['current_mudd_source_sha256'] = hashlib.sha256(
      (Path(__file__).resolve().parents[2] / 'MaxText/layers/mudd.py').read_bytes()).hexdigest()
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({k: v for k, v in result.items() if k not in ('routers', 'cpu_verification')}, indent=2))
  if 'cpu_verification' in result:
    print('PARAMETERS', {k: v['parameters'] for k, v in result['cpu_verification'].items() if isinstance(v, dict)})
    print('32_LAYER', result['cpu_verification']['32layer_schedule'])


if __name__ == '__main__':
  main()

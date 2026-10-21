#!/usr/bin/env python3
"""Resolve the three Pythia comparison arms and count their actual parameter trees.

Run with the pinned maxtext-cpu environment. No weights are allocated or restored.
The full-length configuration is recorded before shortening sequence length and
selecting the JAX attention implementation solely for abstract parameter counting.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ARMS = ('Pythia2p8B', 'MuddPythia3BSerialMLP',
        'BamPythia2p8BSotaIndependentR768Pallas')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ['JAX_PLATFORMS'] = 'cpu'
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
  root = Path(__file__).resolve().parents[2]
  os.chdir(root)
  sys.path[:0] = [str(root / 'MaxText'), str(root / 'MaxText/tests')]
  from flax import linen as nn
  from flax.traverse_util import flatten_dict
  import jax
  import pyconfig
  from pythia_test import PythiaTest

  result = dict(arms={}, parameter_count_overrides=dict(
      max_target_length=8, max_prefill_predict_length=4, loss_chunk_size=4,
      attention='dot_product', bam_splash_attention=False, bam_pallas_core=False))
  case = PythiaTest()
  for arm in ARMS:
    with tempfile.TemporaryDirectory() as output, contextlib.redirect_stdout(io.StringIO()):
      Path(output, 'audit').mkdir()
      cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'],
          exp_class=arm, run_name='audit', base_output_directory=output+'/',
          enable_checkpointing=False, jax_cache_dir='', log_config=False)
      resolved = {key: repr(value) for key, value in cfg.get_keys().items()}
      cfg.get_keys().update(result['parameter_count_overrides'])
      model, mesh = case.model(cfg)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params = jax.eval_shape(lambda key: model.init(
            key, *case.inputs(), enable_dropout=False)['params'], jax.random.key(19))
      flat = flatten_dict(nn.unbox(params))
      buckets = {}
      mlps, routers, norms = {}, {}, {}
      for path, value in flat.items():
        name = '/'.join(path)
        if 'dynamic_dense_conn1' in path or 'dynamic_dense_conn2' in path or any(
            part.startswith('dense_proj2.') for part in path):
          bucket = 'router'
          routers[name] = list(value.shape)
        elif 'mlp' in path:
          bucket = 'mlp'
          mlps[name] = list(value.shape)
        elif 'self_attention' in path:
          bucket = 'attention'
        else:
          bucket = 'other'
        buckets[bucket] = buckets.get(bucket, 0) + math.prod(value.shape)
        if 'qk_norm' in path:
          norms[name] = list(value.shape)
      result['arms'][arm] = dict(resolved=resolved, parameters=sum(buckets.values()),
          parameter_buckets=buckets, mlp_shapes=mlps, router_shapes=routers,
          qk_norm_shapes=norms)
    print(arm, result['arms'][arm]['parameters'], buckets, flush=True)
  keys = sorted(set().union(*(data['resolved'] for data in result['arms'].values())))
  result['differences'] = {key: {arm: data['resolved'].get(key, '<absent>')
      for arm, data in result['arms'].items()} for key in keys
      if len({data['resolved'].get(key, '<absent>') for data in result['arms'].values()}) > 1}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print('OUTPUT', args.output)


if __name__ == '__main__':
  main()

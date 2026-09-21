"""Count actual parameter shapes without allocating weights (CPU, short sequence)."""
import argparse
import json
import math
import tempfile
from pathlib import Path

import jax
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import train_compile


def audit(exp, check_sharding=False):
  with tempfile.TemporaryDirectory() as out:
    Path(out, 'audit').mkdir()
    cfg = pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=exp, run_name='audit',
        enable_checkpointing=False, base_output_directory=out+'/', jax_cache_dir='',
        log_config=False, dataset_type='synthetic', max_target_length=8,
        max_prefill_predict_length=8, query_chunk_size=4, per_device_batch_size=1.)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    args, _, shardings, _ = train_compile.get_shaped_inputs(mesh, cfg)
    leaves = {'/'.join(k): math.prod(v.shape)
              for k, v in flatten_dict(args[0].params).items()}
    groups = {}
    for path, count in leaves.items():
      group = next((x for x in ('local_0','local_1','fetch_2','final_local_layer') if x in path.split('/')), 'other')
      groups[group] = groups.get(group, 0) + count
    if check_sharding:
      assert jax.device_count() > 1, 'Use XLA_FLAGS=--xla_force_host_platform_device_count=8'
      shapes = flatten_dict(args[0].params)
      shards = flatten_dict(shardings.params)
      per_chip = sum(math.prod(shards[path].shard_shape(leaf.shape))
                     for path, leaf in shapes.items())
      axes = ('fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose',
              'tensor_sequence', 'stage', 'expert')
      devices = math.prod(mesh.shape[axis] for axis in axes)
      overhead = per_chip / (sum(leaves.values()) / devices) - 1
      print('SHARDING_AUDIT', exp, 'overhead', overhead, 'limit', cfg.sharding_tolerance)
      assert overhead < cfg.sharding_tolerance
    return dict(exp=exp, total=sum(leaves.values()), groups=groups, leaves=leaves,
                wd_mults=cfg.wd_mults, mlp_dim=cfg.mlp_dim,
                local_v_scale=cfg.bam_local_v_key_scale,
                mlp_pattern=getattr(cfg, 'mlp_dim_by_block', None),
                generic_health=getattr(cfg, 'record_training_health_metrics', True))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('exp', nargs='+')
  parser.add_argument('--output', required=True)
  parser.add_argument('--check-sharding', action='store_true')
  args = parser.parse_args()
  results = [audit(exp, args.check_sharding) for exp in args.exp]
  Path(args.output).write_text(json.dumps(results, indent=2)+'\n')
  for result in results:
    print('PARAM_AUDIT', {k:v for k,v in result.items() if k != 'leaves'})

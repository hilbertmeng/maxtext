"""Abstract full-24 parameter budget and scan-train shape check for QK column ablations."""
import argparse
import json
import math
import tempfile
from pathlib import Path
import jax
from flax.traverse_util import flatten_dict
from flax.linen import partitioning
import max_utils
import pyconfig
import train
import train_compile

BASE = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis'

def audit(name):
  with tempfile.TemporaryDirectory() as out:
    Path(out, 'audit').mkdir()
    cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=name,
        run_name='audit', base_output_directory=out+'/', enable_checkpointing=False,
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        max_target_length=8, max_prefill_predict_length=8,
        query_chunk_size=4, per_device_batch_size=1.)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    args, _, shard, model = train_compile.get_shaped_inputs(mesh, cfg)
    leaves = {'/'.join(k): math.prod(v.shape) for k,v in flatten_dict(args[0].params).items()}
    with mesh, partitioning.axis_rules(cfg.logical_axis_rules):
      _, metrics = jax.eval_shape(lambda s,d,r: train.train_step(model,cfg,shard,s,d,r), *args)
    assert 'learning/raw_grad_norm' in metrics['scalar']
    assert cfg.wd_mults == [] and cfg.scan_layers and cfg.checkpoint_period == 250
    return dict(name=name,total=sum(leaves.values()),leaves=leaves,
                mlp_dim=cfg.mlp_dim,wd_mults=cfg.wd_mults,
                nope_dim=cfg.bam_partial_rope_nope_dim)

if __name__ == '__main__':
  p=argparse.ArgumentParser()
  p.add_argument('--output',required=True)
  a=p.parse_args()
  results=[audit(n) for n in (BASE,'BamXLSharedBasisQKColOnlyMLP','BamXLSharedBasisQKDirectC8MLP')]
  Path(a.output).write_text(json.dumps(results,indent=2)+'\n')
  for r in results: print('AUDIT', {k:v for k,v in r.items() if k!='leaves'})

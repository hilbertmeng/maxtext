"""CPU shape audit of the real LLF train step and per-layer TB capture."""
import json
import math
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from flax.linen import partitioning as nn_partitioning
import pyconfig
import max_utils
import train
import train_compile

names = ['BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow',
         'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRow',
         'BamMediumIndependentLLFBAlignedRowLocalOStaticDynamicRowNoNorm']
results = []
for name in names:
  with tempfile.TemporaryDirectory() as out:
    Path(out, 'audit').mkdir()
    cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=name,
        run_name='audit', base_output_directory=out+'/', enable_checkpointing=False,
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        max_target_length=8,max_prefill_predict_length=8,query_chunk_size=4,
        per_device_batch_size=1.)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    args, _, shardings, model = train_compile.get_shaped_inputs(mesh,cfg)
    leaves = flatten_dict(args[0].params)
    result = dict(name=name, parameters=sum(math.prod(v.shape) for v in leaves.values()))
    if name != names[0]:
      assert cfg.record_training_health_metrics
      for flag in ('record_internal_nn_metrics', 'bam_record_local_routing_metrics',
                   'bam_record_fetched_read_health_metrics', 'bam_record_fetch_route_metrics',
                   'bam_record_fetched_read_amplitude_metrics', 'bam_record_local_qk_amplitude_metrics'):
        assert not getattr(cfg, flag, False), flag
      wd = flatten_dict(train.get_wd_tree(cfg, args[0].params))
      for path in leaves:
        if 'o_row_static_scale' in path:
          assert wd[path] == 0, (path, wd[path])
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        _, metrics = jax.eval_shape(lambda *a: train.train_step(model,cfg,shardings,*a), *args)
      tags = [tag for tag in metrics['scalar'] if tag.startswith('bam/local_o_row_branches/')]
      assert len([t for t in tags if t.endswith('/static_energy_share')]) == 17, tags
      assert all(metrics['scalar'][tag].shape == () for tag in tags)
      assert len([t for t in tags if t.endswith('/a_over_a0')]) == (16 if name == names[1] else 0)
      result['health_tag_count'] = len(tags)
    results.append(result)
print('AUDIT_RESULT', json.dumps(results))

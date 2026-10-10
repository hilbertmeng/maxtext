"""Abstract eight-device parameter layouts; reject replicated read-up weights."""
import math,tempfile
from pathlib import Path
import jax
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,pyconfig,train_compile
assert jax.device_count()==8
for name in ('BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile',
             'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile'):
 with tempfile.TemporaryDirectory() as directory:
  Path(directory,'audit').mkdir()
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',
      enable_checkpointing=False,base_output_directory=directory+'/',jax_cache_dir='',
      log_config=False,dataset_type='synthetic',max_target_length=4,
      max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,bam_splash_attention=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  args,_,shardings,_=train_compile.get_shaped_inputs(mesh,cfg)
  params=flatten_dict(nn.unbox(args[0].params));parts=flatten_dict(nn.unbox(shardings.params))
  total=sum(math.prod(v.shape) for v in params.values())
  local=sum(math.prod(parts[k].shard_shape(v.shape)) for k,v in params.items())
  factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'))
  overhead=local/(total/factor)-1
  assert overhead<cfg.sharding_tolerance,(name,overhead,cfg.sharding_tolerance)
  for k,v in params.items():
   if any(n in k for n in ('W_lq_c8_up','W_lk_c8_up','W_R_up')):
    assert math.prod(parts[k].shard_shape(v.shape))==math.prod(v.shape)//8,(k,v.shape,parts[k])
  print('EIGHT_DEVICE_PARAMETER_LAYOUT_OK',name,overhead,flush=True)

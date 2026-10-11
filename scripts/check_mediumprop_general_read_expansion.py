"""Config-only expansion checks: exact budgets, full scan trees and tiny finite gradients."""
import json,math,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,pyconfig,train_compile,train
from layers.models import Transformer
rows=[{'label': 'A', 'run': 'BamMediumPropK75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile', 'layers': 18, 'v': 48, 'read_rank': 384, 'widths': [3540, 3325, 3540], 'global_params': 123253696, 'nonmlp': [4413936.0, 5189632.0, 4413936.0], 'count': 432106720, 'mha': 432121200, 'delta_mha': -14480}, {'label': 'B', 'run': 'BamMediumPropL27K75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile', 'layers': 27, 'v': 32, 'read_rank': 256, 'widths': [2223, 2096, 2223], 'global_params': 122936000, 'nonmlp': [3449328.0, 3907328.0, 3449328.0], 'count': 432150656, 'mha': 432142800, 'delta_mha': 7856}, {'label': 'C', 'run': 'BamMediumPropL27K75V48EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu384GeneralReadTruePile', 'layers': 27, 'v': 48, 'read_rank': 384, 'widths': [1952, 1736, 1952], 'global_params': 123253696, 'nonmlp': [4413936.0, 5189632.0, 4413936.0], 'count': 432147232, 'mha': 432142800, 'delta_mha': 4432}]
for row in rows:
 with tempfile.TemporaryDirectory() as d:
  Path(d,'audit').mkdir()
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=row['run'],run_name='audit',enable_checkpointing=False,base_output_directory=d+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,bam_splash_attention=False)
  assert cfg.num_decoder_layers==row['layers'] and cfg.bam_v==row['v']
  assert cfg.bam_local_full_m_read_bottleneck_dim==row['read_rank']
  assert cfg.mlp_dim_by_block==row['widths'] and not cfg.bam_static_read_gate_sigmoid
  assert cfg.bam_general_column_read and not cfg.bam_general_matrix_write and cfg.bam_static_read_gate_init==.99
  assert cfg.bam_abs_v_compression_dim is None and not cfg.bam_local_o_compress_v
  assert cfg.bam_record_concat_health and cfg.record_training_health_metrics and cfg.scan_layers
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  args,_,parts,_=train_compile.get_shaped_inputs(mesh,cfg)
  flat=flatten_dict(nn.unbox(args[0].params));count=sum(math.prod(v.shape) for v in flat.values());assert count==row['count'],(row,count)
  ps=flatten_dict(nn.unbox(parts.params));local=sum(math.prod(ps[k].shard_shape(v.shape)) for k,v in flat.items());factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'));overhead=local/(count/factor)-1
  assert overhead<cfg.sharding_tolerance,(row['label'],overhead)
  print('EXPANSION_TREE_OK',row['label'],count,overhead,flush=True)
  # Keep the actual 18/27 scan length and V32/V48; reduce only batch, width, ranks and vocabulary.
  cfg.get_keys().update(base_emb_dim=150,emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,num_query_heads=2,num_kv_heads=2,emb_bam_num_head=2,bam_mlp_write_num_heads=0,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,base_mlp_dim=16,mlp_dim=16,mlp_dim_by_block=[16,12,16],vocab_size=32,dtype='float32',weight_dtype='float32')
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
  tokens=jnp.array([[1,4,8,2]],jnp.int32);pos=jnp.arange(4,dtype=jnp.int32)[None];seg=jnp.ones_like(tokens);rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   params=nn.unbox(model.init(rng,tokens,pos,seg,tokens)['params'])
   def forward(p,health=False):
    result=model.apply({'params':p},tokens,pos,seg,tokens,enable_dropout=False,rngs=rng,mutable=['intermediates'] if health else False)
    if health:
     out,metrics=result;return (out[0] if isinstance(out,tuple) else out),metrics
    return result[0] if isinstance(result,tuple) else result
   value,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(forward(p).astype(jnp.float32)**2)))(params)
   assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in flatten_dict(grad).values())
   _,inter=forward(params,True);metrics={'scalar':{}};train.record_bam_concat_health_metrics(metrics,inter,cfg)
   exported=metrics['scalar'];assert exported and all(np.isfinite(np.asarray(v)).all() for v in exported.values())
   pf=flatten_dict(params);biases=[v for p,v in pf.items() if p[-1].startswith('general_') and p[-1].endswith('_static_gate_bias')];assert len(biases)>0
   for v in biases:np.testing.assert_allclose(np.asarray(v),math.log(9),rtol=2e-6)
   for k,v in exported.items():
    if k.endswith('/mean') and '_static_gate/' in k:np.testing.assert_allclose(float(v),.99,rtol=2e-6)
   for name in ('general_q_pre_bias','general_k_pre_bias','general_vo_pre_bias','general_v_static_gate'):
    leaves=[v for p,v in flatten_dict(grad).items() if name in p];assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,name
   print('EXPANSION_FORWARD_GRAD_HEALTH_OK',row['label'],len(exported),float(value),flush=True)

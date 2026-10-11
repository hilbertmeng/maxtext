"""Targeted config-only rank sweep: actual budget/layout and active read gradients."""
import math,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,pyconfig,train_compile,train
from layers.models import Transformer
STEM='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu'
for rank,expected,widths in ((128,432136160,[3911,3783,3911]),(384,432129248,[3716,3589,3716])):
 with tempfile.TemporaryDirectory() as d:
  Path(d,'audit').mkdir()
  c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=STEM+str(rank)+'GeneralReadTruePile',run_name='audit',enable_checkpointing=False,base_output_directory=d+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,bam_splash_attention=False)
  assert c.bam_general_column_read and not c.bam_general_matrix_write and c.bam_local_full_m_read_share_down
  assert c.bam_local_full_m_read_bottleneck_dim==rank and c.mlp_dim_by_block==widths
  assert c.bam_abs_v_compression_dim is None and not c.bam_local_o_compress_v
  assert c.bam_k==75 and c.bam_v==32 and c.bam_read_key_scale==.1 and c.bam_static_read_gate_init==.99
  assert c.DATASET_VARIANT=='truepile4096' and c.bam_local_vo_independent_gates and not c.qk_norm
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  shaped,_,parts,_=train_compile.get_shaped_inputs(mesh,c);f=flatten_dict(nn.unbox(shaped[0].params));sp=flatten_dict(nn.unbox(parts.params))
  count=sum(math.prod(v.shape) for v in f.values());assert count==expected,(rank,count)
  factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'))
  overhead=sum(math.prod(sp[k].shard_shape(v.shape)) for k,v in f.items())/(count/factor)-1
  assert overhead<c.sharding_tolerance,(rank,overhead)
  for path,value in f.items():
   if 'full_m_read_down' in path:assert value.shape==(1200,6,rank),(path,value.shape)
   if any(n in path for n in ('W_lq_c8_up','W_lk_c8_up','W_R_up')):
    expected_shape=(rank,6,16,1,32) if 'W_R_up' in path else (rank,6,16,32)
    assert value.shape==expected_shape,(path,value.shape)
    assert math.prod(sp[path].shard_shape(value.shape))==math.prod(value.shape)//8
  print('RANK_BUDGET_LAYOUT_OK',rank,count,overhead,flush=True)
  c.get_keys().update(base_emb_dim=600,emb_dim=600,base_num_query_heads=8,base_num_kv_heads=8,num_query_heads=8,num_kv_heads=8,emb_bam_num_head=8,bam_mlp_write_num_heads=0,bam_write_v_bottleneck_dim=16,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16,base_num_decoder_layers=3,num_decoder_layers=3,bam_layer_modes=['local_qk+local_v+local_o']*3,base_mlp_dim=32,mlp_dim=32,mlp_dim_by_block=[32,24,32],vocab_size=128,dtype='float32',weight_dtype='float32')
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes);model=Transformer(c,mesh,quant=None)
  t=jnp.array([[1,4,8,2]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(t);rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   variables=model.init(rng,t,pos,seg,t);assert not any('concat_' in p[-1] for p in flatten_dict(variables))
   params=nn.unbox(variables['params'])
   def loss(p):
    (xent,_,_),stats=model.apply({'params':p},t,pos,seg,t,enable_dropout=False,rngs=rng,mutable=['intermediates'])
    return jnp.mean(xent.astype(jnp.float32)),stats
   (value,stats),grads=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
   assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads))
   gf=flatten_dict(grads)
   for name in ('full_m_read_down','W_lq_c8_up','W_lk_c8_up','W_R_up','general_q_pre_bias','general_k_pre_bias','general_vo_pre_bias','general_v_static_gate'):
    leaves=[v for p,v in gf.items() if name in p];assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,(rank,name)
   metrics={'scalar':{}};train.record_bam_concat_health_metrics(metrics,stats,c);health={k:v for k,v in metrics['scalar'].items() if '/general_' in k}
   assert health and all(np.isfinite(np.asarray(v)).all() for v in health.values())
   for k,v in health.items():
    if k.endswith('/mean') and '_static_gate/' in k:np.testing.assert_allclose(float(v),.99,rtol=2e-6)
  print('RANK_FORWARD_GRAD_HEALTH_OK',rank,len(health),float(value),flush=True)

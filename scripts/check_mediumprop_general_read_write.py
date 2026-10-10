"""Exact budget/layout, parent-equivalent init, gradients and exported health."""
import math,tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,pyconfig,train_compile,train
from layers.models import Transformer
STEM='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMRead'
for suffix,expected,read,write in [('GeneralRead',432111104,True,False),('GeneralWrite',432123776,False,True),('GeneralReadWrite',432130976,True,True)]:
 with tempfile.TemporaryDirectory() as directory:
  Path(directory,'audit').mkdir()
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=STEM+suffix+'TruePile',run_name='audit',
      enable_checkpointing=False,base_output_directory=directory+'/',jax_cache_dir='',log_config=False,
      dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,
      per_device_batch_size=1.,bam_splash_attention=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  args,_,parts,_=train_compile.get_shaped_inputs(mesh,cfg)
  flat=flatten_dict(nn.unbox(args[0].params));count=sum(math.prod(v.shape) for v in flat.values());assert count==expected,(suffix,count,expected)
  ps=flatten_dict(nn.unbox(parts.params));local=sum(math.prod(ps[k].shard_shape(v.shape)) for k,v in flat.items());factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'));overhead=local/(count/factor)-1
  assert overhead<cfg.sharding_tolerance,(suffix,overhead)
  print('GENERAL_TREE_LAYOUT_OK',suffix,count,overhead,flush=True)
  cfg.get_keys().update(base_emb_dim=300,emb_dim=300,base_num_query_heads=4,base_num_kv_heads=4,
      num_query_heads=4,num_kv_heads=4,emb_bam_num_head=4,bam_mlp_write_num_heads=0,
      bam_write_v_bottleneck_dim=16,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16,
      base_num_decoder_layers=3,num_decoder_layers=3,bam_layer_modes=['local_qk+local_v+local_o']*3,
      base_mlp_dim=32,mlp_dim=32,mlp_dim_by_block=[32,24,32],vocab_size=128,
      dtype='float32',weight_dtype='float32')
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
  tokens=jnp.array([[1,4,8,2]],jnp.int32);pos=jnp.arange(4,dtype=jnp.int32)[None];seg=jnp.ones_like(tokens);rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   params=nn.unbox(model.init(rng,tokens,pos,seg,tokens)['params'])
   def forward(p,health=False):
    result=model.apply({'params':p},tokens,pos,seg,tokens,enable_dropout=False,rngs=rng,mutable=['intermediates'] if health else False)
    if health:
     out,metrics=result
     return (out[0] if isinstance(out,tuple) else out),metrics
    return result[0] if isinstance(result,tuple) else result
   initial,intermediates=forward(params,True)
   cfg.get_keys().update(bam_general_column_read=False,bam_general_matrix_write=False)
   parent_params=nn.unbox(model.init(rng,tokens,pos,seg,tokens)['params'])
   parent_flat,child_flat=flatten_dict(parent_params),flatten_dict(params)
   for path,value in parent_flat.items():
    np.testing.assert_array_equal(np.asarray(value),np.asarray(child_flat[path]),err_msg='/'.join(path))
   parent=forward(params)
   np.testing.assert_allclose(np.asarray(initial),np.asarray(parent),rtol=3e-5,atol=3e-6)
   cfg.get_keys().update(bam_general_column_read=read,bam_general_matrix_write=write)
   metrics={'scalar':{}};train.record_bam_concat_health_metrics(metrics,intermediates,cfg)
   exported=metrics['scalar'];gen={k:v for k,v in exported.items() if '/general_' in k}
   assert gen and all(np.isfinite(np.asarray(v)).all() for v in gen.values())
   for k,v in gen.items():
    if k.endswith('/mean') and '_static_gate/' in k:
     expected_open=.01 if 'write_static' in k else .99
     np.testing.assert_allclose(float(v),expected_open,rtol=2e-6,atol=1e-7)
   def loss(p):return jnp.mean(forward(p).astype(jnp.float32)**2)
   value,grad=jax.jit(jax.value_and_grad(loss))(params);assert np.isfinite(float(value));flat=flatten_dict(grad)
   assert all(np.isfinite(np.asarray(v)).all() for v in flat.values())
   names=(['general_q_pre_bias','general_k_pre_bias','general_vo_pre_bias','general_v_static_gate'] if read else [])+(['general_attention_static_address','general_mlp_static_address'] if write else [])
   for name in names:
    leaves=[v for p,v in flat.items() if name in p];assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,(suffix,name)
   if write:
    pf=flatten_dict(params)
    for p in pf:
     if p[-1] in ('general_attention_static_address','general_mlp_static_address'):pf[p]=jnp.full_like(pf[p],.01)
    _,g2=jax.jit(jax.value_and_grad(loss))(unflatten_dict(pf));gf=flatten_dict(g2)
    for name in ('general_attention_static_gate','general_mlp_static_gate'):
     leaves=[v for p,v in gf.items() if name in p];assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,(suffix,name)
   print('GENERAL_INIT_GRAD_HEALTH_OK',suffix,len(gen),float(value),flush=True)

"""Embedding initialization arms: budget/sharding, linearity, lookup and consumed gradients."""
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, pyconfig, train_compile, train
from layers.models import Transformer, EmbeddingBamWrite, EmbeddingBamMatrix
from layers import initializers

PARENT = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
STATIC = 'BamMediumPropFullMGeneralReadStaticEmbeddingTruePile'
LOOKUP = 'BamMediumPropFullMGeneralReadMatrixEmbeddingTruePile'

# Actual-width initialization: retain W_U seed and check the analytic amplitude.
cfg = SimpleNamespace(bam_k=75, bam_v=32, emb_bam_v_bottleneck_dim=256,
    matmul_precision='default', bam_embedding_write_eps=.1, bam_record_concat_health=False,
    bam_sqrt_n_scale=False, bam_write_rms_statistics_dtype='float32',
    normalization_layer_epsilon=1e-6, bam_embedding_init_mode='dynamic',
    bam_embedding_matrix_scale_init=3.74, bam_pallas_core=False)
x = .006 * jax.random.normal(jax.random.key(101), (2, 256, 1200))
old = EmbeddingBamWrite(cfg,16,jnp.float32,jnp.float32,None,initializers.get_init_method('006normal'))
v_old = old.init(jax.random.key(9876),x); m_old = old.apply(v_old,x)
cfg_new = SimpleNamespace(**vars(cfg));cfg_new.bam_embedding_init_mode='static'
new = EmbeddingBamWrite(cfg_new,16,jnp.float32,jnp.float32,None,initializers.get_init_method('006normal'))
v_new = new.init(jax.random.key(9876),x); m_new = new.apply(v_new,x)
p = nn.unbox(v_new['params']); op=nn.unbox(v_old['params'])
np.testing.assert_array_equal(p['W_emb_u']['kernel'],op['W_emb_u']['kernel'])
np.testing.assert_allclose(jnp.mean(p['static_address']**2,-1),1,rtol=2e-6)
np.testing.assert_allclose(p['matrix_scale'],3.74,rtol=1e-6)
np.testing.assert_allclose(new.apply(v_new,2*x),2*m_new,rtol=2e-6,atol=1e-7)
assert set(p)=={'W_emb_u','static_address','matrix_scale'}
rms=lambda a:float(jnp.sqrt(jnp.mean(a.astype(jnp.float32)**2)))
assert .90 < rms(m_new)/rms(m_old) < 1.10,(rms(m_old),rms(m_new))
print('INIT_RMS_OK',rms(m_old),rms(m_new),flush=True)

for name,expected in ((PARENT,432111104),(STATIC,432109616),(LOOKUP,551703504)):
 with tempfile.TemporaryDirectory() as d:
  Path(d,'audit').mkdir()
  c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',enable_checkpointing=False,base_output_directory=d+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,bam_splash_attention=False)
  assert c.bam_general_column_read and not c.bam_general_matrix_write and c.bam_local_full_m_read_share_down
  assert c.bam_local_full_m_read_bottleneck_dim==256 and c.scan_layers and c.DATASET_VARIANT=='truepile4096'
  assert c.bam_k==75 and c.bam_v==32 and c.vocab_size==50432
  if name!=PARENT:assert c.mlp_dim_by_block==[3820,3693,3820]
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  shaped,_,parts,_=train_compile.get_shaped_inputs(mesh,c)
  f=flatten_dict(nn.unbox(shaped[0].params));sp=flatten_dict(nn.unbox(parts.params))
  count=sum(math.prod(v.shape) for v in f.values());assert count==expected,(name,count)
  factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'))
  overhead=sum(math.prod(sp[k].shard_shape(v.shape)) for k,v in f.items())/(count/factor)-1
  assert overhead<c.sharding_tolerance,(name,overhead)
  wd=flatten_dict(train.get_wd_tree(c,nn.unbox(shaped[0].params)))
  for path,v in f.items():
   if 'embedding_bam_matrix' in path:
    assert v.shape==(50432,75,32)
    assert math.prod(sp[path].shard_shape(v.shape))==math.prod(v.shape)//8
    assert wd[path]==c.adam_weight_decay
   if path[-1]=='matrix_scale':assert wd[path]==0
   if path[-1]=='static_address':assert wd[path]==c.adam_weight_decay
  print('BUDGET_SHARDING_WD_OK',name,count,overhead,flush=True)
  c.get_keys().update(base_emb_dim=600,emb_dim=600,base_num_query_heads=8,base_num_kv_heads=8,num_query_heads=8,num_kv_heads=8,emb_bam_num_head=8,bam_mlp_write_num_heads=0,bam_write_v_bottleneck_dim=16,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16,base_num_decoder_layers=3,num_decoder_layers=3,bam_layer_modes=['local_qk+local_v+local_o']*3,base_mlp_dim=32,mlp_dim=32,mlp_dim_by_block=[32,24,32],vocab_size=128,dtype='float32',weight_dtype='float32')
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes);model=Transformer(c,mesh,quant=None)
  t=jnp.array([[1,4,8,2]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(t);rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   variables=model.init(rng,t,pos,seg,t);params=nn.unbox(variables['params'])
   def loss(p):
    (xent,_,_),stats=model.apply({'params':p},t,pos,seg,t,enable_dropout=False,rngs=rng,mutable=['intermediates'])
    return jnp.mean(xent.astype(jnp.float32)),stats
   (value,stats),grads=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
   assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads))
   gf=flatten_dict(grads)
   names = ('W_emb_u','static_address','matrix_scale') if name==STATIC else (('embedding_bam_matrix',) if name==LOOKUP else ('W_emb_u','W_emb_v_down','W_emb_v_up','W_emb_g'))
   for leafname in names:
    leaves=[v for path,v in gf.items() if leafname in path];assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,(name,leafname)
   if name!=PARENT:assert not any(any(z in path for z in ('W_emb_v_down','W_emb_v_up','W_emb_g','emb_gw_b0')) for path in gf)
   if name==LOOKUP:
    assert not any('W_emb_u' in path for path in gf)
    tablegrads=[v for path,v in gf.items() if 'embedding_bam_matrix' in path][0]
    used=np.sum(np.asarray(tablegrads)**2,axis=(1,2))>0
    assert set(np.flatnonzero(used))<=set(np.asarray(t).ravel()) and used.any()
    module=EmbeddingBamMatrix(c);lp=module.init(jax.random.key(3),t)
    got=module.apply(lp,jnp.array([[4,4,8]]))
    np.testing.assert_array_equal(got[:,0],got[:,1])
    np.testing.assert_array_equal(got[:,2],nn.unbox(lp['params'])['embedding'][8][None])
   metrics={'scalar':{}};train.record_bam_concat_health_metrics(metrics,stats,c)
   assert all(np.isfinite(np.asarray(v)).all() for v in metrics['scalar'].values())
   assert ('bam/embedding_write/gate_mean' in metrics['scalar']) == (name==PARENT)
  print('SCANNED_FORWARD_GRAD_HEALTH_OK',name,float(value),flush=True)
print('EMBEDDING_INIT_CHECKS_OK',flush=True)

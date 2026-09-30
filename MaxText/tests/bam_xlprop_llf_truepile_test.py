"""XL matrix-value LLF budget, fetched path, and terminal-L gradient gates."""
import functools
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig,max_utils,train,train_compile
from layers.models import Transformer
from layers import quantizations
EXP='BamXLPropK96EmbedVOnlyQK72LLFTruePile'
with tempfile.TemporaryDirectory() as out:
 Path(out,'audit').mkdir()
 c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=EXP,run_name='audit',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.)
 assert c.DATASET_VARIANT=='truepile4096'
 assert c.scan_layers and c.bam_pair_scan and c.bam_extra_final_local_layer
 assert c.bam_k==96 and c.bam_v==40 and c.bam_abs_v_compression_dim==10
 assert c.bam_write_v_bottleneck_dim==400 and c.emb_bam_v_bottleneck_dim==400
 assert c.bam_local_qk_col_output_dim==72 and c.bam_standard_qk_dim==24 and c.bam_partial_rope_nope_dim==72
 assert c.bam_local_v_replace and c.bam_local_vo_static and c.bam_embedding_write
 assert c.bam_local_vo_shared_read=='local_o' and c.bam_local_vo_independent_gates
 assert c.mlp_dim_by_block==[6294,6294,5654] and c.bam_final_local_mlp_dim==6294
 assert c.record_training_health_metrics and c.bam_record_concat_health
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
 args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
 leaves=flatten_dict(args[0].params)
 count=sum(int(np.prod(x.shape)) for x in leaves.values())
 assert count==1432418340,count
 per={}
 for path,v in leaves.items():
  p='/'.join(path);n=int(np.prod(v.shape))
  for stage in ['local_0','local_1','fetch_2','final_local_layer']:
   if '/'+stage+'/' in p:per[stage]=per.get(stage,0)+n
  if '/value/kernel' in p:assert '/fetch_2/' in p
 for stage in ['local_0','local_1']:
  assert per[stage]//9==44069940
 assert per['fetch_2']//9==44068340
 assert per['final_local_layer']==44069940
 for arm in ['static_v_key','static_o_key']:
  assert any('/local_0/' in '/'.join(p) and p[-1]==arm for p in leaves)
  assert any('/final_local_layer/' in '/'.join(p) and p[-1]==arm for p in leaves)
  assert not any('/fetch_2/' in '/'.join(p) and p[-1]==arm for p in leaves)
 assert any('/fetch_2/' in '/'.join(p) and '/fetch_head_mix/kernel' in '/'.join(p) for p in leaves)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
 assert 'learning/raw_grad_norm' in metrics['scalar']
 assert 'bam/concat/local_q_gate/layer_027/mean' in metrics['scalar']
 assert 'bam/concat/fetched_o_amplitude/layer_026/bam_rms' in metrics['scalar'],list(metrics['scalar'])
 print('FULL_LLF_PARAM_AND_TRAIN_TRACE_OK',count,per,flush=True)
 # Keep exact K96/V40/C10/QK72+RoPE24, exercise LLF plus unscanned finalL.
 c.get_keys().update(base_emb_dim=192,emb_dim=192,num_query_heads=2,num_kv_heads=2,
     base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=4,num_decoder_layers=4,
     base_mlp_dim=128,mlp_dim=128,mlp_dim_by_block=[128,128,96],bam_final_local_mlp_dim=128,
     vocab_size=128,bam_layer_modes=['local_qk+local_v+local_o']*2+['local_qk+full','local_qk+local_v+local_o'],
     bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
 model=Transformer(c,mesh,quantizations.configure_quantization(c))
 tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens)
 call=(tokens,pos,tokens,mask,mask)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
  def loss(p):
   result=model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})
   lp=jax.nn.log_softmax(result.astype(jnp.float32),axis=-1)
   return -jnp.mean(jnp.take_along_axis(lp,tokens[...,None],axis=-1))
  value,grad=jax.jit(jax.value_and_grad(loss))(params)
 assert np.isfinite(float(value))
 assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad))
 flat=flatten_dict(nn.unbox(grad))
 for stage in ['local_0','local_1','fetch_2','final_local_layer']:
  vals=[g for p,g in flat.items() if stage in p]
  assert vals and sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in vals)>0,stage
 value_grad=[g for p,g in flat.items() if 'fetch_2' in p and 'value' in p]
 assert value_grad and sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in value_grad)>0
 print('LLF_FINAL_L_FORWARD_GRAD_OK',float(value),flush=True)

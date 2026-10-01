"""Matrix V in fetched layers: exact budgets, parent regression, gradients and TB tags."""
import functools,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import pyconfig,max_utils,train,train_compile
from layers.models import Transformer
from layers import quantizations
EXP='BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile'
with tempfile.TemporaryDirectory() as out:
 Path(out,'audit').mkdir()
 def config(exp):
  return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=exp,run_name='audit',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.)
 counts={}
 for exp in ('BamMediumPropK75EmbedVOnlyQK57TruePile',EXP):
  c=config(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
  leaves=flatten_dict(args[0].params);counts[exp]=sum(int(np.prod(v.shape)) for v in leaves.values())
  values=['/'.join(p) for p in leaves if '/value/kernel' in '/'.join(p)]
  if exp!=EXP:
   assert counts[exp]==432106784,counts
   assert len(values)==1 and '/fetch_2/' in values[0],values
   continue
  assert c.DATASET_VARIANT=='truepile4096' and c.bam_pair_scan and c.scan_layers
  assert c.mlp_dim_by_block==[3901,3901,3897]
  assert not values,values
  assert counts[exp]==432117152,counts
  for arm in ('static_v_key','W_lv_gate'):
   assert any('/fetch_2/' in '/'.join(p) and arm in p for p in leaves),arm
  assert not any('/fetch_2/' in '/'.join(p) and 'static_o_key' in p for p in leaves)
  assert any('/fetch_2/' in '/'.join(p) and '/W_R/kernel' in '/'.join(p) for p in leaves)
  per={}
  for p,v in leaves.items():
   for stage in ('local_0','local_1','fetch_2'):
    if stage in p:per[stage]=per.get(stage,0)+int(np.prod(v.shape))//6
  print('FULL_BUDGET_OK',counts,per,flush=True)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
  for tag in ('bam/concat/local_v_gate/layer_002/mean',
              'bam/concat/static_v_amplitude/layer_002/bam_rms',
              'bam/concat/local_v_content/layer_002/rms',
              'bam/concat/fetched_o_gate/layer_002/mean'):
   assert tag in metrics['scalar'],tag
  assert 'bam/concat/local_o_gate/layer_002/mean' not in metrics['scalar']
 c=config(EXP);c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,num_decoder_layers=3,base_mlp_dim=128,mlp_dim=128,mlp_dim_by_block=[128,128,128],vocab_size=128,bam_layer_modes=['local_qk+local_v+local_o']*2+['local_qk+local_v+full'],bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c))
 tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens);call=(tokens,pos,tokens,mask,mask)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
  # The production zero-init W_R initially blocks gradients into fetch routing.
  # Awaken that read explicitly to test the live V/fetched-O shared-key path.
  flat_params=flatten_dict(nn.unbox(params))
  for p,v in flat_params.items():
   if 'fetch_2' in p and 'W_R' in p and p[-1]=='kernel':
    flat_params[p]=jax.random.normal(jax.random.key(7),v.shape,v.dtype)*.001
  params=unflatten_dict(flat_params)
  def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
  value,grad=jax.jit(jax.value_and_grad(loss))(params)
 assert np.isfinite(float(value)) and all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad))
 flat=flatten_dict(nn.unbox(grad))
 for name in ('static_v_key','W_R','fetch_head_mix'):
  vals=[g for p,g in flat.items() if 'fetch_2' in p and name in p]
  assert vals and sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in vals)>0,name
 print('SCANNED_MATRIX_V_FETCH_FORWARD_GRAD_OK',float(value),flush=True)

"""Pseudo F: parent parameter regression, local-only reads and vector-V gradients."""
import functools,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig,max_utils,train,train_compile
from layers.models import Transformer
from layers import quantizations
EXP='BamMediumPropK75EmbedVOnlyQK57PseudoFTruePile'
with tempfile.TemporaryDirectory() as out:
 Path(out,'audit').mkdir()
 def config(exp):
  return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=exp,run_name='audit',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.)
 for exp,count in [('BamMediumPropK75EmbedVOnlyQK57TruePile',432106784),('BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile',432091328),('BamMediumPropK75EmbedVOnlyQK57LLFMatrixVTruePile',432095552),(EXP,432102560)]:
  c=config(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
  leaves=flatten_dict(args[0].params);actual=sum(int(np.prod(v.shape)) for v in leaves.values())
  assert actual==count,(exp,actual,count)
  if exp!=EXP:continue
  assert c.DATASET_VARIANT=='truepile4096' and c.bam_pair_scan and c.mlp_dim_by_block==[3901,3901,3507]
  values=[p for p in leaves if '/value/kernel' in '/'.join(p)]
  assert len(values)==1 and 'fetch_2' in values[0],values
  assert not any('fetch_head_mix' in p for p in leaves)
  for stage in ('local_0','local_1','fetch_2'):
   for name in ('static_o_key','W_R'):
    assert any(stage in p and name in p for p in leaves),(stage,name)
   for name in ('static_v_key','W_lv_gate'):
    assert any(stage in p and name in p for p in leaves)==(stage!='fetch_2'),(stage,name)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
  for tag in ('bam/concat/local_o_gate/layer_002/mean','bam/concat/static_o_amplitude/layer_002/bam_rms'):
   assert tag in metrics['scalar'],tag
  assert 'bam/concat/fetched_o_gate/layer_002/mean' not in metrics['scalar']
  assert 'bam/concat/local_v_gate/layer_002/mean' not in metrics['scalar']
  print('FULL_BUDGET_AND_LOCAL_ONLY_HEALTH_OK',actual,flush=True)
 c=config(EXP);c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,num_decoder_layers=3,base_mlp_dim=128,mlp_dim=128,mlp_dim_by_block=[128,128,128],vocab_size=128,bam_layer_modes=['local_qk+local_v+local_o']*2+['local_qk+local_o'],bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c))
 tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens);call=(tokens,pos,tokens,mask,mask)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
  def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
  value,grad=jax.jit(jax.value_and_grad(loss))(params)
 assert np.isfinite(float(value)) and all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad))
 flat=flatten_dict(nn.unbox(grad))
 for name in ('value','static_o_key','W_R'):
  vals=[g for p,g in flat.items() if 'fetch_2' in p and name in p]
  assert vals and sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in vals)>0,name
 print('PSEUDO_F_FORWARD_GRAD_OK',float(value),flush=True)

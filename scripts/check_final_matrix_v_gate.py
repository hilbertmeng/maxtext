"""Full budget/layout plus parent-matched final-M gate initialization and gradients."""
import copy
import math
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, pyconfig, train_compile, train
from layers.models import Transformer

PARENT='BamMediumPropFullMGeneralReadFinalMatrixResidualTruePile'
GATED='BamMediumPropFullMGeneralReadFinalMatrixVGateTruePile'
for name,count_expected in ((PARENT,432118304),(GATED,432156736)):
 with tempfile.TemporaryDirectory() as d:
  Path(d,'audit').mkdir()
  c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',enable_checkpointing=False,base_output_directory=d+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,bam_splash_attention=False)
  assert c.mlp_dim_by_block==[3769,3641,3769] and c.bam_final_matrix_residual
  assert c.bam_static_read_gate_init==.99 and not c.bam_static_read_gate_sigmoid
  assert c.tensorboard_gcs_rotate_period==500 and c.scan_layers and c.DATASET_VARIANT=='truepile4096'
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  shaped,_,parts,_=train_compile.get_shaped_inputs(mesh,c)
  f=flatten_dict(nn.unbox(shaped[0].params));sp=flatten_dict(nn.unbox(parts.params))
  count=sum(math.prod(v.shape) for v in f.values());assert count==count_expected,(name,count)
  factor=math.prod(mesh.shape[a] for a in ('fsdp','fsdp_transpose','sequence','tensor','tensor_transpose','tensor_sequence','stage','expert'))
  overhead=sum(math.prod(sp[k].shard_shape(v.shape)) for k,v in f.items())/(count/factor)-1
  assert overhead<c.sharding_tolerance
  wd=flatten_dict(train.get_wd_tree(c,nn.unbox(shaped[0].params)))
  if name==GATED:
   kernel_path=next(k for k in f if k[-2:]==('final_matrix_v_gate','kernel'))
   bias_path=next(k for k in f if k[-1]=='final_matrix_v_gate_bias')
   assert f[kernel_path].shape==(1200,32)
   assert f[bias_path].shape==(32,)
   assert wd[kernel_path]==c.adam_weight_decay
   assert wd[bias_path]==0
  print('GATE_BUDGET_SHARDING_WD_OK',name,count,overhead,flush=True)

# Use the gated configuration from above, reducing only dimensions irrelevant to the gate.
c.get_keys().update(base_emb_dim=600,emb_dim=600,base_num_query_heads=8,base_num_kv_heads=8,num_query_heads=8,num_kv_heads=8,emb_bam_num_head=8,bam_mlp_write_num_heads=0,bam_write_v_bottleneck_dim=16,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16,base_num_decoder_layers=3,num_decoder_layers=3,bam_layer_modes=['local_qk+local_v+local_o']*3,base_mlp_dim=32,mlp_dim=32,mlp_dim_by_block=[32,24,32],vocab_size=128,dtype='float32',weight_dtype='float32')
mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
model=Transformer(c,mesh,quant=None)
t=jnp.array([[1,4,8,2]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(t)
rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
 p=nn.unbox(model.init(rng,t,pos,seg,t)['params'])
 assert np.count_nonzero(np.asarray(p['decoder']['final_matrix_v_gate']['kernel']))==0
 np.testing.assert_allclose(p['decoder']['final_matrix_v_gate_bias'],math.log(9),rtol=2e-6)
 np.testing.assert_allclose(1.1*jax.nn.sigmoid(p['decoder']['final_matrix_v_gate_bias'])/.99,1,rtol=2e-6)
 def objective(params):return jnp.mean(model.apply({'params':params},t,pos,seg,t,enable_dropout=False,rngs=rng)[0])
 on_value,on_grads=jax.jit(jax.value_and_grad(objective))(p)
 assert float(jnp.linalg.norm(on_grads['decoder']['final_matrix_readout']['kernel']))>0
 for leaf in ('final_matrix_v_gate','final_matrix_v_gate_bias'):
  assert all(np.count_nonzero(np.asarray(x))==0 for x in jax.tree.leaves(on_grads['decoder'][leaf]))
 c.get_keys()['bam_final_matrix_v_gate']=False
 off=nn.unbox(model.init(rng,t,pos,seg,t)['params'])
 f_on=flatten_dict(p);f_off=flatten_dict(off)
 for path,v in f_off.items():np.testing.assert_array_equal(v,f_on[path])
 off_value,off_grads=jax.jit(jax.value_and_grad(lambda q:objective(q)))(off)
 np.testing.assert_array_equal(on_value,off_value)
 fg=flatten_dict(on_grads)
 for path,v in flatten_dict(off_grads).items():np.testing.assert_allclose(v,fg[path],rtol=2e-5,atol=1e-7)
 print('GATE_PARENT_INITIAL_PARAMS_OUTPUT_GRADS_MATCH',float(on_value),flush=True)
 # After the readout learns, the new gate must receive consumed gradients.
 c.get_keys()['bam_final_matrix_v_gate']=True
 active=copy.deepcopy(p)
 active['decoder']['final_matrix_readout']['kernel']=.001*jax.random.normal(jax.random.key(25),(75,32,600))
 value,grads=jax.jit(jax.value_and_grad(lambda q:objective(q)))(active)
 assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
 assert float(jnp.linalg.norm(grads['decoder']['final_matrix_v_gate']['kernel']))>0
 assert float(jnp.linalg.norm(grads['decoder']['final_matrix_v_gate_bias']))>0
 active['decoder']['final_matrix_v_gate']['kernel']=.1*jax.random.normal(jax.random.key(26),(600,32))
 captured={}
 def inspect(next_fun,args,kwargs,context):
  out=next_fun(*args,**kwargs)
  if context.method_name=='__call__':
   if context.module.name=='final_matrix_v_gate':captured['h']=args[0];captured['logits']=out
   if context.module.name=='final_matrix_readout':captured['gated_m']=args[0]
  return out
 with nn.intercept_methods(inspect):conditional=model.apply({'params':active},t,pos,seg,t,enable_dropout=False,rngs=rng)[0]
 c.get_keys()['bam_final_matrix_v_gate']=False
 off_active=copy.deepcopy(active);del off_active['decoder']['final_matrix_v_gate'];del off_active['decoder']['final_matrix_v_gate_bias']
 raw={}
 def inspect_raw(next_fun,args,kwargs,context):
  out=next_fun(*args,**kwargs)
  if context.method_name=='__call__' and context.module.name=='final_matrix_readout':raw['m']=args[0]
  return out
 with nn.intercept_methods(inspect_raw):unconditional=model.apply({'params':off_active},t,pos,seg,t,enable_dropout=False,rngs=rng)[0]
 gate=1.1*jax.nn.sigmoid(captured['logits']+active['decoder']['final_matrix_v_gate_bias'])/.99
 assert gate.shape==(1,4,32) and float(jnp.std(gate))>1e-5
 np.testing.assert_allclose(captured['gated_m'],raw['m']*gate[...,None,:],rtol=2e-6,atol=1e-7)
 assert not np.allclose(conditional,unconditional,rtol=1e-7,atol=1e-7)
 print('GATE_DYNAMIC_COLUMN_BROADCAST_AND_CONSUMED_GRADS_OK',float(value),flush=True)
print('FINAL_MATRIX_V_GATE_CHECKS_OK',flush=True)

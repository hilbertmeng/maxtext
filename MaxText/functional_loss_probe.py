"""Read-only per-token loss and smooth/native BAM parameter-family interventions."""
import argparse,ast,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import checkpointing,max_utils
from layers.models import Transformer
from cross_scale_probe import cohort,config_for,hash_sequence,variant

def changed_params(params,name):
 if name.startswith('no_') and name!='no_M_shift':return variant(params,name)
 flat=flatten_dict(params);out={};matched=0
 for key,value in flat.items():
  leaf=key[-1];new=value
  if name=='no_M_shift' and leaf=='M_shift_gate_b0':new=jnp.full_like(value,-80);matched+=1
  elif name.startswith('shift_bias_') and leaf=='M_shift_gate_b0':new=value+float(name.removeprefix('shift_bias_'));matched+=1
  elif name.startswith('qk_read_bias_') and leaf in ['W_lq_gate_b0','W_lk_gate_b0']:new=value+float(name.removeprefix('qk_read_bias_'));matched+=1
  elif name.startswith('mlp_write_bias_') and leaf=='mlp_write_gate_bias':new=value+float(name.removeprefix('mlp_write_bias_'));matched+=1
  out[key]=new
 if matched==0:raise ValueError(('No matched parameters',name))
 return unflatten_dict(out),matched

def main():
 p=argparse.ArgumentParser()
 for key in ['experiment','checkpoint','training-config','label','output','gcs-output']:p.add_argument('--'+key,required=True)
 p.add_argument('--length',type=int,default=2048);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--sequences',type=int,default=32)
 p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096/pile.tfrecord.b0')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time();Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True);cfg=config_for(a)
 expected={}
 for line in Path(a.training_config).read_text().splitlines():
  key,sep,val=line.partition(': ')
  if sep:expected[key]=val
 critical={'qk_norm','normalization_layer_epsilon','normalization_layer_type','decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_dim_by_block','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','dtype','weight_dtype','attention','dense_conn','fused_mlp'}
 for key,raw in expected.items():
  if key not in cfg.get_keys() or not(key in critical or key.startswith(('bam_','mudd_','pythia_'))):continue
  try:want=ast.literal_eval(raw)
  except (ValueError,SyntaxError):want=raw
  value=getattr(cfg,key)
  if isinstance(value,type):value=getattr(value,'__name__',str(value))
  if isinstance(value,tuple):value=list(value)
  if isinstance(want,tuple):want=list(want)
  if value!=want:raise ValueError(('Model config mismatch',key,want,str(value)))
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.key(a.seed)
 with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
  native=max_utils.unbox_logicallypartioned
  max_utils.unbox_logicallypartioned=lambda tree:jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,tree,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  try:abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
  finally:max_utils.unbox_logicallypartioned=native
  params=checkpointing.load_params_from_path(a.checkpoint,abstract.params)
 data=cohort(a.cohort,a.cohort_source,a.seed,a.sequences)
 # Same label shift, mask and packing as the registered native cohort probe.
 inputs=data['tokens'][:a.sequences,:a.length];targets=data['tokens'][:a.sequences,1:a.length+1]
 mask=(data['labels'][:a.sequences,1:a.length+1]>=0).astype(np.int32)
 batches=[]
 for i in range(0,a.sequences,a.batch_size):
  shape=inputs[i:i+a.batch_size].shape
  batches.append(dict(inputs=jnp.asarray(inputs[i:i+a.batch_size]),targets=jnp.asarray(targets[i:i+a.batch_size]),mask=jnp.asarray(mask[i:i+a.batch_size]),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(a.length),shape)))
 def forward(weights,b):
  value=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  return value[0]
 fn=jax.jit(forward)
 def evaluate(weights):
  values=[]
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   for b in batches:values.append(np.asarray(jax.device_get(fn(weights,b)),np.float32))
  z=np.concatenate(values);assert z.shape==(a.sequences,a.length)
  assert np.isfinite(z).all() and z.min()>=0
  return z
 baseline=evaluate(params)
 np.savez_compressed(out/'token_losses.npz',loss=baseline,inputs=inputs,targets=targets,mask=mask)
 meta=dict(experiment=a.experiment,checkpoint=a.checkpoint,label=a.label,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sequence_hashes=[hash_sequence(x) for x in data['tokens'][:a.sequences,:a.length+1]],native_losses=(np.sum(baseline*mask,axis=1)/mask.sum(1)).tolist(),parameter_count=sum(v.size for v in jax.tree.leaves(params)),config={key:getattr(cfg,key) for key in ['num_decoder_layers','emb_dim','num_query_heads','head_dim','max_target_length','qk_norm','mlp_activations']})
 small={}
 for key,v in flatten_dict(params).items():
  if key[-1].endswith(('_gate_b0','gw_b0')) or key[-1]=='mlp_write_gate_bias':
   z=np.asarray(jax.device_get(v),np.float32)
   small['/'.join(key)]=dict(shape=list(z.shape),mean=float(z.mean()),rms=float(np.sqrt(np.mean(z*z))),min=float(z.min()),max=float(z.max()))
 meta['gate_parameter_statistics']=small
 effects={}
 if cfg.bam_enabled:
  for name in ['no_embedding_M_write','no_attention_M_write','no_mlp_M_write','no_M_read','no_M_shift','shift_bias_-0.5','shift_bias_0.5','qk_read_bias_-0.5','qk_read_bias_0.5','mlp_write_bias_-0.5','mlp_write_bias_0.5']:
   weights,n=changed_params(params,name);loss=evaluate(weights);delta=np.sum((loss-baseline)*mask,axis=1)/mask.sum(1)
   effects[name]=dict(mean_delta=float(delta.mean()),paired_delta=delta.tolist(),changed_tensors=n)
   np.savez_compressed(out/(name+'.npz'),loss=loss)
   print('FUNCTIONAL_VARIANT',a.label,name,float(delta.mean()),flush=True)
 meta['effects']=effects;meta['elapsed_seconds']=time.time()-start
 (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
 subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,sequences=a.sequences,variants=len(effects)))+'\n')
 subprocess.run(['gsutil','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('FUNCTIONAL_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

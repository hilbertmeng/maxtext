"""Read-only Mudd router statistics and cross-text/mean-position replay interventions."""
import argparse,ast,hashlib,json,subprocess,time
from pathlib import Path
import re
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import checkpointing,max_utils
from layers.models import Transformer
from cross_scale_probe import cohort,config_for,hash_sequence,variant,save_capture

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
 if cfg.bam_enabled or not cfg.dense_conn:raise ValueError('This probe requires the original full-history Mudd model')
 import mudd_router_capture as route
 route.install();route.ACTIVE=False
 baseline=evaluate(params)
 route.ACTIVE=True
 def capture(weights,b):
  value,raw=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  return value[0],raw
 cap_fn=jax.jit(capture);caches=[];histories=[];capture_delta=0.
 for i,b in enumerate(batches):
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(cap_fn(params,b))
  capture_delta=max(capture_delta,float(np.max(np.abs(np.asarray(loss).mean(1)-baseline[i*a.batch_size:(i+1)*a.batch_size].mean(1)))))
  if capture_delta>.005:raise ValueError(('Router observer changed loss',capture_delta))
  weights={};norms={}
  for path,v in flatten_dict(raw['intermediates']).items():
   if not path[-1].startswith('stream_router_'):continue
   while isinstance(v,(tuple,list)) and len(v)==1:v=v[0]
   key='/'.join(path[:-1])
   if path[-1]=='stream_router_weights':weights[key]=np.asarray(v,np.float32)
   if path[-1]=='stream_router_history_rms':norms[key]=np.asarray(v,np.float32)
  if set(weights)!=set(norms):raise ValueError('Missing paired router histories')
  caches.append(weights);histories.append(norms)
 route.ACTIVE=False
 if len(caches[0])!=cfg.num_decoder_layers:raise ValueError(('Unexpected number of routers',len(caches[0])))
 half=a.sequences//2;half_batches=half//a.batch_size;pos=route.positions(a.length);paths=sorted(caches[0]);means={};stats={};samples={}
 for path in paths:
  w=np.concatenate([d[path] for d in caches]);h=np.concatenate([d[path] for d in histories]);means[path]=w[:half].mean(0)
  selected=w[:,pos];amplitude=np.abs(selected)*h[:,:,None,:];total=amplitude.sum(-1,keepdims=True);share=amplitude/np.maximum(total,1e-30)
  entropy=-np.sum(share*np.log(np.maximum(share,1e-30)),-1)/np.log(w.shape[-1]);last=share[...,-1];embedding=share[...,0]
  within_position=(w-w.mean(0,keepdims=True))**2
  dependence=np.sqrt(within_position.mean((0,1,3))/np.maximum((w*w).mean((0,1,3)),1e-30))
  layer=re.findall(r'layers_(\d+)',path);depth=int(layer[-1])+1 if layer else cfg.num_decoder_layers+1
  stats[path]=dict(next_layer=depth,final=w.shape[-2]==1,history=w.shape[-1],channels=w.shape[-2],input_variation_rms_over_weight_rms=dependence.tolist(),sampled_mean_amplitude_share=share.mean((0,1)).tolist(),sampled_last_share=last.mean((0,1)).tolist(),sampled_embedding_share=embedding.mean((0,1)).tolist(),sampled_history_entropy=entropy.mean((0,1)).tolist(),negative_weight_fraction=(w<0).mean((0,1,3)).tolist())
  samples[path.replace('/','__')+'__weights']=selected;samples[path.replace('/','__')+'__history_rms']=h
 save_capture(out/'sampled_router_weights.npz',samples);save_capture(out/'first16_position_means.npz',{k.replace('/','__'):v for k,v in means.items()})
 def controlled(weights,b,values,mask):
  route.VALUES=values;route.MASK=mask
  try:return forward(weights,b)
  finally:route.VALUES=None;route.MASK=None
 control_fn=jax.jit(controlled)
 masks={'all':[1,1,1,1,1],'q':[1,0,0,0,0],'k':[0,1,0,0,0],'v':[0,0,1,0,0],'m':[0,0,0,1,0],'final':[0,0,0,0,1],'qkv':[1,1,1,0,0]}
 results={};nulls={};replay_delta=0.
 for name,route_mask in masks.items():
  ident=[];meanloss=[];shuffle=[]
  for i,b in enumerate(batches[half_batches:]):
   original=caches[i+half_batches];foreign=caches[i];average={k:np.broadcast_to(v,(a.batch_size,)+v.shape) for k,v in means.items()}
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    orig=np.asarray(jax.device_get(control_fn(params,b,original,jnp.array(route_mask,dtype=bool))),np.float32)
    avg=np.asarray(jax.device_get(control_fn(params,b,average,jnp.array(route_mask,dtype=bool))),np.float32)
    other=np.asarray(jax.device_get(control_fn(params,b,foreign,jnp.array(route_mask,dtype=bool))),np.float32)
   ident.append(orig);meanloss.append(avg);shuffle.append(other)
  ident,meanloss,shuffle=map(np.concatenate,[ident,meanloss,shuffle]);nulls[name]=ident
  replay_delta=max(replay_delta,float(np.max(np.abs(ident.mean(1)-baseline[half:].mean(1)))))
  if replay_delta>.005:raise ValueError(('Same-input router replay changes native loss',name,replay_delta))
  for mode,loss in [('mean_position',meanloss),('cross_text',shuffle)]:
   delta=loss-ident;key=mode+'_'+name;results[key]=dict(mean_delta=float(delta.mean()),paired_delta=delta.mean(1).tolist());np.savez_compressed(out/(key+'.npz'),loss=loss,neutral=ident)
  print('ROUTER_REPLAY',a.label,name,results['mean_position_'+name]['mean_delta'],results['cross_text_'+name]['mean_delta'],flush=True)
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=sum(v.size for v in jax.tree.leaves(params)),sequence_hashes=[hash_sequence(x) for x in data['tokens'][:a.sequences,:a.length+1]],native_losses=baseline.mean(1).tolist(),sequences=a.sequences,selection_sequences=half,validation_sequences=half,positions=pos.tolist(),routers=stats,effects=results,observer_native_max_loss_delta=capture_delta,same_input_replay_native_max_loss_delta=replay_delta,elapsed_seconds=time.time()-start,definition='Native signed Mudd q/k/v/m and final history coefficients. First16 texts supply position-dependent means or paired foreign coefficients; last16 evaluate. Each intervention uses a mask-matched same-input coefficient replay baseline. No training.')
 np.savez_compressed(out/'token_losses.npz',loss=baseline,inputs=inputs,targets=targets,mask=mask)
 (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
 subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,sequences=a.sequences))+'\n');subprocess.run(['gsutil','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('ROUTER_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

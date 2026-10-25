"""Read-only matched attention contrast counterfactuals on new texts; no optimizer or updates."""
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

def main():
 p=argparse.ArgumentParser()
 for key in ['experiment','checkpoint','training-config','label','output','gcs-output']:p.add_argument('--'+key,required=True)
 p.add_argument('--length',type=int,default=2048);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--sequences',type=int,default=64)
 p.add_argument('--reference-json',default='/tmp/bam_stream_probe/attention_contrast_reference.json');p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/attention_contrast_cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/attention_contrast/cohort64.npz')
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
 assert a.sequences==64 and a.length==2048
 def fetch(path,source):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():subprocess.run(['gsutil','-q','cp',source,str(path)],check=True)
  return path
 cohort_path=fetch(a.cohort,a.cohort_source)
 ref_path=fetch(a.reference_json,a.cohort_source.rsplit('/',1)[0]+'/reference8.json')
 meta_path=fetch('/tmp/bam_stream_probe/contrast_cohort64.json',a.cohort_source.rsplit('/',1)[0]+'/cohort64.json')
 old_path=fetch('/tmp/bam_stream_probe/contrast_old32.npz','gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
 ref=json.loads(ref_path.read_text());cohort_meta=json.loads(meta_path.read_text());key=a.label.removeprefix('attention_contrast_');expected=ref['models'][key]
 assert expected['experiment']==a.experiment and expected['checkpoint']==a.checkpoint
 assert sum(v.size for v in jax.tree.leaves(params))==expected['parameter_count']
 with np.load(cohort_path) as z:data={k:z[k] for k in z.files}
 with np.load(old_path) as z:old={k:z[k] for k in z.files}
 hashes=[hash_sequence(x[:2049]) for x in data['tokens']];old_hashes=[hash_sequence(x[:2049]) for x in old['tokens']]
 assert hashes==cohort_meta['sequence_hashes'] and old_hashes==expected['old32_sequence_hashes']
 assert len(hashes)==64 and len(set(hashes))==64 and not(set(hashes)&set(old_hashes))
 assert np.all(data['labels'][:,1:2049]>=0)
 def make_batch(tokens,labels):
  inputs=tokens[:,:2048];shape=inputs.shape
  return dict(inputs=jnp.asarray(inputs),targets=jnp.asarray(tokens[:,1:2049]),mask=jnp.asarray((labels[:,1:2049]>=0).astype(np.int32)),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape))
 batches=[make_batch(data['tokens'][i:i+a.batch_size],data['labels'][i:i+a.batch_size]) for i in range(0,64,a.batch_size)]
 old_batch=make_batch(old['tokens'][:a.batch_size],old['labels'][:a.batch_size])
 import precise_ce
 precise_ce.install()
 def forward(weights,b):return model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
 native_fn=jax.jit(forward)
 with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):calibration=np.asarray(jax.device_get(native_fn(params,old_batch)),np.float32)
 calibration_error=float(np.max(np.abs(calibration.mean(1)-np.asarray(expected['old32_native_losses'][:a.batch_size]))))
 if calibration_error>.005:raise ValueError(('Oldcohort frozen model calibration failed',calibration_error))
 native=[]
 for b in batches:
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):native.append(np.asarray(jax.device_get(native_fn(params,b)),np.float32))
 native=np.concatenate(native);assert native.shape==(64,2048) and np.isfinite(native).all()
 np.savez_compressed(out/'token_losses.npz',loss=native,inputs=data['tokens'][:,:2048],targets=data['tokens'][:,1:2049],source_shards=data['source_shards'])
 np.savez_compressed(out/'old_calibration.npz',loss=calibration)
 import attention_contrast_capture as hook
 hook.install();L=int(cfg.num_decoder_layers);zero=np.zeros((L,4),np.float32)
 def observed(weights,b,controls,geometry=False):
  hook.CONTROLS=controls;hook.CAPTURE=True;hook.GEOMETRY=geometry;hook.TOKENS=b['inputs']
  try:
   value,raw=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  finally:hook.CONTROLS=None;hook.CAPTURE=False;hook.GEOMETRY=False;hook.TOKENS=None
  flat={}
  for path,entry in flatten_dict(raw['intermediates']).items():
   if not path[-1].startswith('stream_contrast_'):continue
   assert isinstance(entry,(tuple,list)) and len(entry)==1
   flat['__'.join(path)]=entry[0]
  return value[0],flat
 cf=jax.jit(observed,static_argnames=('geometry',))
 def evaluate(controls,geometry=False):
  values=[];arrays={};indices={}
  for i,b in enumerate(batches):
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,capture=jax.device_get(cf(params,b,jnp.asarray(controls),geometry=geometry))
   values.append(np.asarray(loss,np.float32));assert capture
   for name,value in capture.items():
    value=np.asarray(value)
    if name.endswith('stream_contrast_layer_index'):
     if name in indices:np.testing.assert_array_equal(indices[name],value)
     indices[name]=value
    else:arrays.setdefault(name,[]).append(value)
   print('CONTRAST_BATCH',a.label,geometry,i+1,flush=True)
  result=np.concatenate(values);assert result.shape==(64,2048) and np.isfinite(result).all()
  arrays={name:np.concatenate(values,axis=indices[name.rsplit('__',1)[0]+'__stream_contrast_layer_index'].ndim) for name,values in arrays.items()}
  arrays.update(indices)
  assert all(np.isfinite(v).all() for v in arrays.values())
  return result,arrays
 neutral,exposure=evaluate(zero);zero_error=float(np.max(np.abs(neutral.mean(1)-native.mean(1))))
 if zero_error>.005:raise ValueError(('Zero operator changed native CE',zero_error))
 np.savez_compressed(out/'neutral.npz',loss=neutral,controls=zero,**exposure)
 geometry_loss,geometry=evaluate(zero,True);geometry_error=float(np.max(np.abs(geometry_loss.mean(1)-neutral.mean(1))))
 if geometry_error>.005:raise ValueError(('Geometry observer changed CE',geometry_error))
 checks={name:0. for name in ['raw_AV_relative_RMS','projected_output_relative_RMS','directO_sum_relative_RMS']}
 for name,errors in geometry.items():
  if not name.endswith('stream_contrast_errors'):continue
  gram_=geometry[name.removesuffix('errors')+'projected_gram']
  for label,num,den in [('raw_AV_relative_RMS',errors[...,0],errors[...,1]),('projected_output_relative_RMS',errors[...,2],gram_[...,-1,-1]),('directO_sum_relative_RMS',errors[...,3],errors[...,4])]:
   ratio=np.sqrt(np.maximum(num,0)/np.maximum(den,1e-30));checks[label]=max(checks[label],float(ratio.max()))
 if any(v>.02 for v in checks.values()):raise ValueError(('Native geometry reconstruction failed',checks))
 np.savez_compressed(out/'geometry.npz',loss=geometry_loss,**geometry)
 width=L//4;scopes={'all_layers':list(range(L)),'first_quarter':list(range(width)),'last_quarter':list(range(L-width,L))};variants=[]
 for scope,layers in scopes.items():
  for mode,name in enumerate(hook.MODE_NAMES):
   controls=zero.copy();controls[layers,mode]=1;loss,stats=evaluate(controls);filename=scope+'_'+name+'.npz'
   np.testing.assert_array_equal(loss[:,:64],neutral[:,:64])
   np.savez_compressed(out/filename,loss=loss,controls=controls,**stats);variants.append(dict(scope=scope,mode=name,mode_index=mode,file=filename,layer_indices_zero_based=layers))
   subprocess.run(['gsutil','-q','cp',str(out/filename),a.gcs_output+'/'+filename],check=True)
   print('CONTRAST_VARIANT_COMPLETE',a.label,scope,name,float((loss-neutral).mean()),flush=True)
 repeated,_=evaluate(zero);np.testing.assert_array_equal(repeated,neutral)
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=expected['parameter_count'],sequence_hashes=hashes,source_shards=data['source_shards'].tolist(),sequences=64,length=2048,batch_size=a.batch_size,old_cohort_calibration_max_CE_error=calibration_error,zero_native_max_CE_error=zero_error,geometry_zero_max_CE_error=geometry_error,zero_repeat_exact=True,query_start=64,geometry_queries=hook.q_positions(2048).tolist(),vector_names=hook.VECTOR_NAMES,mode_names=hook.MODE_NAMES,exposure_names=hook.EXPOSURE_NAMES,reconstruction_checks=checks,config=dict(num_decoder_layers=L,num_query_heads=cfg.num_query_heads,head_dim=cfg.head_dim,emb_dim=cfg.emb_dim),scopes=scopes,variants=variants,elapsed_seconds=time.time()-start,definition='Frozen-checkpoint attention operators on new64texts. Native zero retained; reference mode replaces AV with FP32 reconstruction, as matched numerical control for no-self, uniform and sink-preserving eligible-key uniform. Onlyq>=64. Downstream/nativeMwrites respond, weights unchanged. Geometry uses effective nativeW_o, FP32 statistical projection, no bias; not an additive loss attribution. Uniform still carries context. Newtexts are same training corpus, not training holdout.')
 assert len(variants)==12
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,variants=12))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('ATTENTION_CONTRAST_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

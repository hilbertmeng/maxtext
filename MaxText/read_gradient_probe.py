"""Read-only gradients of shared BAM reader paths; no optimizer or updates."""
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
 p.add_argument('--length',type=int,default=2048);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--sequences',type=int,default=32)
 p.add_argument('--reference-json',default='/tmp/bam_stream_probe/read_gradient_reference.json');p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/read_gradient_cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
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
 cohort_path=Path(a.cohort);cohort_path.parent.mkdir(parents=True,exist_ok=True)
 if not cohort_path.exists():subprocess.run(['gsutil','-q','cp',a.cohort_source,str(cohort_path)],check=True)
 ref_path=Path(a.reference_json)
 if not ref_path.exists():subprocess.run(['gsutil','-q','cp',a.cohort_source.rsplit('/',1)[0]+'/reference.json',str(ref_path)],check=True)
 reference=json.loads(ref_path.read_text())
 with np.load(cohort_path) as z:data={k:z[k] for k in z.files}
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
 import precise_ce
 precise_ce.install()
 fn=jax.jit(forward)
 def evaluate(weights):
  values=[]
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   for b in batches:values.append(np.asarray(jax.device_get(fn(weights,b)),np.float32))
  result=np.concatenate(values);assert result.shape==(a.sequences,a.length) and np.isfinite(result).all()
  return result
 key=a.label.removeprefix('read_gradient_');expected=reference['models'][key]
 assert expected['experiment']==a.experiment and expected['checkpoint']==a.checkpoint
 assert np.all(data['objective_all']==1) and int(data['objective_midcontent'].sum())==reference['objective_token_counts']['midcontent']
 assert hashlib.sha256(data['objective_midcontent'].tobytes()).hexdigest()==reference['mask_sha256']
 assert cfg.bam_enabled and cfg.bam_local_full_m_read_share_down
 if cfg.get_keys().get('bam_share_read_attention_write_down',False):raise ValueError('This diagnostic needs independent attention write/read downs')
 assert sum(v.size for v in jax.tree.leaves(params))==expected['parameter_count']
 hashes=[hash_sequence(x) for x in data['tokens'][:a.sequences,:a.length+1]];assert hashes==reference['sequence_hashes']
 baseline=evaluate(params);native_error=float(np.max(np.abs(baseline.mean(1)-expected['native_losses'])))
 if native_error>.005:raise ValueError(('Native CE differs from reference',native_error))
 np.savez_compressed(out/'token_losses.npz',loss=baseline,inputs=inputs,targets=targets,mask=mask)
 import prewo_observer as prewo
 prewo.install();prewo.ACTIVE=True
 def observed(weights,b):
  value,raw=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  return value[0],raw
 observed_fn=jax.jit(observed);prewo_arrays={};prewo_drift=0.
 for i,b in enumerate(batches):
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):value,raw=jax.device_get(observed_fn(params,b))
  prewo_drift=max(prewo_drift,float(np.max(np.abs(np.asarray(value).mean(1)-baseline[i*a.batch_size:(i+1)*a.batch_size].mean(1)))))
  if prewo_drift>.005:raise ValueError(('Readout observer changed CE',prewo_drift))
  for path,entries in flatten_dict(raw['intermediates']).items():
   if not path[-1].startswith('stream_prewo_'):continue
   if not isinstance(entries,(tuple,list)):entries=[entries]
   arr=np.maximum.reduce([np.asarray(v) for v in entries]) if path[-1].endswith('reconstruction') else np.sum([np.asarray(v,dtype=np.float64) for v in entries],axis=0)
   prewo_arrays.setdefault('__'.join(path),[]).append(arr)
 prewo.ACTIVE=False
 if not prewo_arrays:raise ValueError('No native readout observations')
 prewo_arrays={k:np.concatenate(values,axis=1 if 'layers' in k.split('__') else 0) for k,values in prewo_arrays.items()}
 prewo_closure_max_relative_rms=0.
 for name,values in prewo_arrays.items():
  if not np.isfinite(values).all():raise ValueError(('Nonfinite native readout moments',name))
  if name.endswith('stream_prewo_O_closure'):
   source=prewo_arrays[name.removesuffix('stream_prewo_O_closure')+'stream_prewo_O_sources']
   denominator=np.maximum(source[...,3],np.max(source[...,3])*1e-12+1e-30)
   error=float(np.sqrt(np.maximum(values,0)/denominator).max())
   prewo_closure_max_relative_rms=max(prewo_closure_max_relative_rms,error)
 if prewo_closure_max_relative_rms>.02:raise ValueError(('Dynamic/static O closure failed',prewo_closure_max_relative_rms))
 np.savez_compressed(out/'prewo_observations.npz',**prewo_arrays)
 subprocess.run(['gsutil','-q','cp',str(out/'prewo_observations.npz'),a.gcs_output+'/prewo_observations.npz'],check=True)
 print('NATIVE_PREWO_OBSERVED',a.label,prewo_drift,flush=True)
 import read_gradient_capture as capture
 import read_gradient_stats as statistics
 capture.install();L=int(cfg.num_decoder_layers);D=int(cfg.emb_dim);H=int(cfg.num_query_heads);V=int(cfg.bam_v);R=int(cfg.bam_local_full_m_read_bottleneck_dim)
 assert cfg.bam_k==cfg.head_dim
 P=jax.sharding.PartitionSpec;NS=lambda spec:jax.sharding.NamedSharding(mesh,spec)
 zero=dict(up=jax.device_put(jnp.zeros((L,2,R,H,1,V),jnp.float32),NS(P(None,None,'fsdp',None,None,None))),down=jax.device_put(jnp.zeros((L,4,D,R),jnp.float32),NS(P(None,None,'fsdp',None))),out=jax.device_put(jnp.zeros((L,3,H,cfg.head_dim,D),jnp.float32),NS(P(None,None,None,None,'fsdp'))))
 def objective(weights,delta,b,objective_mask):
  capture.DELTA=delta
  try:loss=forward(weights,b)
  finally:capture.DELTA=None
  selected=objective_mask*b['mask'];denominator=jnp.sum(selected)
  return jnp.sum(loss*selected)/jnp.maximum(denominator,1.),jnp.sum(loss*b['mask'],axis=1)/jnp.maximum(b['mask'].sum(1),1)
 grad_fn=jax.jit(jax.value_and_grad(objective,argnums=1,has_aux=True));gram_fn=statistics.sharded_gram_function(mesh)
 finite_fn=jax.jit(lambda tree:jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(tree)])))
 def gate_bias(weights,offset):
  flat=flatten_dict(weights);new={};matched=0
  for p,v in flat.items():
   if p[-1]=='W_lv_gate_b0':v=v+offset;matched+=1
   new[p]=v
  if not matched:raise ValueError('No V gate bias parameters')
  return unflatten_dict(new)
 weight_norms={'up':[None]*L,'down':[None]*L,'out':[None]*L}
 import re,gc
 for p,v in flatten_dict(params).items():
  path='/'.join(p);kind='up' if path.endswith('/self_attention/W_R_up/kernel') else 'down' if path.endswith('/self_attention/full_m_read_down/kernel') else 'out' if path.endswith('/self_attention/out/kernel') else None
  if kind is None:continue
  layer=re.search(r'/layers_(\d+)/',path);tail=re.search(r'/final_local_layer(?:_(\d+))?/',path)
  if layer:pairs=[(int(layer[1]),v)]
  elif tail:
   tails=cfg.bam_final_local_layer_count
   if tails is None:tails=int(bool(cfg.bam_extra_final_local_layer))
   pairs=[(L-tails+int(tail[1] or 0),v)]
  else:
   role=re.search(r'/(local|fetch)_(\d+)/',path);assert role
   pairs=[(i*cfg.bam_local_fetch_block_size+int(role[2]),jnp.take(v,i,axis=1)) for i in range(v.shape[1])]
  for layer,w in pairs:weight_norms[kind][layer]=float(jax.device_get(jnp.sqrt(jnp.sum(w.astype(jnp.float32)**2))))
 assert all(v is not None for values in weight_norms.values() for v in values)
 modes={}
 for bias in [0.,.5]:
  weights=params if bias==0 else gate_bias(params,bias)
  expected_losses=expected['native_losses' if bias==0 else 'Vbias0p5_losses']
  for objective_name in ['all','midcontent']:
   mode=('native' if bias==0 else 'Vbias0p5')+'__'+objective_name;gradients=[];counts=[];values=[];predictions=[];mode_start=time.time()
   masks=data['objective_'+objective_name][:a.sequences,:a.length]
   for i,b in enumerate(batches):
    selected=masks[i*a.batch_size:(i+1)*a.batch_size];count=float(selected.sum());assert count>0
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):(value,prediction),g=grad_fn(weights,zero,b,jnp.asarray(selected))
    value.block_until_ready()
    if not bool(jax.device_get(finite_fn(g))):raise ValueError(('Nonfinite path gradients',mode,i))
    gradients.append(g);counts.append(count);values.append(float(value));predictions.extend(np.asarray(jax.device_get(prediction),np.float32).tolist())
    print('READ_GRADIENT_BATCH',a.label,mode,i,float(value),flush=True)
   drift=float(np.max(np.abs(np.asarray(predictions)-expected_losses)))
   if drift>.005:raise ValueError(('Zero-delta forward changed CE',mode,drift))
   stacked=jax.tree.map(lambda *xs:jnp.stack(xs),*gradients)
   jax.block_until_ready(stacked)
   del gradients,g;gc.collect()
   grams=jax.device_get(gram_fn(stacked))
   assert all(np.isfinite(x).all() for x in grams.values())
   assert grams['up_gram'].shape==(L,H,2,2,len(batches),len(batches))
   assert grams['down_gram'].shape==(L,4,4,len(batches),len(batches))
   assert grams['out_gram'].shape==(L,H,3,3,len(batches),len(batches))
   np.savez_compressed(out/(mode+'.npz'),**grams,batch_token_counts=np.asarray(counts),batch_objective_CE=np.asarray(values),per_sequence_CE=np.asarray(predictions))
   modes[mode]=dict(file=mode+'.npz',V_gate_offset=bias,objective=objective_name,total_objective_tokens=sum(counts),native_reference_max_CE_delta=drift,elapsed_seconds=time.time()-mode_start)
   (out/'progress.json').write_text(json.dumps(dict(label=a.label,completed=list(modes)),indent=2)+'\n')
   subprocess.run(['gsutil','-q','cp',str(out/(mode+'.npz')),a.gcs_output+'/'+mode+'.npz'],check=True)
   print('READ_GRADIENT_MODE_COMPLETE',a.label,mode,flush=True)
   del stacked,grams;gc.collect()
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=expected['parameter_count'],sequence_hashes=hashes,sequences=a.sequences,length=a.length,batch_size=a.batch_size,config=dict(num_decoder_layers=L,emb_dim=D,num_query_heads=H,bam_v=V,read_hidden=R),native_losses=baseline.mean(1).tolist(),native_reference_max_CE_delta=native_error,prewo_observer_max_CE_delta=prewo_drift,prewo_O_closure_max_relative_rms=prewo_closure_max_relative_rms,reader_weight_norms=weight_norms,modes=modes,elapsed_seconds=time.time()-start,definition='Zero-delta parameter-path gradients, no parameter updates. Four down paths Q/K/V/O sum to native shared full_m_read_down gradient, two up paths V/O sum to native W_R_up gradient; output-projection paths MHA/directO/rounding sum to native W_o gradient in FP32 checks. BF16 rounding is explicitly separate. Per-batch pairwise Gram matrices permit exact bootstrap of weighted pooled gradient cosines. FP32 CE statistics on native model/logits dtype. Raw gradients are not AdamW updates and do not establish training-gain attribution.')
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,modes=len(modes)))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('READ_GRADIENT_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

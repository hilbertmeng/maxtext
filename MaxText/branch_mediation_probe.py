"""Read-only branch/FFN mediation counterfactuals; no optimizer or updates."""
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
 p.add_argument('--reference-json',default='/tmp/bam_stream_probe/branch_mediation_reference.json');p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/branch_mediation_cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
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
 key=a.label.removeprefix('branch_mediation_');expected=reference['models'][key]
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
 import branch_mediation_capture as mediation
 import prewo_observer as observer
 mediation.install();mediation.OBSERVE=True;mediation.RETAIN_LOCAL=False
 L=int(cfg.num_decoder_layers);zero=np.zeros((L,3),np.float32)
 # Primary: pure FFN mediation at non-MLP-write sites, matched by relative depth.
 nonwrite=[i for i in range(L) if not mediation.layer_writes_m(cfg,i)]
 layer_indices=[min(nonwrite,key=lambda i:(abs((i+1)-L*f),i)) for f in [.25,.5,.75,1.0]]
 assert len(set(layer_indices))==4
 # Secondary: one nearby MLP-write site, explicitly marked as combined mediation.
 writes=[i for i in range(L-1) if mediation.layer_writes_m(cfg,i)]
 if writes:layer_indices.append(min(writes,key=lambda i:(abs((i+1)-.75*L),i)))
 assert len(set(layer_indices))==len(layer_indices)
 def controlled(weights,b,coefficients,live):
  mediation.CONTROLS=dict(coefficients=coefficients,live=live);observer.ACTIVE=True
  try:
   value,raw=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  finally:mediation.CONTROLS=None;observer.ACTIVE=False
  moments={}
  for path,entries in flatten_dict(raw['intermediates']).items():
   if path[-1] not in ['stream_mediation_tangent','stream_prewo_O_closure','stream_prewo_O_sources']:continue
   if not isinstance(entries,(tuple,list)):entries=[entries]
   moments['__'.join(path)]=sum(entries)
  return value[0],moments
 controlled_fn=jax.jit(controlled)
 def evaluate_control(coefficients,live):
  losses=[];observations={}
  for b in batches:
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):value,moments=jax.device_get(controlled_fn(params,b,jnp.asarray(coefficients),jnp.asarray(live)))
   losses.append(np.asarray(value,np.float32))
   for name,values in moments.items():observations.setdefault(name,[]).append(np.asarray(values))
  observations={k:np.concatenate(v,axis=1 if 'layers' in k.split('__') else 0) for k,v in observations.items()}
  if not observations:raise ValueError('Missing mediation moments')
  for name,values in observations.items():
   if not np.isfinite(values).all():raise ValueError(('Nonfinite observation',name))
   if name.endswith('stream_prewo_O_closure'):
    source=observations[name.removesuffix('stream_prewo_O_closure')+'stream_prewo_O_sources'];den=np.maximum(source[...,3],np.max(source[...,3])*1e-12+1e-30)
    if np.sqrt(np.maximum(values,0)/den).max()>.02:raise ValueError(('Dynamic/static O closure failed',name))
  value=np.concatenate(losses)
  assert value.shape==(a.sequences,a.length) and np.isfinite(value).all()
  return value,observations
 neutral,moments=evaluate_control(zero,False);neutral_live,_=evaluate_control(zero,True)
 if np.max(np.abs(neutral.mean(1)-baseline.mean(1)))>.005:raise ValueError('Instrumented neutral changed native CE')
 np.testing.assert_array_equal(neutral_live,neutral)
 np.savez_compressed(out/'neutral.npz',loss=neutral,**moments)
 subprocess.run(['gsutil','-q','cp',str(out/'neutral.npz'),a.gcs_output+'/neutral.npz'],check=True)
 print('MEDIATION_NEUTRAL_NATIVE_AND_LIVE_IDENTITY_VERIFIED',a.label,flush=True)
 variants=[]
 for layer in layer_indices:
  for arm,name in [(0,'MHA'),(1,'dynamicO')]:
   for dose in [-.1,-.05,.05,.1]:
    coefficients=zero.copy();coefficients[layer,arm]=dose
    for live in [False,True]:
     label=f'L{layer+1}_{name}_{dose:+.2f}_'+('live' if live else 'frozen')
     value,moments=evaluate_control(coefficients,live)
     filename=label+'.npz';np.savez_compressed(out/filename,loss=value,**moments)
     delta=(value-neutral).mean(1)
     variants.append(dict(label=label,file=filename,layer_one_based=layer+1,writes_M_here=mediation.layer_writes_m(cfg,layer),source=name,dose=dose,live_FFN_and_M_writes=live,per_sequence_CE_delta=delta.tolist(),mean_CE_delta=float(delta.mean())))
     subprocess.run(['gsutil','-q','cp',str(out/filename),a.gcs_output+'/'+filename],check=True)
     print('MEDIATION_VARIANT_COMPLETE',a.label,label,float(delta.mean()),flush=True)
 # Repeat neutral to guard global-state leakage and parameter mutation.
 repeated,_=evaluate_control(zero,False);np.testing.assert_array_equal(repeated,neutral)
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=expected['parameter_count'],sequence_hashes=hashes,sequences=a.sequences,length=a.length,batch_size=a.batch_size,native_reference_max_CE_delta=native_error,instrumented_neutral_max_CE_delta=float(np.max(np.abs(neutral.mean(1)-baseline.mean(1)))),instrumented_neutral_abs_token_CE_delta_quantiles=np.quantile(np.abs(neutral-baseline),[.5,.9,.99,1]).tolist(),layer_indices_zero_based=layer_indices,layers_one_based=[v+1 for v in layer_indices],variants=variants,config=dict(num_decoder_layers=L,bam_local_fetch_block_size=cfg.bam_local_fetch_block_size,bam_final_local_layer_count=cfg.bam_final_local_layer_count,bam_extra_final_local_layer=cfg.bam_extra_final_local_layer),moment_fields=['perturbation_energy','common_mode_energy','ideal_RMS_tangent_energy','ideal_LN_tangent_energy','element_count','actual_norm_JVP_energy','norm_isotropic_reference_energy'],elapsed_seconds=time.time()-start,definition='Identical projected branch perturbation added at block vector exit in both modes; live mode additionally changes post-attention FFN normalization input, letting native FFN and M writes respond. Frozen mode preserves native FFN/M writes. Identical residual addition order controls floating-point association artifacts. No parameter/optimizer updates or checkpoint writes. Local finite interventions are not retraining benefit or additive shares of loss.')
 assert len(variants)==len(layer_indices)*2*4*2
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,variants=len(variants)))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('BRANCH_MEDIATION_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

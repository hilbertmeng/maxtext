"""Read-only native value-route and attention-mass observations; no optimizer or updates."""
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
 p.add_argument('--reference-json',default='/tmp/bam_stream_probe/value_route_reference.json');p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/value_route_cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
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
 key=a.label.removeprefix('value_route_');expected=reference['models'][key]
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
 import value_route_capture as hook
 hook.install();hook.ACTIVE=True
 def observed(weights,b):
  value,raw=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  captured={}
  for path,entries in flatten_dict(raw['intermediates']).items():
   if not path[-1].startswith('stream_route_'):continue
   assert isinstance(entries,(tuple,list)) and len(entries)==1
   captured['__'.join(path)]=entries[0]
  return value[0],captured
 observed_fn=jax.jit(observed);arrays={};losses=[]
 for i,b in enumerate(batches):
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,stats=jax.device_get(observed_fn(params,b))
  assert stats
  losses.append(np.asarray(loss,np.float32))
  for name,v in stats.items():arrays.setdefault(name,[]).append(np.asarray(v))
  print('VALUE_ROUTE_BATCH_COMPLETE',a.label,i+1,len(batches),flush=True)
 arrays={k:np.concatenate(v,axis=1 if 'layers' in k.split('__') else 0) for k,v in arrays.items()}
 loss=np.concatenate(losses);drift=float(np.max(np.abs(loss.mean(1)-baseline.mean(1))))
 if drift>.005:raise ValueError(('Observer changed native CE',drift))
 checks={name:[] for name in ['V_closure_error','O_closure_error','AV_reconstruction_error']}
 for name,values in arrays.items():
  if not np.isfinite(values).all():raise ValueError(('Nonfinite value-route observation',name))
  if not name.endswith('stream_route_energy'):continue
  # Per model-layer / text, pool the fixed16queries and all heads, never divide by nearzero individual heads.
  for field,den in [('V_closure_error','V'),('O_closure_error','O'),('AV_reconstruction_error','MHA')]:
   numerator=values[...,hook.ENERGY_NAMES.index(field)].sum(axis=(-2,-1));denominator=values[...,hook.ENERGY_NAMES.index(den)].sum(axis=(-2,-1))
   checks[field].extend(np.sqrt(numerator/np.maximum(denominator,1e-30)).reshape(-1).tolist())
 for field,values in checks.items():
  if max(values)>.02:raise ValueError(('Native value-route reconstruction failed',field,max(values)))
 np.savez_compressed(out/'observations.npz',**arrays)
 np.savez_compressed(out/'observed_losses.npz',loss=loss)
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=expected['parameter_count'],sequence_hashes=hashes,sequences=a.sequences,length=a.length,batch_size=a.batch_size,native_reference_max_CE_delta=native_error,observer_max_CE_delta=drift,observer_abs_token_CE_delta_quantiles=np.quantile(np.abs(loss-baseline),[.5,.9,.99,1]).tolist(),query_positions=hook.positions(a.length).tolist(),energy_names=hook.ENERGY_NAMES,rank_names=hook.RANK_NAMES,rank_fields=['raw_participation_rank','unit_direction_participation_rank','mean_square'],self_fields=['diagonal_probability_mass','self_argmax_fraction','normalized_entropy'],self_query_groups=['all','q_ge_64'],reconstruction_max_RMS={k:max(v) for k,v in checks.items()},config=dict(num_decoder_layers=cfg.num_decoder_layers,num_query_heads=cfg.num_query_heads,head_dim=cfg.head_dim,bam_local_fetch_block_size=cfg.bam_local_fetch_block_size,bam_final_local_layer_count=cfg.bam_final_local_layer_count,bam_extra_final_local_layer=cfg.bam_extra_final_local_layer),elapsed_seconds=time.time()-start,definition='Native forward unchanged. Observe actual Q/K/V, MHA and dynamic/static V/O readouts. FP32 reconstructed softmax; full-query self statistics plus fixed16queries for value moments. DynamicV/O share ungated read direction. Headwise projection onto current dynamicO is a geometric redundancy measurement, not a causal contextual-information or training-benefit fraction. No optimizer updates or checkpoint writes.')
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,sequences=a.sequences))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('VALUE_ROUTE_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

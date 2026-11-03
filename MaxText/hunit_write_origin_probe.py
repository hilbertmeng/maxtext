"""Native same-cohort attribution of HUnit residual writes to fixed43k Wc weak basis."""
import argparse,ast,gc,hashlib,io,json,subprocess,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
import exp,max_utils,pyconfig,head_spectral_diagnostic as hd
from layers.models import Transformer
from hunit_write_origin import WriteObserver,LastReadAblation,aggregate,NAMES
_STORES={}
def blob(uri,value=None,missing=False):
 prefix,key=uri.rsplit('/',1)
 if prefix not in _STORES:_STORES[prefix]=hd.ts.KvStore.open(prefix+'/').result()
 store=_STORES[prefix]
 if value is not None:store.write(key,value).result();return
 q=store.read(key).result()
 if q.state=='missing' and missing:return None
 assert q.state=='value',uri
 return q.value
def jblob(uri):
 b=blob(uri,missing=True);return None if b is None else json.loads(b)
def emit(path,uri):blob(uri,path.read_bytes())
def main():
 p=argparse.ArgumentParser()
 for k in ['manifest','output','training-config']:p.add_argument('--'+k,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit'];root=Path(a.output);root.mkdir(exist_ok=True,parents=True)
 jax.config.update('jax_compilation_cache_dir',str(root/'jax_cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
 if jblob(m['gcs_root']+'/complete.json'):print('WRITE_ORIGIN_REUSED_ALL',flush=True);return
 attrs=dict(model_name=m['experiment'],only_eval=True,enable_checkpointing=False,per_device_batch_size=1.,eval_per_device_batch_size=1.,target_global_batch_size=1,max_target_length=2048,max_prefill_predict_length=2048,dataset_type='synthetic',log_config=False,jax_cache_dir='',enable_goodput_recording=False,monitor_goodput=False,record_training_health_metrics=False,record_internal_nn_metrics=False,remat_policy='none',tensorboard_dir='/tmp/hunit_origin/tensorboard/',base_output_directory='/tmp/hunit_origin/')
 exp.OriginHUnitInference=type('OriginHUnitInference',(getattr(exp,m['experiment']),),attrs)
 cfg=pyconfig.initialize(['hunit_origin','MaxText/configs/base.yml'],exp_class='OriginHUnitInference',only_eval=True,enable_checkpointing=False,run_name='hunit_origin',base_output_directory='/tmp/hunit_origin/')
 critical={'decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','partial_scan_layers','dtype','weight_dtype','attention','attention_type','logits_via_embedding','normalize_embedding_logits','normalization_layer_epsilon','qk_norm','dense_conn'};checked=[]
 for line in Path(a.training_config).read_text().splitlines():
  k,sep,raw=line.partition(': ')
  if not sep or k not in cfg.get_keys() or not (k in critical or k.startswith(('bam_','pythia_'))):continue
  try:want=ast.literal_eval(raw)
  except (ValueError,SyntaxError):want=raw
  have=getattr(cfg,k)
  if isinstance(have,type):have=have.__name__
  if isinstance(have,tuple):have=list(have)
  if isinstance(want,tuple):want=list(want)
  assert have==want,(k,str(want),str(have));checked.append(k)
 assert cfg.bam_pallas_core and cfg.bam_splash_attention and not cfg.bam_context_local_o_complement and not cfg.dropout_rate and not cfg.dense_conn
 assert cfg.emb_dim==2560 and cfg.vocab_size==50304
 u=jnp.asarray(np.load(io.BytesIO(blob(m['basis_gcs']))),jnp.float32);assert u.shape==(2560,128)
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.PRNGKey(20261021);max_utils.cross_entropy_with_logits=hd.precise_cross_entropy
 @jax.jit
 def feature(params,x,y):
  g=params['params']['decoder']['lm_head']['decoder_norm']['scale'];b=params['params']['decoder']['lm_head']['decoder_norm']['bias'];observer=WriteObserver(u,g)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),observer.observe():out,coll=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  result=aggregate(observer,coll,g,b,cfg.normalization_layer_epsilon);return out[0],result
 @jax.jit
 def causal(params,x,y,mode,reference_h):
  observer=LastReadAblation(mode)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),observer.observe():out=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  h=hd.BF16_round(observer.h);raw=hd.BF16_round(observer.raw);pc=jnp.matmul(h,u,precision=jax.lax.Precision.HIGHEST)
  return out[0],jnp.mean(jnp.sum(pc*pc,-1)/2560),jnp.sqrt(jnp.mean(h*h)),jnp.sqrt(jnp.mean(raw*raw)),jnp.sqrt(jnp.mean((h-reference_h)**2)/jnp.mean(reference_h**2))
 all_results={}
 for step in m['steps']:
  uri=m['gcs_root']+f'/{step}';folder=root/str(step);folder.mkdir(exist_ok=True);done=jblob(uri+'/complete.json')
  if done:assert done['source_commit']==commit;all_results[str(step)]=done;continue
  print('WRITE_ORIGIN_RESTORE_START',step,flush=True);flat,identity=hd.restore(m['checkpoints'][str(step)],cache_dir=str(root/'weight_cache'/str(step)));assert identity['state_step']==step+1 and identity['parameter_count']==m['parameter_count'];params=jax.device_put({'params':unflatten_dict(flat)});del flat;gc.collect();print('WRITE_ORIGIN_RESTORE_COMPLETE',step,flush=True);rows=[]
  for i in range(32):
   stem=f'sequence_{i:03d}';marker=jblob(uri+'/'+stem+'.json')
   if marker:assert marker['source_commit']==commit and marker['input_hash']==m['cohort_hashes'][i];rows.append(marker);continue
   source=m['reference_gcs']+f'/results/{step}/{stem}';meta=json.loads(blob(source+'.json'));assert meta['source_commit']==m['reference_capture_commit'] and meta['checkpoint']==m['checkpoints'][str(step)] and meta['input_hash']==m['cohort_hashes'][i]
   with np.load(io.BytesIO(blob(source+'.npz'))) as z:x=z['inputs'].copy();y=z['targets'].copy();goldh=z['h'].copy();goldloss=z['native_loss'].copy();goldraw=z['raw'].copy()
   tokens=np.concatenate([x[0],y[0,-1:]]).astype(np.int32);assert hashlib.sha256(tokens.tobytes()).hexdigest()==m['cohort_hashes'][i]
   tick=time.time();loss,s=feature(params,jnp.asarray(x),jnp.asarray(y));loss,s=jax.device_get((loss,s));seconds=time.time()-tick
   h=s.pop('h');raw=s.pop('raw');relative=float(np.linalg.norm(h-goldh)/np.linalg.norm(goldh));raw_relative=float(np.linalg.norm(raw-goldraw)/np.linalg.norm(goldraw));drift=float(loss.mean()-goldloss.mean());assert relative<.015 and abs(drift)<.005,('Observation graph failed archival replay gate',step,i,relative,drift)
   metrics={k:float(s[k]) for k in ['head_allocation_closure','raw_allocation_closure','projection_ledger_relative_error','layer_continuity_max','layer_continuity_relative_error']};print('WRITE_ORIGIN_LEDGER',step,i,metrics,flush=True)
   assert s['head_allocation'].shape==(32,7);assert abs(s['head_allocation_closure'])<2e-4 and abs(s['raw_allocation_closure'])<2e-4 and s['projection_ledger_relative_error']<1e-4 and s['layer_continuity_relative_error']<2e-6,(step,i,'Attribution ledger does not close',metrics)
   assert all(np.isfinite(v).all() for v in s.values())
   cf=[jax.device_get(causal(params,jnp.asarray(x),jnp.asarray(y),jnp.asarray(mode,jnp.int32),jnp.asarray(h))) for mode in range(3)]
   assert float(cf[0][4])<.015 and abs(float(cf[0][0].mean()-loss.mean()))<.005,('Causal keep mismatch',step,i)
   s.update(causal_CE=np.stack([q[0].reshape(-1) for q in cf]),causal_head_weak_power=np.array([q[1] for q in cf]),causal_head_RMS=np.array([q[2] for q in cf]),causal_raw_RMS=np.array([q[3] for q in cf]),causal_head_relative_to_observed=np.array([q[4] for q in cf]))
   path=folder/(stem+'.npz');np.savez_compressed(path,**s);emit(path,uri+'/'+path.name)
   marker=dict(source_commit=commit,training_commit=m['training_commit'],checkpoint=identity['checkpoint'],state_step=identity['state_step'],label_step=step,sequence=i,input_hash=m['cohort_hashes'][i],native_CE=float(loss.mean()),archival_CE=float(goldloss.mean()),CE_replay_difference=drift,head_relative_RMS=relative,raw_relative_RMS=raw_relative,head_allocation_closure=float(s['head_allocation_closure']),raw_allocation_closure=float(s['raw_allocation_closure']),projection_ledger_relative_error=float(s['projection_ledger_relative_error']),layer_continuity_max=float(s['layer_continuity_max']),seconds=seconds)
   path=folder/(stem+'.json');path.write_text(json.dumps(marker,indent=2)+'\n');emit(path,uri+'/'+path.name);rows.append(marker);print('WRITE_ORIGIN_CASE_COMPLETE',step,i,'CE',marker['native_CE'],'Hrel',relative,'seconds',seconds,flush=True)
  result=dict(source_commit=commit,checkpoint=identity['checkpoint'],state_step=identity['state_step'],sequences=32,sequence_hashes=m['cohort_hashes'],components=NAMES,fixed_basis_checkpoint=43000,checked_configuration_keys=checked,sequence=rows);path=folder/'complete.json';path.write_text(json.dumps(result,indent=2)+'\n');emit(path,uri+'/complete.json');all_results[str(step)]=result;del params;gc.collect();print('WRITE_ORIGIN_CHECKPOINT_COMPLETE',step,flush=True)
 result=dict(source_commit=commit,device_kind=jax.devices()[0].device_kind,jax_version=jax.__version__,steps=m['steps'],cases=64,checkpoints=all_results,scope='Native Pallas/Splash inference only. Fixed43k Wc bottom128 basis. Signed inner-product allocation of actual residual writes, both centered pre-LN and frozen final-LN denominator mapping. Static Mo uses same native read with dynamic O scale zero; dynamic is total minus static. Full FP32 Wo linear decomposition, native bias and rounding accounted separately. Direct write consumers, not upstream memory writer causality or retraining benefit.')
 path=root/'complete.json';path.write_text(json.dumps(result,indent=2)+'\n');emit(path,m['gcs_root']+'/complete.json');print('WRITE_ORIGIN_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()

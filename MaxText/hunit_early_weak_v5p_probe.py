"""Native same-cohort attribution of HUnit residual writes to fixed43k Wc weak basis."""
import argparse,ast,gc,hashlib,io,json,subprocess,time,fcntl
from pathlib import Path
from diagnostic_json import dumps as diagnostic_dumps
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
import exp,max_utils,pyconfig,head_spectral_diagnostic as hd
from layers.models import Transformer
from hunit_write_origin import WriteObserver,LastReadAblation,BoundaryObserver,aggregate,aggregate_boundaries,NAMES,layers
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
class MemoryObserver(WriteObserver):
 """Additional statistics only on physical final-layer memory and native read outputs."""
 def read(self,*args,**kw):
  out=super().read(*args,**kw)
  if self.active is not None:
   memory=hd.BF16_round(args[0]);assert memory.ndim==4
   # v8t memory is B,T,address,data; keep the token dimension for quantiles.
   self.active['memory_stats']=dict(memory_token_power=jnp.mean(memory*memory,(-2,-1)),memory_token_mean=jnp.mean(memory,(-2,-1)),memory_data_centered_power=jnp.mean((memory-memory.mean(-1,keepdims=True))**2,(-2,-1)))
   static=self.active['mo_static'].astype(jnp.float32);total=out[3].astype(jnp.float32);dyn=total-static
   self.active['memory_stats']['read_three_power']=jnp.stack([jnp.mean(a*a,(0,1,3)) for a in [dyn,static,total]])
  return out

 def interceptor(self,next_fun,args,kw,ctx):
  mod=ctx.module
  if mod.__class__.__name__=='SubDecoderLayer' and ctx.method_name=='__call__' and (not self.last_only or any('final_local_layer' in p for p in mod.scope.path)):
   def captured(*a,**k):
    out=next_fun(*a,**k)
    for name,value in self.active['memory_stats'].items():mod.sow('intermediates','memory_'+name,value)
    return out
   return super().interceptor(captured,args,kw,ctx)
  return super().interceptor(next_fun,args,kw,ctx)

MEMORY_NAMES=['memory_token_power','memory_token_mean','memory_data_centered_power','read_three_power']
def memory_values(collection):
 return {name:layers(collection,'memory_'+name)[-1] for name in MEMORY_NAMES}

def main():
 p=argparse.ArgumentParser()
 for k in ['manifest','output','training-config']:p.add_argument('--'+k,required=True)
 p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
 a=p.parse_args();assert 0<=a.shard_index<a.num_shards<=4
 m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit'];root=Path(a.output);root.mkdir(exist_ok=True,parents=True)
 assert jax.device_count()==1,('Each inference process must see one chip',jax.devices())
 print('V5P_SHARD_DEVICE',a.shard_index,jax.devices(),flush=True)
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
 def native(params,x,y):
  observer=hd.HeadCapture()
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(observer.interceptor):
   out=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  return out[0],hd.BF16_round(observer.h),hd.BF16_round(observer.raw)
 @jax.jit
 def focused(params,x,y,basis):
  g=params['params']['decoder']['lm_head']['decoder_norm']['scale'];b=params['params']['decoder']['lm_head']['decoder_norm']['bias'];observer=MemoryObserver(basis,g,last_only=True)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),observer.observe():out,coll=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  result=aggregate(observer,coll,g,b,cfg.normalization_layer_epsilon);result.update(memory_values(coll))
  return out[0],result
 @jax.jit
 def causal(params,x,y,mode,reference_h):
  observer=LastReadAblation(mode)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),observer.observe():out=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  h=hd.BF16_round(observer.h);raw=hd.BF16_round(observer.raw);pc=jnp.matmul(h,u,precision=jax.lax.Precision.HIGHEST)
  return out[0],jnp.mean(jnp.sum(pc*pc,-1)/2560),jnp.sqrt(jnp.mean(h*h)),jnp.sqrt(jnp.mean(raw*raw)),jnp.sqrt(jnp.mean((h-reference_h)**2)/jnp.mean(reference_h**2))
 all_results={}
 cohort=np.load(io.BytesIO(blob(m['cohort_gcs'])))['tokens'];assert cohort.shape==(32,2049)
 for i,tokens in enumerate(cohort):assert hashlib.sha256(tokens.astype(np.int32).tobytes()).hexdigest()==m['cohort_hashes'][i]
 for step in m.get('execution_order',m['steps']):
  uri=m['gcs_root']+f'/{step}';folder=root/str(step);folder.mkdir(exist_ok=True);done=jblob(uri+f'/complete_shard_{a.shard_index}.json')
  if done:assert done['source_commit']==commit;all_results[str(step)]=done;continue
  print('WRITE_ORIGIN_RESTORE_START',step,flush=True);cache=root.parent/'weight_cache'/str(step);cache.mkdir(parents=True,exist_ok=True)
  with (cache/'restore.lock').open('a+b') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);flat,identity=hd.restore(m['checkpoints'][str(step)],cache_dir=str(cache))
  assert identity['state_step']==step+1 and identity['parameter_count']==m['parameter_count'];scalars={'/'.join(k):np.asarray(v).reshape(-1).tolist() for k,v in flat.items() if 'final_local_layer' in '/'.join(k) and np.asarray(v).size<=32 and ('scale' in '/'.join(k) or 'gate' in '/'.join(k))};params=jax.device_put({'params':unflatten_dict(flat)});del flat;gc.collect();print('WRITE_ORIGIN_RESTORE_COMPLETE',step,flush=True);rows=[]
  own=jnp.asarray(np.load(io.BytesIO(blob(m['own_basis_root']+f'/{step}.npy'))),jnp.float32);assert own.shape==u.shape
  for i,tokens in enumerate(cohort):
   if i%a.num_shards!=a.shard_index:continue
   stem=f'sequence_{i:03d}';marker=jblob(uri+'/'+stem+'.json')
   if marker:assert marker['source_commit']==commit and marker['input_hash']==m['cohort_hashes'][i];rows.append(marker);continue
   x=jnp.asarray(tokens[None,:-1]);y=jnp.asarray(tokens[None,1:]);tick=time.time()
   goldloss,goldh,goldraw=jax.device_get(native(params,x,y));stats={};fidelities=[];archive_fidelity=None
   if step in [33000,43000]:
    source=m['reference_gcs']+f'/results/{step}/{stem}';oldmeta=json.loads(blob(source+'.json'));assert oldmeta['checkpoint']==identity['checkpoint'] and oldmeta['input_hash']==m['cohort_hashes'][i]
    with np.load(io.BytesIO(blob(source+'.npz'))) as old:
     oldh=old['h'];oldce=old['native_loss'];oldraw=old['raw']
    archive_fidelity=dict(head_relative=float(np.linalg.norm(goldh-oldh)/np.linalg.norm(oldh)),raw_relative=float(np.linalg.norm(goldraw-oldraw)/np.linalg.norm(oldraw)),mean_CE_difference=float(goldloss.mean()-oldce.mean()))
    basis=np.asarray(u);native_wp=np.mean(np.sum(np.matmul(goldh,basis)**2,-1));archived_wp=np.mean(np.sum(np.matmul(oldh,basis)**2,-1));archive_fidelity['weak_power_relative']=float(native_wp/archived_wp-1)
    archive_fidelity['accepted']=bool(abs(archive_fidelity['weak_power_relative'])<.015 and archive_fidelity['head_relative']<.015 and abs(archive_fidelity['mean_CE_difference'])<.005)
    if m.get('require_cross_hardware_identity',True):assert archive_fidelity['accepted'],('Cross-hardware native fidelity failed',step,i,archive_fidelity)
    print('V5P_V6E_ANCHOR_FIDELITY',step,i,archive_fidelity,flush=True)
   for label,basis in [('fixed43k',u),('own',own)]:
    loss,f=jax.device_get(focused(params,x,y,basis));fh=f.pop('h');fr=f.pop('raw');frel=float(np.linalg.norm(fh-goldh)/np.linalg.norm(goldh));fdrift=float(loss.mean()-goldloss.mean())
    assert frel<.015 and abs(fdrift)<.005,(step,i,label,'Observation changed forward',frel,fdrift)
    assert f['head_allocation'].shape==(1,7) and abs(f['head_allocation_closure'])<2e-4 and abs(f['raw_allocation_closure'])<2e-4 and f['projection_ledger_relative_error']<1e-4
    baseweak=np.mean(np.sum(np.einsum('btd,dr->btr',goldh,np.asarray(basis))**2,-1)/2560)
    assert abs(float(f['head_total_power'])/baseweak-1)<.015,(step,i,label,'Weak power fidelity')
    f.update(native_CE=np.asarray(loss).reshape(-1),baseline_head_weak_power=np.asarray(baseweak))
    stats.update({label+'_'+k:v for k,v in f.items()});fidelities.append(dict(basis=label,head_relative=frel,CE_difference=fdrift,weak_power_relative=float(f['head_total_power'])/baseweak-1))
   cf=[jax.device_get(causal(params,x,y,jnp.asarray(mode,jnp.int32),jnp.asarray(goldh))) for mode in range(3)]
   assert float(cf[0][4])<.015 and abs(float(cf[0][0].mean()-goldloss.mean()))<.005,(step,i,'Causal keep mismatch')
   stats.update(causal_CE=np.stack([q[0].reshape(-1) for q in cf]),causal_head_weak_power=np.array([q[1] for q in cf]),causal_head_RMS=np.array([q[2] for q in cf]),causal_raw_RMS=np.array([q[3] for q in cf]),causal_head_relative=np.array([q[4] for q in cf]),baseline_CE=goldloss.reshape(-1),baseline_head_power=np.mean(goldh**2),baseline_raw_power=np.mean(goldraw**2))
   assert all(np.isfinite(v).all() for v in stats.values())
   path=folder/(stem+'.npz');np.savez_compressed(path,**stats);emit(path,uri+'/'+path.name)
   marker=dict(source_commit=commit,checkpoint=identity['checkpoint'],state_step=identity['state_step'],label_step=step,sequence=i,input_hash=m['cohort_hashes'][i],baseline_CE=float(goldloss.mean()),archive_fidelity=archive_fidelity,shard_index=a.shard_index,num_shards=a.num_shards,device_kind=jax.devices()[0].device_kind,fidelity=fidelities,seconds=time.time()-tick)
   path=folder/(stem+'.json');path.write_text(diagnostic_dumps(marker,indent=2)+'\n');emit(path,uri+'/'+path.name);rows.append(marker);print('WRITE_ORIGIN_CASE_COMPLETE',step,i,'CE',marker['baseline_CE'],'seconds',marker['seconds'],flush=True)
  result=dict(source_commit=commit,checkpoint=identity['checkpoint'],state_step=identity['state_step'],sequences=len(rows),sequence_hashes=[q['input_hash'] for q in rows],shard_index=a.shard_index,num_shards=a.num_shards,components=NAMES,final_small_parameters=scalars,checked_configuration_keys=checked,sequence=rows);path=folder/f'complete_shard_{a.shard_index}.json';path.write_text(diagnostic_dumps(result,indent=2)+'\n');emit(path,uri+f'/complete_shard_{a.shard_index}.json');all_results[str(step)]=result;del params;gc.collect();print('WRITE_ORIGIN_CHECKPOINT_COMPLETE',step,flush=True)
 result=dict(source_commit=commit,device_kind=jax.devices()[0].device_kind,jax_version=jax.__version__,steps=m['steps'],cases=sum(q['sequences'] for q in all_results.values()),shard_index=a.shard_index,num_shards=a.num_shards,checkpoints=all_results,scope=m['scope'])
 path=root/f'complete_shard_{a.shard_index}.json';path.write_text(diagnostic_dumps(result,indent=2)+'\n');emit(path,m['gcs_root']+f'/complete_shard_{a.shard_index}.json');print('WRITE_ORIGIN_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()

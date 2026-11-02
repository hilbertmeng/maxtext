"""Exact HUnit FastKernel source, same fixed32 texts at33k/43k, no training."""
import argparse,ast,gc,hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
import exp,max_utils,pyconfig
from layers.models import Transformer
import head_spectral_diagnostic as hd

def save(path,**arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for k,v in arrays.items():
   with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(v),allow_pickle=False)
def json_gs(uri):
 p=subprocess.run(['gsutil','cat',uri],capture_output=True,text=True)
 if not p.returncode:return json.loads(p.stdout)
 if any(s in p.stderr for s in ['No URLs matched','matched no objects','404','NotFound']):return None
 raise RuntimeError(p.stderr[-500:])
def publish(path,meta,dest):
 subprocess.run(['gsutil','-q','cp',str(path),dest+'/'+path.name],check=True);p=path.with_suffix('.json');p.write_text(json.dumps(meta,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(p),dest+'/'+p.name],check=True)
def main():
 p=argparse.ArgumentParser()
 for k in ['manifest','output','training-config']:p.add_argument('--'+k,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit']
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 attrs=dict(model_name=m['experiment'],only_eval=True,enable_checkpointing=False,per_device_batch_size=1.,eval_per_device_batch_size=1.,target_global_batch_size=1,max_target_length=2048,max_prefill_predict_length=2048,dataset_type='synthetic',log_config=False,jax_cache_dir='',enable_goodput_recording=False,monitor_goodput=False,record_training_health_metrics=False,record_internal_nn_metrics=False,remat_policy='none',tensorboard_dir='/tmp/hunit_spectral/tensorboard/',base_output_directory='/tmp/hunit_spectral/')
 exp.SpectralHUnitInference=type('SpectralHUnitInference',(getattr(exp,m['experiment']),),attrs)
 cfg=pyconfig.initialize(['hunit_spectral','MaxText/configs/base.yml'],exp_class='SpectralHUnitInference',run_name='hunit_spectral',base_output_directory='/tmp/hunit_spectral/',enable_checkpointing=False,only_eval=True)
 critical={'decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','partial_scan_layers','dtype','weight_dtype','attention','attention_type','logits_via_embedding','normalize_embedding_logits','normalization_layer_epsilon','qk_norm'}
 checked=[]
 for line in Path(a.training_config).read_text().splitlines():
  k,sep,raw=line.partition(': ')
  if not sep or k not in cfg.get_keys() or not (k in critical or k.startswith(('bam_','pythia_'))):continue
  try:want=ast.literal_eval(raw)
  except (ValueError,SyntaxError):want=raw
  have=getattr(cfg,k)
  if isinstance(have,type):have=have.__name__
  if isinstance(have,tuple):have=list(have)
  if isinstance(want,tuple):want=list(want)
  assert have==want,('Saved configuration mismatch',k,str(want),str(have));checked.append(k)
 assert cfg.vocab_size==50304 and cfg.emb_dim==2560
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.PRNGKey(20261021)
 max_utils.cross_entropy_with_logits=hd.precise_cross_entropy
 @jax.jit
 def feature(params,x,y):
  observer=hd.HeadCapture()
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(observer.interceptor):
   native=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  assert observer.h is not None
  return observer.raw,observer.h,native[0]
 readout=jax.jit(hd.FP32_readout);native_ce=jax.jit(hd.native_ce)
 @jax.jit
 def stats(h,w,y,u,uc,mu,z):
  hf=h[0].astype(jnp.float32);proj=jnp.einsum('td,dk->tk',hf,u,precision=jax.lax.Precision.HIGHEST);pc=jnp.einsum('td,dk->tk',hf,uc,precision=jax.lax.Precision.HIGHEST)
  centered=z-z.mean(-1,keepdims=True);q=hd.BF16_round(z);qc=hd.BF16_round(centered)
  loss=lambda v:-jnp.take_along_axis(jax.nn.log_softmax(v),y[0,:,None],-1)[:,0]
  axis=mu/jnp.linalg.norm(mu);common=jnp.einsum('td,d->t',hf,axis,precision=jax.lax.Precision.HIGHEST)**2/2560
  return dict(head_power=jnp.mean(hf*hf,-1),raw_bands=jnp.sum(proj.reshape(2048,20,128)**2,-1)/2560,centered_bands=jnp.sum(pc.reshape(2048,20,128)**2,-1)/2560,common_axis_power=common,common_logit_offset=z.mean(-1),centered_logit_power=jnp.mean(centered*centered,-1),precision_CE=jnp.stack([loss(z),loss(q),loss(qc)]))
 for step in m['steps']:
  dest=m['gcs_root']+f'/results/{step}';folder=root/str(step);folder.mkdir(exist_ok=True)
  done=json_gs(dest+'/complete.json')
  if done:
   assert done['source_commit']==commit and done['checkpoint']==m['checkpoints'][str(step)];print('HUNIT_REUSED_COMPLETE',step,flush=True);continue
  listed=subprocess.run(['gsutil','ls',dest+'/'],capture_output=True,text=True)
  if not listed.returncode:subprocess.run(['gsutil','-q','-m','rsync','-r',dest,str(folder)],check=True)
  elif not any(s in listed.stderr for s in ['No URLs matched','matched no objects','404','NotFound']):raise RuntimeError(listed.stderr[-500:])
  spectrum=folder/'spectrum.npz';subprocess.run(['gsutil','-q','cp',m['gcs_root']+f'/head_spectra/{step}.npz',str(spectrum)],check=True)
  with np.load(spectrum) as z:u=jnp.asarray(z['basis']);uc=jnp.asarray(z['centered_basis']);mu=jnp.asarray(z['vocabulary_mean'])
  all_captured=all((folder/f'sequence_{i:03d}.json').exists() for i in range(32));flat,identity=hd.restore(m['checkpoints'][str(step)],all_captured)
  assert identity['state_step']==m['checkpoint_state_steps'][str(step)]
  w=jnp.asarray(flat[('decoder','lm_head','logits_dense')]);scale=jnp.asarray(flat[('decoder','lm_head','decoder_norm','scale')]);bias=jnp.asarray(flat[('decoder','lm_head','decoder_norm','bias')])
  if not all_captured:
   assert identity['parameter_count']==m['parameter_count'],identity
   params=jax.device_put({'params':unflatten_dict(flat)});del flat;gc.collect()
   cohort=folder/'cohort.npz';subprocess.run(['gsutil','-q','cp',m['cohort_source'],str(cohort)],check=True)
   with np.load(cohort) as z:tokens=z['tokens'][:,:2049].copy()
   assert tokens.shape==(32,2049) and tokens.max()<50304
   for i in range(32):
    path=folder/f'sequence_{i:03d}.npz'
    if path.with_suffix('.json').exists():continue
    assert hashlib.sha256(tokens[i].astype(np.int32).tobytes()).hexdigest()==m['cohort_hashes'][i]
    x=jnp.asarray(tokens[i:i+1,:2048]);y=jnp.asarray(tokens[i:i+1,1:2049]);raw,h,loss=feature(params,x,y)
    replay=native_ce(h,w,y);drift=float(abs(loss.mean()-replay.mean()));assert drift<.005,(step,i,drift)
    rc=raw.astype(jnp.float32)-raw.astype(jnp.float32).mean(-1,keepdims=True);norm=((rc*jax.lax.rsqrt((rc*rc).mean(-1,keepdims=True)+1e-5))*scale+bias).astype(jnp.bfloat16)
    relative=float(jnp.sqrt(jnp.mean((norm.astype(jnp.float32)-h.astype(jnp.float32))**2)/jnp.mean(h.astype(jnp.float32)**2)));assert relative<.015
    save(path,raw=np.asarray(raw,np.float32),h=np.asarray(h,np.float32),native_loss=np.asarray(loss),inputs=np.asarray(x),targets=np.asarray(y))
    publish(path,dict(source_commit=commit,training_commit=m['training_commit'],checkpoint=identity['checkpoint'],state_step=identity['state_step'],label_step=step,sequence=i,input_hash=m['cohort_hashes'][i],native_CE_replay_difference=drift,final_norm_relative_RMS=relative),dest)
    print('HUNIT_CAPTURE_COMPLETE',step,i,flush=True)
   del params;gc.collect()
  else:del flat
  total=np.zeros((2560,50304),np.float64);sequence_stats=[]
  for i in range(32):
   path=folder/f'sequence_{i:03d}.npz';marker=json.loads(path.with_suffix('.json').read_text());assert marker['source_commit']==commit and marker['input_hash']==m['cohort_hashes'][i]
   with np.load(path) as z:h=jnp.asarray(z['h'],jnp.bfloat16);y=jnp.asarray(z['targets']);native=float(z['native_loss'].mean());rawpower=np.mean(z['raw'].astype(np.float64)**2,-1)
   g,ce,z=readout(h,w,y);total+=np.asarray(g)/32;s=stats(h,w,y,u,uc,mu,z);arrays={k:np.asarray(v) for k,v in s.items()};arrays.update(FP32_CE=np.asarray(ce),raw_power=rawpower)
   save(folder/f'head_{i:03d}.npz',**arrays);publish(folder/f'head_{i:03d}.npz',dict(source_commit=commit,checkpoint=identity['checkpoint'],label_step=step,sequence=i,input_hash=m['cohort_hashes'][i]),dest)
   sequence_stats.append(dict(index=i,native_CE=native,FP32_CE=float(ce.mean())));print('HUNIT_HEAD_CASE',step,i,flush=True)
  g2=float(np.sum(total**2));gd=np.asarray(u[:,:128],np.float64).T@total;gc_=np.asarray(uc[:,:128],np.float64).T@total;axis=np.asarray(mu,np.float64);axis/=np.linalg.norm(axis);ga=axis@total
  result=dict(source_commit=commit,training_commit=m['training_commit'],experiment=m['experiment'],run_name=m['run_name'],checkpoint=identity['checkpoint'],state_step=identity['state_step'],label_step=step,sequences=32,sequence_hashes=m['cohort_hashes'],checked_configuration_keys=checked,global_head_gradient_norm=float(np.sqrt(g2)),raw_dark_gradient_energy=float(np.sum(gd*gd)),centered_dark_gradient_energy=float(np.sum(gc_*gc_)),common_axis_gradient_energy=float(np.sum(ga*ga)),sequence=sequence_stats,scope='Native BF16 network and FP32 diagnostic CE, frozen-head FP32 conditional gradient of identical32x2048 texts. Raw-W and vocabulary-centered-W bottom5% distinct; not formal training batch1024 backward or validation ranking.')
  p=folder/'complete.json';p.write_text(json.dumps(result,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(p),dest+'/complete.json'],check=True);print('HUNIT_CHECKPOINT_COMPLETE',step,flush=True)
if __name__=='__main__':main()

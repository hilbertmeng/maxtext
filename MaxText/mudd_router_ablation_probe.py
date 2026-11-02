"""Parameter-only M-route ablation with the original verified capture graph."""
import argparse,gc,hashlib,json,subprocess,time,shutil
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils
import mudd_spectral_capture as cap
import mudd_spectral_io as io
from mudd_spectral_model import serial_setup
from mudd_spectral_probe import save_npz,gs_json,publish

def reset_m_routes(params,targets,layers=32):
 """New pytree: targeted M route current-only; Q/K/V and every other leaf unchanged."""
 flat=flatten_dict(params['params']);result=dict(flat)
 for target in targets:
  prefix=('decoder','compose_final','mlp') if target==layers else ('decoder',f'layers_{target}','compose','mlp')
  key=prefix+('dynamic_dense_conn2','kernel');bias=prefix+('dense_proj2.bias',)
  w,b=flat[key],flat[bias];assert w.ndim==3 and b.ndim==2 and w.shape[-2:]==b.shape
  result[key]=w.at[:,-1,:].set(0)
  row=jnp.zeros_like(b[-1]).at[-1].set(1);result[bias]=b.at[-1].set(row)
 return {'params':unflatten_dict(result)}

def relative(a,b):return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-20))
def main():
 p=argparse.ArgumentParser()
 for n in ['manifest','output','training-config']:p.add_argument('--'+n,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit']
 info=m['models']['serial'];checkpoint=info['checkpoints']['53000'];root=Path(a.output);root.mkdir(parents=True,exist_ok=True);dest=m['gcs_root']+'/serial/53000'
 local_only=m.get('backend')=='cuda_math_reference'
 def finish_json(path,uri):
  if not local_only:subprocess.run(['gsutil','-q','cp',str(path),uri],check=True)
 def publish_frame(path,meta,uri):
  if local_only:path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
  else:publish(path,meta,uri)
 done=json.loads((root/'complete.json').read_text()) if local_only and (root/'complete.json').exists() else (None if local_only else gs_json(dest+'/complete.json'))
 if done:assert done['source_commit']==commit;print('ORIGIN_REUSED_COMPLETE',flush=True);return
 parent=root/'parent';parent.mkdir(exist_ok=True)
 if local_only:
  for source in Path(m['local_parent_data'],'results/serial/53000').glob('*'):
   target=parent/source.name
   if not target.exists():target.symlink_to(source)
 else:subprocess.run(['gsutil','-q','-m','rsync','-r',m['parent_artifact_root']+'/53000',str(parent)],check=True)
 ref_path=root/'reference.json'
 if local_only:shutil.copyfile(Path(m['local_parent_data'])/'softmax/serial/53000/complete.json',ref_path)
 else:subprocess.run(['gsutil','-q','cp',m['parent_gcs_root']+'/softmax/serial/53000/complete.json',str(ref_path)],check=True)
 reference=json.loads(ref_path.read_text())['global_head_gradients'][0]
 for fname,remote in [('raw_spectrum.npz','/head_spectra/serial_53000.npz'),('centered_spectrum.npz','/head_spectra/serial_53000_vocabulary_centered.npz')]:
  if local_only:shutil.copyfile(Path(m['local_parent_data'])/remote.lstrip('/'),root/fname)
  else:subprocess.run(['gsutil','-q','cp',m['parent_gcs_root']+remote,str(root/fname)],check=True)
 with np.load(root/'raw_spectrum.npz') as z:effective=jnp.asarray(z['effective_basis'][:,:128])
 with np.load(root/'centered_spectrum.npz') as z:dark=jnp.asarray(z['basis'][:,:128])
 cfg,checked,mesh,model,rng,abstract=serial_setup(a.training_config,label='mudd_router_ablation_gpu' if local_only else 'mudd_router_ablation',backend=m.get('backend','native_tpu'));assert cfg.mudd_full_history and not cfg.mudd_prenorm and not cfg.mudd_postnorm
 max_utils.cross_entropy_with_logits=cap.precise_cross_entropy
 print('ORIGIN_RESTORE_START',flush=True)
 flat,identity=io.params(checkpoint,'serial',False,workers=16,cache_dir=m.get('weight_cache'));assert identity['step']==53001 and identity['parameter_count']==2784962589
 print('ORIGIN_RESTORE_COMPLETE',identity,flush=True)
 w=jnp.asarray(flat[('decoder','lm_head','logits_dense')]);params=jax.device_put({'params':unflatten_dict(flat)});del flat;gc.collect()
 # Copy the original a03 native capture graph, including its unchanged outputs.
 def observed(params,x,y,dark_eff):
  observer=cap.Capture('serial',dark_eff,2048,32)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(observer.interceptor):
   native_out,collection=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  f=observer.result(collection['intermediates']);f['same_graph_native_loss']=native_out[0];return f
 feature=jax.jit(observed)
 @jax.jit
 def head_stats(h,href,y):
  hf=h[0].astype(jnp.float32);hr=href[0].astype(jnp.float32);wf=w.astype(jnp.bfloat16).astype(jnp.float32)
  z=jnp.einsum('td,dv->tv',hf,wf,precision=jax.lax.Precision.HIGHEST);zr=jnp.einsum('td,dv->tv',hr,wf,precision=jax.lax.Precision.HIGHEST)
  err=jax.nn.softmax(z).at[jnp.arange(2048),y[0]].add(-1);er=jax.nn.softmax(zr).at[jnp.arange(2048),y[0]].add(-1)
  g=jnp.einsum('td,tv->dv',hf,err,precision=jax.lax.Precision.HIGHEST)/2048
  proj=jnp.einsum('td,dk->tk',hf,dark,precision=jax.lax.Precision.HIGHEST);pr=jnp.einsum('td,dk->tk',hr,dark,precision=jax.lax.Precision.HIGHEST)
  gf=jnp.einsum('tk,tv->kv',proj,er,precision=jax.lax.Precision.HIGHEST)/2048;gr=jnp.einsum('tk,tv->kv',pr,er,precision=jax.lax.Precision.HIGHEST)/2048
  return g,gf,gr,jnp.mean(hf*hf,-1),jnp.sum(proj*proj,-1)/2560,cap.cross_entropy(z,y[0])
 initial=[('keep',[]),('all_m',list(range(1,33))),('all_body',list(range(1,32))),('body_02_08',list(range(1,8))),('body_09_16',list(range(8,16))),('body_17_24',list(range(16,24))),('body_25_32',list(range(24,32))),('final_only',[32])]
 results={};original_dark=None
 def arm(name,targets):
  directory=root/name;directory.mkdir(exist_ok=True);uri=dest+'/'+name;done=json.loads((directory/'complete.json').read_text()) if local_only and (directory/'complete.json').exists() else (None if local_only else gs_json(uri+'/complete.json'))
  if done:
   assert done['source_commit']==commit and done['targets']==targets
   if not local_only:subprocess.run(['gsutil','-q','-m','rsync','-r',uri,str(directory)],check=True)
   results[name]=done;print('ORIGIN_REUSED_ARM',name,flush=True);return done
  if not local_only:
   listed=subprocess.run(['gsutil','ls',uri+'/'],capture_output=True,text=True)
   if not listed.returncode:subprocess.run(['gsutil','-q','-m','rsync','-r',uri,str(directory)],check=True)
  variant=reset_m_routes(params,targets);full=np.zeros((2560,50432),np.float64);fixed=np.zeros((128,50432),np.float64);ref=np.zeros_like(fixed);power=[];d_power=[];losses=[];seq=[]
  for i in range(32):
   fpath=directory/f'sequence_{i:03d}.npz'
   with np.load(parent/fpath.name) as z:x=z['inputs'].copy();y=z['targets'].copy();href=z['h'].copy();old_ce=float(z['native_loss'].mean())
   assert hashlib.sha256(np.concatenate([x[0],y[0,-1:]]).astype(np.int32).tobytes()).hexdigest()==m['sequence_hashes'][i]
   if not fpath.with_suffix('.json').exists():
    f=jax.device_get(feature(variant,jnp.asarray(x),jnp.asarray(y),effective));h=np.asarray(f['h'],np.float32);native=np.asarray(f['same_graph_native_loss'],np.float32);comp=np.asarray(f['composition'])
    for target in targets:assert float(np.max(comp[:,target-1,3,1]))<1e-8,(name,target,'M-route did not reset')
    drift=float(native.mean()-old_ce);hr=relative(h,href)
    save_npz(fpath,h=h,native_loss=native,raw_RMS=np.sqrt(np.mean(np.asarray(f['raw'],np.float64)**2)),targets=y)
    meta=dict(source_commit=commit,checkpoint=checkpoint,state_step=53001,label_step=53000,arm=name,targets=targets,sequence=i,input_hash=m['sequence_hashes'][i],head_relative_RMS_difference=hr,native_CE_difference_from_original=drift)
    publish_frame(fpath,meta,uri)
    if name=='keep':assert abs(drift)<.005,(name,i,'Native CE mismatch',drift)
    print('ORIGIN_CAPTURE_COMPLETE',name,i,flush=True)
   else:
    meta=json.loads(fpath.with_suffix('.json').read_text());assert meta['source_commit']==commit and meta['targets']==targets and meta['input_hash']==m['sequence_hashes'][i]
    with np.load(fpath) as z:h=z['h'].copy();native=z['native_loss'].copy()
   g,gf,gr,p,dp,ce=head_stats(jnp.asarray(h,jnp.bfloat16),jnp.asarray(href,jnp.bfloat16),jnp.asarray(y));full+=np.asarray(g)/32;fixed+=np.asarray(gf)/32;ref+=np.asarray(gr)/32
   power.append(np.asarray(p));d_power.append(np.asarray(dp));losses.append(native.mean());seq.append(dict(index=i,native_CE=float(native.mean()),FP32_CE=float(ce.mean()),head_RMS_difference=meta['head_relative_RMS_difference']))
   print('ORIGIN_HEAD_CASE',name,i,flush=True)
  projected=np.asarray(dark,np.float64).T@full;g2=float(np.sum(full*full));gd2=float(np.sum(projected**2));ref2=float(np.sum(ref**2));assert abs(ref2/reference['dark_gradient_energy']-1)<2e-4
  row=dict(source_commit=commit,checkpoint=checkpoint,state_step=53001,arm=name,targets=targets,sequences=32,sequence_hashes=m['sequence_hashes'],native_CE=float(np.mean(losses)),head_RMS=float(np.sqrt(np.mean(power))),dark_RMS=float(np.sqrt(np.mean(d_power))),dark_input_energy_fraction=float(np.sum(d_power)/np.sum(power)),global_head_gradient_norm=float(np.sqrt(g2)),dark_gradient_energy=gd2,dark_gradient_fraction=gd2/g2,fixed_original_error_dark_gradient_norm=float(np.linalg.norm(fixed)),fixed_original_error_aligned_reduction=float(np.sum((ref-fixed)*ref)/reference['gradient_norm']**2),fixed_original_error_dark_delta_norm=float(np.linalg.norm(fixed-ref)),sequence=seq)
  if name=='keep':
   assert abs(row['global_head_gradient_norm']/reference['gradient_norm']-1)<.02,('Gradient reproduction',row)
   assert abs(row['dark_gradient_fraction']-reference['dark_gradient_fraction'])<.01,('Dark fraction reproduction',row)
   assert abs(row['dark_input_energy_fraction']-.7696534410623249)<.015,('Input energy reproduction',row)
  vectors=dict(fixed_error_dark=fixed.astype(np.float32),actual_dark=projected.astype(np.float32))
  if name=='keep':vectors['original_dark']=ref.astype(np.float32)
  save_npz(directory/'gradient_vectors.npz',**vectors);publish_frame(directory/'gradient_vectors.npz',dict(source_commit=commit,arm=name,targets=targets,original_reference='keep/gradient_vectors.npz'),uri)
  p=directory/'complete.json';p.write_text(json.dumps(row,indent=2)+'\n');finish_json(p,uri+'/complete.json')
  results[name]=row;print('ORIGIN_ARM_COMPLETE',name,json.dumps({k:row[k] for k in ['native_CE','head_RMS','dark_RMS','dark_input_energy_fraction','dark_gradient_fraction','fixed_original_error_aligned_reduction']}),flush=True);return row
 for name,targets in initial:arm(name,targets)
 candidates=[x for x in initial if x[0].startswith('body_')]
 selected=max(candidates,key=lambda x:abs(results[x[0]]['fixed_original_error_aligned_reduction']))
 for target in selected[1]:arm(f'single_before_block_{target+1:02d}',[target])
 result=dict(backend=m.get('backend','native_tpu'),backend_overrides={'attention':{'training':'flash','inference':'dot_product'}} if local_only else {},source_commit=commit,checkpoint=checkpoint,state_step=53001,label_step=53000,sequences=32,sequence_hashes=m['sequence_hashes'],reference_gradient=reference,selected_group=selected[0],selection_rule='Largest absolute fixed-original-error aligned gradient change among four body groups; every layer of that group then measured individually',arms=results,method='Parameter-only M-route reset to initialization current-only, preserving Q/K/V rows and all other parameters. Same original a03 capture graph, matched32-text inference. Recomputed errors and separately fixed original errors retained. Native hidden-vector differences from archival run recorded; baseline native CE, head gradient norm, Wc-dark gradient fraction and input energy must reproduce. No training and no full-QKV no-MUDD claim.')
 p=root/'complete.json';p.write_text(json.dumps(result,indent=2)+'\n');finish_json(p,dest+'/complete.json');print('ORIGIN_ANALYSIS_COMPLETE',selected[0],flush=True)
if __name__=='__main__':main()

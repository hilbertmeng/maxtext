"""Read-only native serial53k: normalization-aware gradient source attribution."""
import argparse,gc,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict
import max_utils
import mudd_spectral_capture as cap
import mudd_spectral_io as io
from mudd_spectral_model import serial_setup
from mudd_gradient_origin_capture import OriginCapture
from mudd_spectral_probe import save_npz,gs_json,publish

def rel(a,b):return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-20))
def main():
 p=argparse.ArgumentParser()
 for name in ['manifest','output','training-config']:p.add_argument('--'+name,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit']
 step=53000;info=m['models']['serial'];checkpoint=info['checkpoints'][str(step)];root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 dest=m['gcs_root']+'/serial/53000';parent=root/'parent';parent.mkdir(exist_ok=True)
 subprocess.run(['gsutil','-q','-m','rsync','-r',m['parent_artifact_root']+'/53000',str(parent)],check=True)
 done=gs_json(dest+'/complete.json')
 if done:
  assert done['source_commit']==commit and done['checkpoint']==checkpoint;print('ORIGIN_REUSED_COMPLETE',flush=True);return
 listed=subprocess.run(['gsutil','ls',dest+'/'],capture_output=True,text=True)
 if listed.returncode==0:subprocess.run(['gsutil','-q','-m','rsync','-r',dest,str(root)],check=True)
 elif not any(x in listed.stderr for x in ['No URLs matched','matched no objects','404','NotFound']):raise RuntimeError(listed.stderr[-400:])
 spectrum=root/'centered.npz';subprocess.run(['gsutil','-q','cp',m['parent_gcs_root']+'/head_spectra/serial_53000_vocabulary_centered.npz',str(spectrum)],check=True)
 with np.load(spectrum) as z:basis=jnp.asarray(z['basis'][:,:128])
 complete=json.loads((parent/'complete.json').read_text());assert complete['checkpoint']==checkpoint and complete['sequences']==32
 all_captured=all((root/f'sequence_{i:03d}.json').exists() for i in range(32))
 flat,identity=io.params(checkpoint,'serial',head_only=all_captured,workers=16);assert identity['step']==53001
 w,gamma,bias=io.flatten_head(flat,'serial');w=jnp.asarray(w);gamma=jnp.asarray(gamma);bias=jnp.asarray(bias)
 cfg,checked,mesh,model,rng,abstract=serial_setup(a.training_config,label='mudd_gradient_origin')
 assert cfg.mudd_full_history and not cfg.mudd_prenorm and not cfg.mudd_postnorm
 max_utils.cross_entropy_with_logits=cap.precise_cross_entropy
 def feature(params,x,y):
  observer=OriginCapture(gamma,bias,basis,32)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),nn.intercept_methods(observer.interceptor):
   native,col=model.apply(params,x,jnp.broadcast_to(jnp.arange(2048),x.shape),decoder_target_tokens=y,decoder_target_mask=jnp.ones_like(x),decoder_segment_ids=jnp.ones_like(x),enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},mutable=['intermediates'])
  f,_=observer.result(col['intermediates'])
  # Only small projections leave the forward, not the full per-layer D-vectors.
  relative=lambda a,b:jnp.sqrt(jnp.mean((a-b)**2)/jnp.maximum(jnp.mean(b*b),1e-20))
  return dict(h=f['h'],native_loss=native[0],global_components=f['normalized_global_components'][:,0],final_components=f['normalized_final_components'][:,0],raw_telescoping_error=f['raw_telescoping_error'],global_IG_error=relative(f['global_IG_sum'],f['global_IG_expected']),final_IG_error=relative(f['final_IG_sum'],f['final_IG_expected']),global_IG_sum=f['global_IG_sum'],global_IG_expected=f['global_IG_expected'],final_IG_sum=f['final_IG_sum'],final_IG_expected=f['final_IG_expected'])
 feature=jax.jit(feature)
 if not all_captured:
  assert identity['parameter_count']==2784962589
  params=jax.device_put({'params':unflatten_dict(flat)});del flat;gc.collect()
  for i in range(32):
   path=root/f'sequence_{i:03d}.npz'
   if path.with_suffix('.json').exists():
    marker=json.loads(path.with_suffix('.json').read_text());assert marker['source_commit']==commit and marker['input_hash']==m['sequence_hashes'][i];continue
   with np.load(parent/path.name) as z:x=jnp.asarray(z['inputs']);y=jnp.asarray(z['targets']);previous_h=z['h'].copy();previous_loss=z['native_loss'].copy()
   f={k:np.asarray(v,dtype=np.float32) for k,v in jax.device_get(feature(params,x,y)).items()}
   assert abs(f['native_loss'].mean()-previous_loss.mean())<.005,(i,'Native CE mismatch')
   assert rel(f['h'],previous_h)<.015,(i,'Native H mismatch')
   assert float(f['raw_telescoping_error'])<1e-5
   assert float(f['global_IG_error'])<.002 and float(f['final_IG_error'])<.002,(i,f['global_IG_error'],f['final_IG_error'])
   projected=np.asarray(f['h'],np.float32)@np.asarray(basis)
   for key in ['global_components','final_components']:assert rel(f[key].sum(0),projected[0])<2e-5
   save_npz(path,**f,targets=np.asarray(y),inputs=np.asarray(x))
   publish(path,dict(source_commit=commit,checkpoint=checkpoint,state_step=53001,family='serial',label_step=53000,sequence=i,input_hash=m['sequence_hashes'][i],sequence_hashes=m['sequence_hashes'],native_CE_difference=float(f['native_loss'].mean()-previous_loss.mean()),head_relative_RMS_difference=rel(f['h'],previous_h),global_IG_error=float(f['global_IG_error']),final_IG_error=float(f['final_IG_error']),raw_telescoping_error=float(f['raw_telescoping_error'])),dest)
   print('ORIGIN_CAPTURE_COMPLETE',i,flush=True)
  del params;gc.collect()
 else:del flat
 # Group1: normalized frozen ordinary trajectory + IG of each M-route increment.
 # Group2: normalized last block + final router historical-source IG.
 mudd_names=[f'mudd_before_block_{i+1:02d}' for i in range(1,32)]+['mudd_final_readout']
 global_names=['frozen_ordinary_path_normalized']+mudd_names+['native_LN_and_quadrature_remainder']
 history_names=['history_embedding']+[f'history_block_{i:02d}' for i in range(1,33)]+['final_routing_numerical_remainder']
 final_names=['last_block_through_own_LN']+history_names+['native_LN_and_quadrature_remainder']
 names=global_names+final_names;groups={'global':list(range(len(global_names))),'final':list(range(len(global_names),len(names)))}
 @jax.jit
 def errors(h,y):
  hf=h[0].astype(jnp.float32);wf=w.astype(jnp.bfloat16).astype(jnp.float32)
  z=jnp.einsum('td,dv->tv',hf,wf,precision=jax.lax.Precision.HIGHEST);e=jax.nn.softmax(z).at[jnp.arange(2048),y[0]].add(-1)
  g=jnp.einsum('td,tv->dv',hf,e,precision=jax.lax.Precision.HIGHEST)/2048
  return e,g,cap.cross_entropy(z,y[0])
 @jax.jit
 def chunk_gradient(parts,e):return jnp.einsum('itk,tv->ikv',parts,e,precision=jax.lax.Precision.HIGHEST)/2048
 chunk=4;sum_grad=np.zeros((len(names),128,50432),np.float64);full=np.zeros((2560,50432),np.float64);sequence=[]
 for i in range(32):
  path=root/f'sequence_{i:03d}.npz';marker=json.loads(path.with_suffix('.json').read_text());assert marker['source_commit']==commit and marker['input_hash']==m['sequence_hashes'][i]
  with np.load(path) as z:h=jnp.asarray(z['h'],jnp.bfloat16);y=jnp.asarray(z['targets']);parts=np.concatenate([z['global_components'],z['final_components']]);native=float(z['native_loss'].mean())
  e,g,ce=errors(h,y);full+=np.asarray(g)/32;ce=float(ce.mean());sequence.append(dict(index=i,native_CE=native,FP32_CE=ce))
  for j in range(0,len(names),chunk):
   v=parts[j:j+chunk];n=len(v)
   if n<chunk:v=np.pad(v,((0,chunk-n),(0,0),(0,0)))
   grad=chunk_gradient(jnp.asarray(v),e);sum_grad[j:j+n]+=np.asarray(grad)[:n]/32
  print('ORIGIN_HEAD_CASE',i,flush=True)
 projected=np.asarray(basis,np.float64).T@full;dark2=float(np.sum(projected**2));g2=float(np.sum(full*full))
 gram=sum_grad.reshape(len(names),-1)@sum_grad.reshape(len(names),-1).T
 group_rows={};closure={}
 for group,ids in groups.items():
  aggregate=sum_grad[ids].sum(0);closure[group]=rel(aggregate,projected);assert closure[group]<1e-4
  allocated=(sum_grad[ids].reshape(len(ids),-1)@projected.reshape(-1))
  group_rows[group]=[dict(name=names[idx],gradient_norm=float(np.sqrt(gram[idx,idx])),signed_dark_energy=float(allocated[k]),signed_share_of_dark=float(allocated[k]/dark2),signed_share_of_total_head=float(allocated[k]/g2)) for k,idx in enumerate(ids)]
 result=dict(source_commit=commit,checkpoint=checkpoint,state_step=53001,family='serial',label_step=53000,sequences=32,sequence_hashes=m['sequence_hashes'],names=names,groups=group_rows,group_indices=groups,group_gradient_closure=closure,global_head_gradient_norm=float(np.sqrt(g2)),dark_gradient_energy=dark2,dark_gradient_fraction=dark2/g2,sequence=sequence,scope='Fixed-error conditional source attribution. All-m path integrates final LayerNorm while ordinary block outputs stay frozen; it is not a rerun of a no-MUDD network and includes indirect QKV-MUDD effects in ordinary updates. Final path integrates final-LN effect of historical terms of final router. Cross terms included by <Gi,Gdark>. Negative or >100% shares allowed.')
 f=root/'origin_gradient_gram.npz';save_npz(f,gram=gram)
 publish(f,dict(source_commit=commit,checkpoint=checkpoint,names=names),dest)
 marker=root/'complete.json';marker.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');subprocess.run(['gsutil','-q','cp',str(marker),dest+'/complete.json'],check=True)
 print('ORIGIN_ANALYSIS_COMPLETE',json.dumps(dict(gradient_norm=result['global_head_gradient_norm'],dark_fraction=result['dark_gradient_fraction'],closure=closure)),flush=True)
if __name__=='__main__':main()

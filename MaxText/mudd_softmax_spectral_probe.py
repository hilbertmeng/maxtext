"""Head-only follow-up: remove vocabulary-common shifts before defining dark."""
import argparse,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import mudd_spectral_capture as cap
import mudd_spectral_io as io
from mudd_spectral_probe import save_npz,gs_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit']
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 grad_fn=jax.jit(cap.readout_gradient);native_ce=jax.jit(cap.native_head_ce)
 @jax.jit
 def precision_stats(h,w,y):
  z=jnp.einsum('btd,dv->btv',h.astype(jnp.float32),w.astype(jnp.bfloat16).astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
  zc=cap.centered(z);q=z.astype(jnp.bfloat16).astype(jnp.float32);qc=zc.astype(jnp.bfloat16).astype(jnp.float32)
  # Same exact FP32 logits, translated before the final BF16 rounding. This
  # isolates common-shift quantization from changing W or upstream activations.
  losses=jnp.stack([cap.cross_entropy(x,y) for x in [z,q,qc]])
  noise=cap.centered(q-z);centered_noise=cap.centered(qc-zc)
  return losses,jnp.stack([jnp.mean(z,axis=-1),jnp.mean(noise*noise,axis=-1),jnp.mean(centered_noise*centered_noise,axis=-1),jnp.mean(zc*zc,axis=-1)])
 @jax.jit
 def geometry(h,u,axis):
  h=h.astype(jnp.float32);v=jnp.einsum('btd,dk->btk',h,u,precision=jax.lax.Precision.HIGHEST)
  bands=jnp.sum(v.reshape(*v.shape[:-1],20,128)**2,axis=-1)/h.shape[-1]
  total=jnp.mean(h*h,axis=-1);common=jnp.einsum('btd,d->bt',h,axis,precision=jax.lax.Precision.HIGHEST)**2/h.shape[-1]
  projected=jnp.einsum('btk,dk->btd',v[...,:128],u[:,:128],precision=jax.lax.Precision.HIGHEST)
  return bands,total,common,(h-projected).astype(jnp.bfloat16)
 for family,info in m['models'].items():
  for step in info['steps']:
   dest=m['gcs_root']+f'/softmax/{family}/{step}';folder=root/family/str(step);folder.mkdir(parents=True,exist_ok=True)
   done=gs_json(dest+'/complete.json')
   if done is not None:
    assert done['source_commit']==commit and done['checkpoint']==info['checkpoints'][str(step)]
    print('SOFTMAX_REUSED_COMPLETE',family,step,flush=True);continue
   origin=info.get('artifact_root',m['gcs_root']+'/'+family)+'/'+str(step)
   subprocess.run(['gsutil','-q','-m','rsync','-r',origin,str(folder)],check=True)
   source=json.loads((folder/'complete.json').read_text());assert source['source_commit']==info['diagnostic_commit'] and source['sequence_hashes']==m['sequence_hashes']
   spectrum=folder/'vocabulary_centered.npz';subprocess.run(['gsutil','-q','cp',m['gcs_root']+f'/head_spectra/{family}_{step}_vocabulary_centered.npz',str(spectrum)],check=True)
   with np.load(spectrum) as z:
    u=jnp.asarray(z['basis']);mu=z['vocabulary_mean'];axis=jnp.asarray(mu/np.linalg.norm(mu))
   flat,identity=io.params(info['checkpoints'][str(step)],family,True,workers=3)
   w,_,_=io.flatten_head(flat,family);w=jnp.asarray(w);del flat
   bands=[];totals=[];commons=[];ces=[];native=[];summaries=[];precision_losses=[];precision_geometry=[];sums=[None,None]
   for i in range(32):
    marker=json.loads((folder/f'sequence_{i:03d}.json').read_text());assert marker['source_commit']==info['diagnostic_commit'] and marker['input_hash']==m['sequence_hashes'][i]
    with np.load(folder/f'sequence_{i:03d}.npz') as z:
     h=jnp.asarray(z['h'],jnp.bfloat16);y=jnp.asarray(z['targets']);tokens=np.concatenate([z['inputs'][0],z['targets'][0,-1:]])
     assert hashlib.sha256(tokens.astype(np.int32).tobytes()).hexdigest()==m['sequence_hashes'][i]
    band,total,common,filtered=geometry(h,u,axis);bands.append(np.asarray(band));totals.append(np.asarray(total));commons.append(np.asarray(common))
    pl,pg=precision_stats(h,w,y);precision_losses.append(np.asarray(pl));precision_geometry.append(np.asarray(pg))
    ci=[];ni=[];si=[]
    for v,hv in enumerate([h,filtered]):
     g,ce,summary=grad_fn(hv,w,y,u[:,:128]);sums[v]=g/32 if sums[v] is None else sums[v]+g/32
     ci.append(np.asarray(ce));ni.append(np.asarray(native_ce(hv,w,y)));si.append(np.asarray(summary))
    ces.append(ci);native.append(ni);summaries.append(si)
    print('SOFTMAX_HEAD_CASE',family,step,i,flush=True)
   global_stats=[]
   for g in sums:
    gd=jnp.einsum('dk,dv->kv',u[:,:128],g,precision=jax.lax.Precision.HIGHEST)
    ga=jnp.einsum('d,dv->v',axis,g,precision=jax.lax.Precision.HIGHEST)
    g2=float(jnp.sum(g*g));d2=float(jnp.sum(gd*gd));a2=float(jnp.sum(ga*ga))
    global_stats.append(dict(gradient_norm=float(np.sqrt(g2)),dark_gradient_energy=d2,dark_gradient_fraction=d2/g2,common_axis_gradient_energy=a2,common_axis_gradient_fraction=a2/g2))
   assert abs(global_stats[0]['gradient_norm']/source['global_head_gradients']['keep']['gradient_norm']-1)<1e-5
   f=folder/'softmax_stats.npz';save_npz(f,spectral_energy=np.stack(bands)[:,0],head_power=np.stack(totals)[:,0],common_axis_power=np.stack(commons)[:,0],native_CE=np.asarray(native),FP32_CE=np.asarray(ces),head_summary=np.asarray(summaries),precision_losses=np.asarray(precision_losses),precision_geometry=np.asarray(precision_geometry))
   result=dict(source_commit=commit,capture_commit=source['source_commit'],family=family,label_step=step,state_step=identity['step'],checkpoint=identity['checkpoint'],capture_root=origin,sequence_hashes=m['sequence_hashes'],sequences=32,variants=['keep','remove_vocabulary_centered_dark5'],global_head_gradients=global_stats,scope='Exploratory softmax-centered W spectrum, separate from original paper uncentered W spectrum.')
   marker=folder/'softmax_complete.json';marker.write_text(json.dumps(result,indent=2)+'\n')
   subprocess.run(['gsutil','-q','cp',str(f),dest+'/softmax_stats.npz'],check=True);subprocess.run(['gsutil','-q','cp',str(marker),dest+'/complete.json'],check=True)
   print('SOFTMAX_CHECKPOINT_COMPLETE',family,step,json.dumps(global_stats),flush=True)
   del sums,w,u;gc.collect()
if __name__=='__main__':main()

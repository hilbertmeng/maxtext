"""Recompute frozen-head metrics from immutable captures using strict BF16 weights."""
import argparse,gc,hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import head_spectral_diagnostic as hd

def save(path,**arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
  for key,value in arrays.items():
   with archive.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(value),allow_pickle=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==m['commit']
 D=Path(m['local_DATA']);C=Path(m['capture_DATA']);(D/'results').mkdir(parents=True,exist_ok=True)
 readout=jax.jit(hd.FP32_readout)
 @jax.jit
 def stats(h,u,uc,mu,z):
  hf=h[0].astype(jnp.float32);pr=jnp.einsum('td,dk->tk',hf,u,precision=jax.lax.Precision.HIGHEST);pc=jnp.einsum('td,dk->tk',hf,uc,precision=jax.lax.Precision.HIGHEST)
  centered=z-z.mean(-1,keepdims=True);axis=mu/jnp.linalg.norm(mu)
  return dict(head_power=jnp.mean(hf*hf,-1),raw_bands=jnp.sum(pr.reshape(2048,20,128)**2,-1)/2560,centered_bands=jnp.sum(pc.reshape(2048,20,128)**2,-1)/2560,common_axis_power=jnp.einsum('td,d->t',hf,axis,precision=jax.lax.Precision.HIGHEST)**2/2560,common_logit_offset=z.mean(-1),centered_logit_power=jnp.mean(centered*centered,-1),quantized_logits=hd.BF16_round(z),center_quantized_logits=hd.BF16_round(centered))
 @jax.jit
 def CE(z,y):return -jnp.take_along_axis(jax.nn.log_softmax(z),y[0,:,None],-1)[:,0]
 for step in m['steps']:
  folder=D/'results'/str(step);folder.mkdir(exist_ok=True);capture=C/'results'/str(step);original=json.loads((capture/'complete.json').read_text());assert original['source_commit']==m['capture_commit'] and original['checkpoint']==m['checkpoints'][str(step)] and original['sequence_hashes']==m['cohort_hashes'] and original['sequences']==32
  done=folder/'complete.json'
  if done.exists():assert json.loads(done.read_text())['source_commit']==m['commit'];continue
  with np.load(D/'head_spectra'/f'{step}.npz') as spectrum:u=jnp.asarray(spectrum['basis']);uc=jnp.asarray(spectrum['centered_basis']);mu=jnp.asarray(spectrum['vocabulary_mean'])
  effective=np.load(D/'head_spectra'/f'{step}_effective_head.npy',mmap_mode='r');assert effective.shape==(2560,50304) and not np.any(effective.view(np.uint32)&np.uint32(65535));w=jnp.asarray(effective,jnp.bfloat16)
  total=np.zeros((2560,50304),np.float64);seq=[]
  for i in range(32):
   name=f'sequence_{i:03d}';meta=json.loads((capture/(name+'.json')).read_text());assert meta['source_commit']==m['capture_commit'] and meta['input_hash']==m['cohort_hashes'][i] and meta['checkpoint']==m['checkpoints'][str(step)] and meta['native_CE_replay_difference']<.005 and meta['final_norm_relative_RMS']<.015
   for suffix in ['.npz','.json']:
    link=folder/(name+suffix)
    if not link.exists():link.symlink_to(capture/(name+suffix))
   with np.load(capture/(name+'.npz')) as z:
    hx=z['h'];assert hx.shape==(1,2048,2560) and not np.any(hx.view(np.uint32)&np.uint32(65535));h=jnp.asarray(hx,jnp.bfloat16);y=jnp.asarray(z['targets']);native=float(z['native_loss'].mean());raw_power=np.mean(z['raw'].astype(np.float64)**2,-1)
    tokens=np.concatenate([z['inputs'][0],z['targets'][0,-1:]]).astype(np.int32);assert hashlib.sha256(tokens.tobytes()).hexdigest()==m['cohort_hashes'][i]
   g,ce,z=readout(h,w,y);total+=np.asarray(g)/32;s=stats(h,u,uc,mu,z);q=s.pop('quantized_logits');qc=s.pop('center_quantized_logits');arrays={k:np.asarray(v) for k,v in s.items()};arrays.update(FP32_CE=np.asarray(ce),raw_power=raw_power,precision_CE=np.stack([np.asarray(ce),np.asarray(CE(q,y)),np.asarray(CE(qc,y))]))
   save(folder/f'head_{i:03d}.npz',**arrays);marker=dict(source_commit=m['commit'],capture_commit=m['capture_commit'],checkpoint=original['checkpoint'],label_step=step,sequence=i,input_hash=m['cohort_hashes'][i],weight_rounding='Materialized BF16 input and explicit integer RNE, no eliminable conversion chain');(folder/f'head_{i:03d}.json').write_text(json.dumps(marker,indent=2)+'\n');seq.append(dict(index=i,native_CE=native,FP32_CE=float(ce.mean())));print('STRICT_EFFECTIVE_HEAD_CASE',step,i,flush=True)
  g2=float(np.sum(total**2));gd=np.asarray(u[:,:128],np.float64).T@total;gc_=np.asarray(uc[:,:128],np.float64).T@total;axis=np.asarray(mu,np.float64);axis/=np.linalg.norm(axis)
  result=dict(original);result.update(source_commit=m['commit'],capture_commit=m['capture_commit'],head_analysis_backend='cuda',head_analysis_device=jax.devices()[0].device_kind,head_weight_rounding='Actual BF16 forward-effective W, materialized before FP32 readout; explicit integer RNE also enforced in helper',global_head_gradient_norm=float(np.sqrt(g2)),raw_dark_gradient_energy=float(np.sum(gd*gd)),centered_dark_gradient_energy=float(np.sum(gc_*gc_)),common_axis_gradient_energy=float(np.sum((axis@total)**2)),sequence=seq,scope='Immutable original network captures; strict effective BF16 W with FP32 head accumulation and CE; same32x2048; no retraining or full-network backward.')
  done.write_text(json.dumps(result,indent=2)+'\n');print('STRICT_EFFECTIVE_HEAD_CHECKPOINT_COMPLETE',step,result['global_head_gradient_norm'],result['centered_dark_gradient_energy']/g2,flush=True);del w,total;gc.collect()

if __name__=='__main__':main()

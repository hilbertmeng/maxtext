"""Replay original chunked head backward, preserving training CE custom VJP."""
import argparse,gc,json,time
from functools import partial
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import max_utils
import head_spectral_diagnostic as hd

def head_loss(w,h,y,mode='training',chunk_size=4096):
 # Cast once, as OutputHead.setup does: both chunks share this BF16 value.
 wb=w.astype(jnp.bfloat16);total=jnp.float32(0)
 for start in range(0,h.shape[1],chunk_size):
  logits=jnp.tensordot(h[:,start:start+chunk_size],wb,axes=((-1,),(0,)))
  targets=jax.nn.one_hot(y[:,start:start+chunk_size],w.shape[1])
  ce=(max_utils.cross_entropy_with_logits if mode=='training' else hd.precise_cross_entropy)(logits,targets,0.)[0]
  total=total+ce.sum()
 return total/y.size

def main():
 R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(R/'arms_gpu.json'));p.add_argument('--output',default=str(R/'head_precision_controls.json'));a=p.parse_args();m=json.loads(Path(a.manifest).read_text());D=Path(m['local_DATA'])
 assert jax.devices()[0].platform=='gpu' and len(jax.devices())==1
 out=dict(source_commit=m['commit'],capture_commit=m.get('capture_commit',m['commit']),input_capture_backend=m.get('backend'),original_training_source=m['training_commit'],backend='cuda_math_reference_head',weight_values='Checkpoint BF16 forward-effective values, materialized and strict integer RNE in FP32 readout',configured_loss_chunk_size=4096,scope='Actual original CE custom VJP vs FP32 CE, BF16 head matmul and original4096-token configured chunks (one effective2048-token chunk). Same frozen H, 32 sequences. CUDA rather than TPU backward, no full-network backward. Diagnostic per-text gradients averaged across32, formal training globalbatch1024.',checkpoints={})
 for step in m['steps']:
  folder=D/'results'/str(step);summary=json.loads((folder/'complete.json').read_text())
  with np.load(D/'head_spectra'/f'{step}.npz') as z:u=z['basis'][:,:128].astype(np.float64);uc=z['centered_basis'][:,:128].astype(np.float64);mu=z['vocabulary_mean'].astype(np.float64)
  axis=mu/np.linalg.norm(mu);w=jnp.asarray(np.load(D/'head_spectra'/f'{step}_effective_head.npy'))
  modes={}
  for mode in ['training','FP32_CE_BF16_logits','FP32_readout']:
   if mode=='FP32_readout':
    @jax.jit
    def fn(w,h,y):
     g,ce,_=hd.FP32_readout(h,w,y)
     return ce.mean(),g
   else:fn=jax.jit(jax.value_and_grad(partial(head_loss,mode=mode)))
   full=np.zeros(w.shape,np.float64);loss=[]
   for i in range(32):
    with np.load(folder/f'sequence_{i:03d}.npz') as z:h=jnp.asarray(z['h'],jnp.bfloat16);y=jnp.asarray(z['targets']);native=float(z['native_loss'].mean())
    ce,g=fn(w,h,y);value=float(ce);full+=np.asarray(g,np.float32)/32;loss.append(value)
    if mode=='FP32_CE_BF16_logits':assert abs(value-native)<.005,(step,i,value,native)
    print('HEAD_PRECISION_CONTROL',step,mode,i,value,flush=True)
   g2=float(np.sum(full**2));pg=uc.T@full;pr=u.T@full;gm=full.mean(-1)
   result=dict(loss=float(np.mean(loss)),per_sequence_loss=loss,gradient_norm=float(np.sqrt(g2)),centered_dark_gradient_energy=float(np.sum(pg*pg)),raw_dark_gradient_energy=float(np.sum(pr*pr)),common_input_axis_gradient_energy=float(np.sum((axis@full)**2)),vocabulary_constant_gradient_energy=float(w.shape[1]*np.sum(gm*gm)))
   result['centered_dark_fraction']=result['centered_dark_gradient_energy']/g2;result['raw_dark_fraction']=result['raw_dark_gradient_energy']/g2;result['vocabulary_constant_gradient_fraction']=result['vocabulary_constant_gradient_energy']/g2
   result['centered_first_ranks_gradient_fraction']={str(n):float(np.sum(pg[:n]**2)/g2) for n in [1,2,4,16,128]}
   if mode=='FP32_readout':
    norm_error=abs(result['gradient_norm']/summary['global_head_gradient_norm']-1);energy_error=abs(result['centered_dark_gradient_energy']/summary['centered_dark_gradient_energy']-1)
    cross_device=m.get('backend')=='native_tpu' and not m.get('head_analysis_backend','').startswith('cuda')
    norm_tolerance=1e-3 if cross_device else 1e-5;energy_tolerance=1e-3 if cross_device else 2e-5
    result['reference_reproduction']=dict(norm_relative_error=norm_error,weak_energy_relative_error=energy_error,norm_tolerance=norm_tolerance,energy_tolerance=energy_tolerance,cross_device=cross_device)
    (folder/'head_precision_reproduction_check.json').write_text(json.dumps(result,indent=2)+'\n')
    assert norm_error<norm_tolerance and energy_error<energy_tolerance,(step,result['reference_reproduction'])
   np.savez_compressed(folder/f'head_precision_{mode}_vectors.npz',centered_dark=pg.astype(np.float32),raw_dark=pr.astype(np.float32),vocabulary_mean_gradient=gm.astype(np.float32))
   modes[mode]=result;out['checkpoints'][str(step)]=modes;Path(a.output).write_text(json.dumps(out,indent=2)+'\n');del full;gc.collect();print('HEAD_PRECISION_MODE_COMPLETE',step,mode,result['gradient_norm'],result['centered_dark_fraction'],flush=True)
  modes['FP32_readout_reference']=dict(gradient_norm=summary['global_head_gradient_norm'],centered_dark_gradient_energy=summary['centered_dark_gradient_energy'],centered_dark_fraction=summary['centered_dark_gradient_energy']/summary['global_head_gradient_norm']**2)
  out['checkpoints'][str(step)]=modes;Path(a.output).write_text(json.dumps(out,indent=2)+'\n');del w;gc.collect()
 print('HEAD_PRECISION_CONTROLS_COMPLETE',flush=True)

if __name__=='__main__':main()

"""Old MUDD43k/53k: change only the frozen lm_head probability computation."""
import argparse,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import max_utils
import mudd_spectral_io as io
from mudd_spectral_probe import save_npz
from mudd_router_ablation_probe import read_blob,write_blob,json_blob

@jax.jit
def project(h,w):
 assert h.dtype==w.dtype==jnp.bfloat16
 return jnp.tensordot(h,w,axes=((-1,),(0,)))
@jax.jit
def pax_fp32(z,y):
 lp=jax.nn.log_softmax(z.astype(jnp.float32));target=jax.nn.one_hot(y,z.shape[-1],dtype=jnp.float32)
 return -jnp.sum(lp*target,-1,dtype=jnp.float32)
@jax.jit
def pax_bf16(z,y):
 lp=jax.nn.log_softmax(z);assert lp.dtype==jnp.bfloat16
 target=jax.nn.one_hot(y,z.shape[-1],dtype=jnp.float32)
 return -jnp.sum(lp*target,-1,dtype=jnp.float32)
@jax.jit
def maxtext_train_bf16(z,y):
 # Calling the registered custom-VJP forward directly reproduces train-time
 # CE, rather than its different primal-only evaluation expression.
 return max_utils._cross_entropy_with_logits_fwd(z,jax.nn.one_hot(y,z.shape[-1],dtype=jnp.float32),0.)[0][0]
@jax.jit
def bf16_log_probs(z):
 lp=jax.nn.log_softmax(z);assert lp.dtype==jnp.bfloat16;return lp
@jax.jit
def strict_loss_and_renormalized(lp,y):
 assert lp.dtype==jnp.bfloat16
 l=lp.astype(jnp.float32);ce=-jnp.take_along_axis(l,y[...,None],-1)[...,0];normalizer=jax.scipy.special.logsumexp(l,-1)
 return ce,ce+normalizer,normalizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert commit==m['commit'];root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 done=json_blob(m['gcs_root']+'/complete.json')
 if done:assert done['source_commit']==commit and done['cases']==64;print('SOFTMAX_REUSED_COMPLETE',flush=True);return
 results={}
 for step in m['steps']:
  uri=m['gcs_root']+f'/old/{step}';folder=root/str(step);folder.mkdir(exist_ok=True);done=json_blob(uri+'/complete.json')
  if done:assert done['source_commit']==commit;results[str(step)]=done;continue
  flat,identity=io.params(m['checkpoints'][str(step)],'old',True,workers=3,cache_dir=str(root/'params'/str(step)));assert identity['step']==step
  w0=io.flatten_head(flat,'old')[0];assert w0.shape==(2560,50432);w=jnp.asarray(w0,jnp.bfloat16);del flat;rows=[]
  for i in range(32):
   stem=f'sequence_{i:03d}';marker=json_blob(uri+'/'+stem+'.json')
   if marker:assert marker['source_commit']==commit and marker['input_hash']==m['cohort_hashes'][i];rows.append(marker);continue
   source=m['capture_root']+f'/{step}/{stem}'
   def capture_bytes(ext):
    if m.get('local_capture_root'):return Path(m['local_capture_root'],str(step),stem+ext).read_bytes()
    return read_blob(source+ext)
   meta=json.loads(capture_bytes('.json'));assert meta['source_commit']==m['capture_commit'] and meta['checkpoint']==m['checkpoints'][str(step)] and meta['input_hash']==m['cohort_hashes'][i]
   import io as pyio
   with np.load(pyio.BytesIO(capture_bytes('.npz'))) as z:
    h=jnp.asarray(z['h'],jnp.bfloat16);x=z['inputs'].copy();y0=z['targets'].copy();native=z['native_loss'].reshape(-1).copy()
   tokens=np.concatenate([x[0],y0[0,-1:]]).astype(np.int32);assert hashlib.sha256(tokens.tobytes()).hexdigest()==m['cohort_hashes'][i];y=jnp.asarray(y0)
   logits=project(h,w);logits.block_until_ready();assert logits.dtype==jnp.bfloat16
   fp=pax_fp32(logits,y);bf=pax_bf16(logits,y);mt=maxtext_train_bf16(logits,y);lp=bf16_log_probs(logits);lp.block_until_ready();strict,renorm,norm=strict_loss_and_renormalized(lp,y)
   arrays=dict(inputs=x,targets=y0,archive_native_CE=native,CE_fp32=np.asarray(fp).reshape(-1),CE_pax_bf16=np.asarray(bf).reshape(-1),CE_maxtext_train=np.asarray(mt).reshape(-1),CE_strict_bf16=np.asarray(strict).reshape(-1),CE_bf16_renormalized=np.asarray(renorm).reshape(-1),bf16_logprob_log_normalizer=np.asarray(norm).reshape(-1))
   assert all(np.isfinite(v).all() for v in arrays.values());drift=float(arrays['CE_fp32'].mean()-native.mean());assert abs(drift)<.005,(step,i,'Frozen projection replay mismatch',drift)
   path=folder/(stem+'.npz');save_npz(path,**arrays);write_blob(path,uri+'/'+path.name)
   marker=dict(source_commit=commit,checkpoint=identity['checkpoint'],state_step=step,label_step=step,sequence=i,input_hash=m['cohort_hashes'][i],capture_commit=m['capture_commit'],projection_dtype='bfloat16_materialized',softmax_baseline_dtype='float32',softmax_intervention_dtype='bfloat16',CE_reduction_dtype='float32',native_replay_difference=drift,means={k:float(v.mean()) for k,v in arrays.items() if k not in ['inputs','targets']});path=folder/(stem+'.json');path.write_text(json.dumps(marker,indent=2)+'\n');write_blob(path,uri+'/'+path.name);rows.append(marker);print('SOFTMAX_CASE_COMPLETE',step,i,marker['means']['CE_pax_bf16']-marker['means']['CE_fp32'],flush=True)
  drift=float(np.mean([r['native_replay_difference'] for r in rows]));assert abs(drift)<1e-4,('Cohort native projection replay drift',step,drift)
  result=dict(source_commit=commit,checkpoint=identity['checkpoint'],state_step=step,sequences=32,sequence_hashes=m['cohort_hashes'],sequence=rows,cohort_replay_drift=drift);path=folder/'complete.json';path.write_text(json.dumps(result,indent=2)+'\n');write_blob(path,uri+'/complete.json');results[str(step)]=result;del w;gc.collect();print('SOFTMAX_CHECKPOINT_COMPLETE',step,flush=True)
 result=dict(source_commit=commit,cases=64,steps=m['steps'],checkpoints=results,jax_version=jax.__version__,device_kind=jax.devices()[0].device_kind,scope='Immutable old native H, materialized unchanged BF16 output projection. Only log-softmax/CE precision changed; no training. Standardized old/new FP32-CE gap uses exactly the same32 texts.')
 path=root/'complete.json';path.write_text(json.dumps(result,indent=2)+'\n');write_blob(path,m['gcs_root']+'/complete.json');print('SOFTMAX_ALL_COMPLETE',flush=True)
if __name__=='__main__':main()

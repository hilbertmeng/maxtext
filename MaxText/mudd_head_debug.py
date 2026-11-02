"""Read-only fixed-H head algebra/precision diagnostic for a failed capture."""
import fcntl,json,subprocess
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import mudd_spectral_capture as cap
import mudd_spectral_io as io

with open('/tmp/mudspec_job/job.lock','a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=Path('/tmp/mudd_spectral/results/serial/33000/failed_capture_031.npz')
 with np.load(p) as z:
  h=jnp.asarray(z['h'],jnp.bfloat16);y=jnp.asarray(z['targets']);raw=jnp.asarray(z['raw'],jnp.float32)
  actual=float(z['native_loss'].mean());old=float(z['captured_native_loss'].mean())
 m=json.loads(Path('/tmp/mudspec_job/arms.json').read_text());flat,_=io.params(m['models']['serial']['checkpoints']['33000'],'serial',True,3)
 w,g,b=io.flatten_head(flat,'serial');w=jnp.asarray(w);g=jnp.asarray(g);b=jnp.asarray(b)
 def native_style(h,w,y):return cap.precise_cross_entropy(jnp.tensordot(h,w.astype(jnp.bfloat16),axes=((-1,),(0,))),jax.nn.one_hot(y,50432))[0]
 def einsum_style(h,w,y):return cap.precise_cross_entropy(jnp.einsum('btd,dv->btv',h,w.astype(jnp.bfloat16)),jax.nn.one_hot(y,50432))[0]
 def explicit_logits(h,w,y):
  z=jnp.tensordot(h,w.astype(jnp.bfloat16),axes=((-1,),(0,)))
  return cap.precise_cross_entropy(z,jax.nn.one_hot(y,50432))[0],z
 def fp32(h,w,y):return cap.cross_entropy(jnp.einsum('btd,dv->btv',h.astype(jnp.float32),w.astype(jnp.bfloat16).astype(jnp.float32),precision=jax.lax.Precision.HIGHEST),y)
 def full_w(h,w,y):return cap.cross_entropy(jnp.einsum('btd,dv->btv',h.astype(jnp.float32),w,precision=jax.lax.Precision.HIGHEST),y)
 r=dict(actual_native_CE=actual,standalone_replay_CE=old,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 r['target_min']=int(y.min());r['target_max']=int(y.max());assert int(y.min())>=0 and int(y.max())<50432
 for name,fn in [('native_style',native_style),('einsum_logsoftmax',einsum_style),('lse_gather',lambda h,w,y:cap.cross_entropy(jnp.einsum('btd,dv->btv',h,w.astype(jnp.bfloat16)),y)),('FP32_dot_BF16_W',fp32),('FP32_dot_original_W',full_w)]:
  v=jax.jit(fn)(h,w,y);r[name]=float(v.mean());print(name,r[name],flush=True)
 v,z=jax.jit(explicit_logits)(h,w,y);r['native_style_with_logits_output']=float(v.mean());r['logits_dtype']=str(z.dtype);r['logit_mean_RMS']=float(jnp.sqrt(jnp.mean(jnp.mean(z.astype(jnp.float32),axis=-1)**2)))
 materialized=jnp.asarray(np.array(z))
 r['materialized_logits_logsoftmax']=float(jax.jit(lambda z,y:cap.precise_cross_entropy(z,jax.nn.one_hot(y,50432))[0])(materialized,y).mean())
 r['materialized_logits_lse_gather']=float(jax.jit(cap.cross_entropy)(materialized,y).mean())
 norm=raw-raw.mean(-1,keepdims=True);norm=norm*jax.lax.rsqrt(jnp.mean(norm*norm,axis=-1,keepdims=True)+1e-5);hf=norm*g+b
 r['FP32_unrounded_norm_BF16_W']=float(jax.jit(fp32)(hf,w,y).mean());r['FP32_unrounded_norm_original_W']=float(jax.jit(full_w)(hf,w,y).mean())
 out=Path('/tmp/mudspec_job/head_debug.json');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
 subprocess.run(['gsutil','-q','cp',str(out),m['gcs_root']+'/head_debug.json'],check=True)

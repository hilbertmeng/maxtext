"""Effective BF16 vocabulary-centered head bases, read-only checkpoint loads."""
from pathlib import Path
import json,os,subprocess,sys,time,io
import numpy as np,ml_dtypes
from scipy.linalg import eigh
R=Path(__file__).resolve().parent;m=json.loads((R/'arms.json').read_text());H=Path(m['local_DATA']).parent/'head_spectra';H.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,m['worktree']+'/MaxText');import head_spectral_diagnostic as hd
for step in m['steps']:
 start=time.time();marker=H/f'{step}.json';target=m['own_basis_root']+f'/{step}.npy'
 if marker.exists():continue
 print('HEAD_RESTORE',step,flush=True)
 os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['/home/mengqy/google-cloud-sdk/bin/gcloud','auth','print-access-token'],text=True).strip()
 flat,identity=hd.restore(m['checkpoints'][str(step)],head_only=True);assert identity['state_step']==step+1
 w=flat[('decoder','lm_head','logits_dense')];g=flat[('decoder','lm_head','decoder_norm','scale')];b=flat[('decoder','lm_head','decoder_norm','bias')]
 wf=np.asarray(w,ml_dtypes.bfloat16).astype(np.float64);assert wf.shape==(2560,50304);mu=wf.mean(1);wf-=mu[:,None];gram=wf@wf.T;del wf
 e,u=eigh(gram,driver='evd',check_finite=False);assert e.min()>0
 basis=u[:,:128].astype(np.float32);path=H/f'{step}.npy';np.save(path,basis)
 np.savez_compressed(H/f'{step}.npz',centered_singular=np.sqrt(e),centered_basis=u.astype(np.float32),gamma=g,bias=b,vocabulary_mean=mu)
 subprocess.run(['/home/mengqy/google-cloud-sdk/bin/gsutil','-q','cp',str(path),target],check=True)
 row=dict(checkpoint=identity['checkpoint'],state_step=identity['state_step'],label_step=step,gamma_RMS=float(np.sqrt(np.mean(g*g))),bias_RMS=float(np.sqrt(np.mean(b*b))),sigma_min=float(np.sqrt(e[0])),sigma_max=float(np.sqrt(e[-1])),seconds=time.time()-start,basis=target)
 marker.write_text(json.dumps(row,indent=2)+'\n');print('HEAD_BASIS_READY',row,flush=True)

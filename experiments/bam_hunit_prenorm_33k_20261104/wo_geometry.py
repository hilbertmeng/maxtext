"""Weight-only last-Wo gain in the same fixed weak coordinates; no model forward."""
from pathlib import Path
import ast,concurrent.futures,io,json,os,subprocess,time
import numpy as np
import ml_dtypes,tensorstore as ts
R=Path(__file__).resolve().parent
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['gcloud','auth','print-access-token'],text=True).strip()
reports={13000:R.parent/'bam_hunit_prenorm_13k_20261103',23000:R.parent/'bam_hunit_prenorm_23k_20261103',33000:R}
manifest=json.loads((R/'prenorm.json').read_text());s=ts.KvStore.open(manifest['basis_gcs'].rsplit('/',1)[0]+'/').result();b=s.read(manifest['basis_gcs'].rsplit('/',1)[1]).result();assert b.state=='value'
u=np.load(io.BytesIO(b.value)).astype(np.float64);assert u.shape==(2560,128);uc=u-u.mean(0)
key=('params','params','decoder','final_local_layer','block','self_attention','out','kernel')
ctx=ts.Context({'cache_pool#ocdbt':{'total_bytes_limit':268435456}})
def work(job):
 arm,step=job;m=json.loads((reports[step]/(arm+'.json')).read_text());root=m['checkpoints'][str(step)]
 meta=json.loads(subprocess.check_output(['gsutil','cat',root+'/_METADATA'],text=True));assert repr(key) in meta['tree_metadata']
 a=ts.open({'driver':'zarr3' if meta.get('use_zarr3') else 'zarr','kvstore':{'driver':'ocdbt','base':root+'/','cache_pool':'cache_pool#ocdbt'},'path':'.'.join(key)},open=True,read=True,context=ctx).result()
 w=np.asarray(a.read().result(),ml_dtypes.bfloat16).astype(np.float64).reshape(-1,2560);assert w.shape==(2560,2560)
 projected=w@uc;gram=projected.T@projected;e=np.linalg.eigvalsh(gram)
 row=dict(arm=arm,step=step,checkpoint=root,parameter_path='/'.join(key[2:]),shape=list(a.shape),fixed43k_isotropic_weak_gain=float(np.linalg.norm(projected)/np.sqrt(2560)),full_isotropic_gain=float(np.linalg.norm(w)/np.sqrt(2560)),fixed43k_weak_operator_fraction=float(np.sum(projected*projected)/np.sum(w*w)),fixed43k_projected_operator_top_singular=float(np.sqrt(e[-1])))
 row['fixed43k_projected_stable_rank']=row['fixed43k_isotropic_weak_gain']**2*2560/row['fixed43k_projected_operator_top_singular']**2
 print('WO_GEOMETRY',json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(3) as pool:rows=list(pool.map(work,[(arm,step) for arm in ['baseline','prenorm'] for step in reports]))
out=dict(rows=rows,scope='BF16-effective final attention Wo, flatten heads to Din2560 by Dout2560. U is the shared old43k head basis, centered over output D to match existing raw-write projection. Isotropic input unit-RMS expected post-Wo weak RMS=||Wo C U||F/sqrt(Dout). Actual gain/isotropic gain measures readout-Wo joint alignment relative to isotropic input; it depends on read covariance and Wo spectral shape/orientation, not read-activation change alone or causal attribution. All checkpoints read-only; no forward, loss or gradient computed here.')
(R/'wo-geometry.json').write_text(json.dumps(out,indent=2)+'\n')

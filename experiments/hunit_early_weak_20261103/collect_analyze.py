"""Collect and independently validate five-checkpoint final-write statistics."""
from pathlib import Path
import concurrent.futures,json,os,subprocess
import numpy as np,tensorstore as ts
R=Path(__file__).resolve().parent;m=json.loads((R/'arms.json').read_text());root=Path(m['local_DATA']);root.mkdir(exist_ok=True,parents=True)
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['/home/mengqy/google-cloud-sdk/bin/gcloud','auth','print-access-token'],text=True).strip();store=ts.KvStore.open(m['gcs_root']+'/').result()
def fetch(key):
 q=store.read(key).result();assert q.state=='value',key;f=root/key;f.parent.mkdir(exist_ok=True,parents=True);f.write_bytes(q.value)
fetch('complete.json');done=json.loads((root/'complete.json').read_text());assert done['cases']==32*len(m['steps']) and done['source_commit']==m['commit']
keys=[f'{t}/{s}' for t in m['steps'] for s in ['complete.json']+[f'sequence_{i:03d}.{ext}' for i in range(32) for ext in ['json','npz']]]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,keys))
boot=np.random.default_rng(20261103).integers(0,32,(5000,32));rows={};data={};checks=[]
def estimate(v):
 v=np.asarray(v);return dict(mean=v.mean(0).tolist(),ci95=np.quantile(v[boot].mean(1),[.025,.975],axis=0).tolist())
def ratio(a,b):
 a=np.asarray(a);b=np.asarray(b);return dict(mean=(a.mean(0)/b.mean()).tolist(),ci95=np.quantile(a[boot].mean(1)/b[boot].mean(1).reshape((-1,)+(1,)*(a.ndim-1)),[.025,.975],axis=0).tolist())
for t in m['steps']:
 seq=[];metas=[]
 for i in range(32):
  p=root/str(t);q=json.loads((p/f'sequence_{i:03d}.json').read_text());assert q['source_commit']==m['commit'] and q['state_step']==t+1 and q['checkpoint']==m['checkpoints'][str(t)] and q['input_hash']==m['cohort_hashes'][i]
  assert all(f['head_relative']<.015 and abs(f['CE_difference'])<.005 and abs(f['weak_power_relative'])<.015 for f in q['fidelity'])
  with np.load(p/f'sequence_{i:03d}.npz') as z:v={k:z[k].copy() for k in z.files}
  assert all(np.isfinite(a).all() for a in v.values())
  for basis in ['fixed43k','own']:
   b={k[len(basis)+1:]:a for k,a in v.items() if k.startswith(basis+'_')};assert b['head_allocation'].shape==(1,7)
   hcl=(b['head_allocation'].sum()+b['extra_head_allocation'].sum())/b['head_total_power']-1;rcl=(b['raw_allocation'].sum()+b['extra_raw_embedding_allocation'])/b['raw_total_power']-1
   assert abs(hcl)<2e-4 and abs(rcl)<2e-4 and b['projection_ledger_relative_error']<1e-4
  assert v['causal_head_relative'][0]<.015 and abs(v['causal_CE'][0].mean()-v['baseline_CE'].mean())<.005
  seq.append(v);metas.append(q)
 v={k:np.asarray([s[k] for s in seq],np.float64) for k in seq[0]};data[str(t)]=v;row=dict(CE=estimate(v['baseline_CE'].mean(-1)),head_RMS=float(np.sqrt(v['baseline_head_power'].mean())),raw_RMS=float(np.sqrt(v['baseline_raw_power'].mean())))
 for label in ['fixed43k','own']:
  b={k[len(label)+1:]:a for k,a in v.items() if k.startswith(label+'_')};hp=b['head_total_power'];rp=b['raw_total_power'];ha=b['head_allocation'][:,0];ra=b['raw_allocation'][:,0]
  row[label]=dict(head_weak_power=estimate(hp),raw_weak_power=estimate(rp),head_weak_fraction=ratio(hp,v['baseline_head_power']),head_share=ratio(ha,hp),raw_share=ratio(ra,rp),head_allocation=estimate(ha),raw_allocation=estimate(ra),head_self_RMS=np.sqrt(b['head_self_power'].mean(0))[0].tolist(),raw_self_RMS=np.sqrt(b['raw_self_power'].mean(0))[0].tolist(),prefix_LN_head_share=ratio(b['extra_head_allocation'],hp),mo_head_share=ratio(ha[:,1:3].sum(-1),hp),mo_raw_share=ratio(ra[:,1:3].sum(-1),rp),max_ledger_error=float(np.abs(b['projection_ledger_relative_error']).max()),mean_CE_drift=float(np.mean([q['fidelity'][0 if label=='fixed43k' else 1]['CE_difference'] for q in metas])))
  assert abs(row[label]['mean_CE_drift'])<.001
  assert abs(hp.mean()/b['baseline_head_weak_power'].mean()-1)<.015
 b={k[9:]:a for k,a in v.items() if k.startswith('fixed43k_')}
 # Memory statistics refer to the unshifted physical M entering the native final read.
 row['memory_RMS']=float(np.sqrt(b['memory_token_power'].mean()));row['memory_token_RMS_quantiles']=np.quantile(np.sqrt(b['memory_token_power']).ravel(),[.1,.5,.9,.99]).tolist();row['memory_data_centered_RMS']=float(np.sqrt(b['memory_data_centered_power'].mean()));row['memory_mean_RMS']=float(np.sqrt((b['memory_token_mean']**2).mean()));row['read_three_RMS']=np.sqrt(b['read_three_power'].mean((0,2))).tolist();row['mha_Mo_RMS']=np.sqrt(b['read_head_power'].mean((0,1,3))).tolist()
 ce=v['causal_CE'].mean(-1);row['causal']={name:dict(CE=estimate(ce[:,i]),delta_CE=estimate(ce[:,i]-ce[:,0]),fixed43k_weak_power=estimate(v['causal_head_weak_power'][:,i]),weak_power_ratio_to_keep=float(v['causal_head_weak_power'][:,i].mean()/v['causal_head_weak_power'][:,0].mean())) for i,name in enumerate(['keep','remove_mo','remove_mha'])}
 head=json.loads((root.parent/'head_spectra'/f'{t}.json').read_text());u=np.load(root.parent/'head_spectra'/f'{t}.npy');uf=np.load(R.parent/'bam_hunit_write_origin_20261103/fixed43k_basis.npy');row['head_parameters']=head;row['gamma_isotropic_control']=json.loads((R/'gamma_isotropic_control.json').read_text())[str(t)];row['own_fixed43k_subspace_overlap']=float(np.sum((u.T@uf)**2)/128)
 if t in [33000,43000]:
  old=json.loads((R.parent/'bam_hunit_write_origin_20261103/results.json').read_text())['rows'][str(t)]['focused_last_layer']
  row['previous_observer_share_max_absolute_difference']=max(float(np.max(np.abs(np.asarray(row['fixed43k'][sp+'_share']['mean'])-old[sp+'_share']))) for sp in ['raw','head'])
  assert row['previous_observer_share_max_absolute_difference']<.005
  refroot=Path('/data0/mengqy/bam_diagnostics/bam_hunit_spectral_20261102/native_tpu_v2/results')/str(t)
  gold=np.stack([np.load(refroot/f'sequence_{i:03d}.npz')['native_loss'].reshape(-1) for i in range(32)])
  row['archived_native_max_CE_difference']=float(np.max(np.abs(v['baseline_CE']-gold)));assert row['archived_native_max_CE_difference']<.01
 d=json.loads((root/str(t)/'complete.json').read_text());row['final_small_parameters']=d['final_small_parameters'];rows[str(t)]=row
 checks.append(dict(step=t,sequences=32,max_H_relative=max(f['head_relative'] for q in metas for f in q['fidelity']),max_CE_difference=max(abs(f['CE_difference']) for q in metas for f in q['fidelity']),max_weak_power_difference=max(abs(f['weak_power_relative']) for q in metas for f in q['fidelity'])))
 print('CHECKPOINT',t,'CE',row['CE']['mean'],'memory_RMS',row['memory_RMS'],'own weak fraction',row['own']['head_weak_fraction']['mean'],'fixed dyn raw RMS',row['fixed43k']['raw_self_RMS'][1],flush=True)
growth={}
for a,b in zip(m['steps'][:-1],m['steps'][1:]):
 va,vb=data[str(a)],data[str(b)];g={}
 for space in ['raw','head']:
  delta_p=vb[f'fixed43k_{space}_total_power']-va[f'fixed43k_{space}_total_power'];delta_c=vb[f'fixed43k_{space}_allocation'][:,0]-va[f'fixed43k_{space}_allocation'][:,0]
  g[space]=dict(delta_power=estimate(delta_p),component_growth_share=ratio(delta_c,delta_p),mo_growth_share=ratio(delta_c[:,1:3].sum(-1),delta_p))
 growth[f'{a}-{b}']=g
out=dict(complete=True,manifest=m,device=done['device_kind'],components=done['checkpoints']['43000']['components'],rows=rows,growth=growth,verification=checks,interpretation='Signed direct final writes with all cross terms allocated by inner product; own RMS separate. Frozen actual final-LN denominator and gamma map head writes. Raw same-U is a coordinate comparison, not raw effective readout SVD. Own checkpoint Wc bottom128 is contemporaneous, fixed43k basis retrospective. Both MHA and Mo consume memory. Whole-branch removal changes all directions and recomputes final FFN/LN; no causal retraining or M normalization benefit tested.')
(R/'results.json').write_text(json.dumps(out,indent=2)+'\n');(R/'verification.json').write_text(json.dumps(dict(complete=True,source_commit=m['commit'],checks=checks),indent=2)+'\n');print('ALL160_EARLY_WRITE_CASES_VERIFIED',flush=True)

"""Verify native direct-write ledgers; bootstrap texts, signed allocations may cancel."""
from pathlib import Path
import concurrent.futures,json,os,subprocess
import numpy as np,tensorstore as ts
R=Path(__file__).resolve().parent;m=json.loads((R/'arms.json').read_text());root=Path(m['local_DATA']);root.mkdir(exist_ok=True,parents=True)
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['/home/mengqy/google-cloud-sdk/bin/gcloud','auth','print-access-token'],text=True).strip();store=ts.KvStore.open(m['gcs_root']+'/').result()
def fetch(key):
 q=store.read(key).result();assert q.state=='value',key;f=root/key;f.parent.mkdir(exist_ok=True,parents=True);t=f.with_suffix(f.suffix+'.tmp');t.write_bytes(q.value);t.replace(f)
fetch('complete.json');done=json.loads((root/'complete.json').read_text());assert done['cases']==64 and done['source_commit']==m['commit']
keys=[f'{t}/{s}' for t in m['steps'] for s in ['complete.json']+[f'sequence_{i:03d}.{ext}' for i in range(32) for ext in ['json','npz']]]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,keys))
boot=np.random.default_rng(20261103).integers(0,32,(5000,32));rows={};data={};checks=[]
def estimate(v):return dict(mean=np.mean(v,0).tolist(),ci95=np.quantile(np.asarray(v)[boot].mean(1),[.025,.975],axis=0).tolist())
for t in sorted(m['steps']):
 seq=[];metas=[]
 for i in range(32):
  p=root/str(t);q=json.loads((p/f'sequence_{i:03d}.json').read_text());assert q['source_commit']==m['commit'] and q['state_step']==t+1 and q['checkpoint']==m['checkpoints'][str(t)] and q['input_hash']==m['cohort_hashes'][i]
  assert q['focused_head_relative_RMS']<.015 and abs(q['focused_CE_replay_difference'])<.005
  with np.load(p/f'sequence_{i:03d}.npz') as z:v={k:z[k].copy() for k in z.files}
  assert all(np.isfinite(x).all() for x in v.values());assert v['head_allocation'].shape==v['head_self_power'].shape==(32,7)
  hc=(v['head_allocation'].sum()+v['extra_head_allocation'].sum())/v['head_total_power']-1;rc=(v['raw_allocation'].sum()+v['extra_raw_embedding_allocation'])/v['raw_total_power']-1
  assert abs(hc)<2e-4 and abs(rc)<2e-4 and v['projection_ledger_relative_error']<1e-4 and v['layer_continuity_relative_error']<2e-6,(t,i,hc,rc)
  fc=(v['focus_head_allocation'].sum()+v['focus_extra_head_allocation'].sum())/v['focus_head_total_power']-1;assert abs(fc)<2e-4 and float(v['causal_head_relative_to_observed'][0])<.015
  seq.append(v);metas.append(q)
 v={k:np.asarray([s[k] for s in seq],np.float64) for k in seq[0]};data[str(t)]=v;h=v['head_total_power'];r=v['raw_total_power'];ha=v['head_allocation'];ra=v['raw_allocation'];gold=m['gold_head_weak_power'][str(t)]
 # Heavy internal observation remains exploratory; do not filter its samples.
 row=dict(head_total_power=estimate(h),raw_total_power=estimate(r),head_allocation=estimate(ha),raw_allocation=estimate(ra),head_share=dict(mean=(ha.mean(0)/h.mean()).tolist(),ci95=np.quantile(ha[boot].mean(1)/h[boot].mean(1)[:,None,None],[.025,.975],axis=0).tolist()),raw_share=dict(mean=(ra.mean(0)/r.mean()).tolist(),ci95=np.quantile(ra[boot].mean(1)/r[boot].mean(1)[:,None,None],[.025,.975],axis=0).tolist()),head_self_RMS=np.sqrt(v['head_self_power'].mean(0)).tolist(),raw_self_RMS=np.sqrt(v['raw_self_power'].mean(0)).tolist(),extra_head_share=(v['extra_head_allocation'].mean(0)/h.mean()).tolist(),embedding_raw_share=float(v['extra_raw_embedding_allocation'].mean()/r.mean()),read_head_RMS=np.sqrt(v['read_head_power'].mean(0)).tolist(),max_head_relative_RMS=max(s['head_relative_RMS'] for s in metas),mean_CE_replay_difference=float(np.mean([s['CE_replay_difference'] for s in metas])),head_power_relative_to_original=float(h.mean()/gold-1),sequence=metas)
 ce=v['causal_CE'].mean(-1);wp=v['causal_head_weak_power'];row['last_layer_causal_controls']={name:dict(CE=estimate(ce[:,i]),delta_CE=estimate(ce[:,i]-ce[:,0]),weak_power=estimate(wp[:,i]),weak_power_ratio_to_keep=float(wp[:,i].mean()/wp[:,0].mean()),head_RMS=float(np.sqrt(np.mean(v['causal_head_RMS'][:,i]**2))),raw_RMS=float(np.sqrt(np.mean(v['causal_raw_RMS'][:,i]**2)))) for i,name in enumerate(['keep','remove_mo','remove_mha'])}
 fh=v['focus_head_total_power'];fa=v['focus_head_allocation'];fr=v['focus_raw_total_power'];fra=v['focus_raw_allocation']
 assert abs(fh.mean()/gold-1)<.015 and abs(np.mean([s['focused_CE_replay_difference'] for s in metas]))<.001
 fs=fa.mean(0)/fh.mean();rs=fra.mean(0)/fr.mean();disagreement=float(np.max(np.abs(ha.mean(0)[-1]/h.mean()-fs[0])))
 row['focused_last_layer']=dict(head_share=fs[0].tolist(),head_share_ci95=np.quantile(fa[boot].mean(1)/fh[boot].mean(1)[:,None,None],[.025,.975],axis=0)[:,0].tolist(),raw_share=rs[0].tolist(),head_self_RMS=np.sqrt(v['focus_head_self_power'].mean(0))[0].tolist(),raw_self_RMS=np.sqrt(v['focus_raw_self_power'].mean(0))[0].tolist(),head_allocation=fa.mean(0)[0].tolist(),head_total_power=float(fh.mean()),raw_total_power=float(fr.mean()),prefix_and_LN_head_share=(v['focus_extra_head_allocation'].mean(0)/fh.mean()).tolist(),full_vs_focus_max_absolute_share_difference=disagreement,max_head_relative_RMS=max(s['focused_head_relative_RMS'] for s in metas),mean_CE_replay_difference=float(np.mean([s['focused_CE_replay_difference'] for s in metas])))
 mo=fa[:,0,1:3].sum(1);fd=fa[:,0,5]-mo
 row['focused_last_layer']['mo_total_share']=dict(mean=float(mo.mean()/fh.mean()),ci95=np.quantile(mo[boot].mean(1)/fh[boot].mean(1),[.025,.975]).tolist())
 row['focused_last_layer']['ffn_minus_mo_share']=dict(mean=float(fd.mean()/fh.mean()),ci95=np.quantile(fd[boot].mean(1)/fh[boot].mean(1),[.025,.975]).tolist())
 row['full_depth_fidelity_accepted']=all(s['full_depth_fidelity_accepted'] for s in metas) and abs(h.mean()/gold-1)<.015
 bh=v['boundary_head_total_power'];br=v['boundary_raw_total_power'];ba=v['boundary_head_allocation'];bra=v['boundary_raw_allocation']
 accepted=all(s['boundary_fidelity_accepted'] for s in metas) and abs(bh.mean()/gold-1)<.015 and abs(np.mean([s['boundary_CE_replay_difference'] for s in metas]))<.001
 row['layer_boundaries']=dict(accepted=bool(accepted),head_share=(ba.mean(0)/bh.mean()).tolist(),raw_share=(bra.mean(0)/br.mean()).tolist(),head_allocation=ba.mean(0).tolist(),raw_allocation=bra.mean(0).tolist(),head_total_power=float(bh.mean()),raw_total_power=float(br.mean()),max_head_relative_RMS=max(s['boundary_head_relative_RMS'] for s in metas),mean_CE_replay_difference=float(np.mean([s['boundary_CE_replay_difference'] for s in metas])),head_power_relative_to_original=float(bh.mean()/gold-1),max_head_budget_closure=float(np.max(np.abs(v['boundary_head_allocation_closure']))))
 rows[str(t)]=row;checks.append(dict(step=t,sequences=32,max_head_allocation_closure=float(np.abs(v['head_allocation_closure']).max()),max_raw_allocation_closure=float(np.abs(v['raw_allocation_closure']).max()),max_projection_ledger_error=float(v['projection_ledger_relative_error'].max()),max_continuity_error=float(v['layer_continuity_max'].max()),max_relative_continuity_error=float(v['layer_continuity_relative_error'].max()),max_focused_head_relative_RMS=row['focused_last_layer']['max_head_relative_RMS'],boundary_accepted=row['layer_boundaries']['accepted']))
a,b=data['33000'],data['43000'];dh=b['head_total_power'].mean()-a['head_total_power'].mean();dr=b['raw_total_power'].mean()-a['raw_total_power'].mean();growth=dict(head_power=float(dh),raw_power=float(dr),head_allocation_growth_share=((b['head_allocation'].mean(0)-a['head_allocation'].mean(0))/dh).tolist(),raw_allocation_growth_share=((b['raw_allocation'].mean(0)-a['raw_allocation'].mean(0))/dr).tolist(),extra_head_growth_share=((b['extra_head_allocation'].mean(0)-a['extra_head_allocation'].mean(0))/dh).tolist())
out=dict(complete=True,manifest=m,device=done['device_kind'],scope=done['scope'],components=done['checkpoints']['43000']['components'],rows=rows,growth=growth,verification=checks,interpretation='Signed direct-write allocation: <component projection, final weak projection>/final weak power. Own RMS and aligned contribution differ; negative values cancel positive writes; allocations can exceed100%. Components are linear FP32 post-Wo branches of native BF16 reads; bias and all native rounding assigned separately. Final-LN denominator frozen to each actual token for additive accounting, not causal removal; compare pre-LN raw allocations as well. Fixed43k weak basis avoids rotation. Primary results use independent final-only and physical carry-boundary observations; full internal capture is exploratory and its fidelity flags remain explicit. M_o identifies the read consumer, not the upstream memory writer or training cause.')
growth['focused_last_layer_head_growth_share']=((b['focus_head_allocation'].mean(0)-a['focus_head_allocation'].mean(0))/(b['focus_head_total_power'].mean()-a['focus_head_total_power'].mean()))[0].tolist()
growth['focused_last_layer_raw_growth_share']=((b['focus_raw_allocation'].mean(0)-a['focus_raw_allocation'].mean(0))/(b['focus_raw_total_power'].mean()-a['focus_raw_total_power'].mean()))[0].tolist()
growth['boundary_head_net_growth_share']=((b['boundary_head_allocation'].mean(0)-a['boundary_head_allocation'].mean(0))/(b['boundary_head_total_power'].mean()-a['boundary_head_total_power'].mean())).tolist()
for label in ['raw','head']:
 delta_c=b[f'focus_{label}_allocation'][:,0]-a[f'focus_{label}_allocation'][:,0];delta_p=b[f'focus_{label}_total_power']-a[f'focus_{label}_total_power']
 growth[f'focused_last_layer_{label}_growth_ci95']=np.quantile(delta_c[boot].mean(1)/delta_p[boot].mean(1)[:,None],[.025,.975],axis=0).tolist()
 mo=delta_c[:,1:3].sum(1);growth[f'focused_mo_{label}_growth_share']=dict(mean=float(mo.mean()/delta_p.mean()),ci95=np.quantile(mo[boot].mean(1)/delta_p[boot].mean(1),[.025,.975]).tolist())
out['scope']='Primary independent layer carry-boundary and final-only native observations; heavy full internal capture is exploratory, with failed fidelity flags retained rather than filtering texts.'
(R/'results.json').write_text(json.dumps(out,indent=2)+'\n');(R/'verification.json').write_text(json.dumps(dict(complete=True,source_commit=m['commit'],checks=checks),indent=2)+'\n')
for t,row in rows.items():
 ha=np.asarray(row['head_share']['mean']);ra=np.asarray(row['raw_share']['mean']);print('CHECKPOINT',t,'all components',dict(zip(out['components'],ha.sum(0))),flush=True);print('LAST LAYER HEAD',ha[-1].tolist(),'RAW',ra[-1].tolist(),flush=True)
print('NATIVE_WRITE_ORIGIN_ALL64_VERIFIED',flush=True)

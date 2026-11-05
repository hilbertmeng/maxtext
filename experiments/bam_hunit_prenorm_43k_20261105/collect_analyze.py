"""Paired cohort estimates and strict per-text fidelity checks, no sample filtering."""
from pathlib import Path
import concurrent.futures, json, os, subprocess
import numpy as np
import tensorstore as ts

R=Path(__file__).resolve().parent
os.environ['GOOGLE_AUTH_TOKEN_FOR_TESTING']=subprocess.check_output(['gcloud','auth','print-access-token'],text=True).strip()
boot=np.random.default_rng(20261103).integers(0,32,(5000,32))
def estimate(a):
    a=np.asarray(a)
    return dict(mean=a.mean(0).tolist(),ci95=np.quantile(a[boot].mean(1),[.025,.975],axis=0).tolist())
def ratio(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return dict(mean=(a.mean(0)/b.mean()).tolist(),ci95=np.quantile(
        a[boot].mean(1)/b[boot].mean(1).reshape((-1,)+(1,)*(a.ndim-1)),[.025,.975],axis=0).tolist())

rows={};data={};verification={};manifests={}
for arm in ['prenorm']:
    m=json.loads((R/(arm+'.json')).read_text());manifests[arm]=m
    root=Path(m['local_DATA']);root.mkdir(parents=True,exist_ok=True)
    store=ts.KvStore.open(m['gcs_root']+'/').result()
    def fetch(key):
        q=store.read(key).result();assert q.state=='value',key
        p=root/key;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(q.value)
    keys=['complete.json','43000/complete.json']+[f'43000/sequence_{i:03d}.{ext}' for i in range(32) for ext in ['json','npz']]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(fetch,keys))
    done=json.loads((root/'complete.json').read_text());assert done['source_commit']==m['commit'] and done['cases']==32
    seq=[];metas=[]
    for i in range(32):
        p=root/'43000';q=json.loads((p/f'sequence_{i:03d}.json').read_text())
        assert q['source_commit']==m['commit'] and q['checkpoint']==m['checkpoints']['43000'] and q['state_step']==43001
        assert q['device_kind']==m['expected_device_kind']
        assert q['input_hash']==m['cohort_hashes'][i] and q['sequence']==i and q['shard_index']==i%4
        assert all(f['head_relative']<.015 and abs(f['CE_difference'])<.005 and abs(f['weak_power_relative'])<.015 for f in q['fidelity'])
        with np.load(p/f'sequence_{i:03d}.npz') as z:v={k:z[k].copy() for k in z.files}
        assert all(np.isfinite(a).all() for a in v.values())
        for basis in ['fixed43k','own']:
            b={k[len(basis)+1:]:a for k,a in v.items() if k.startswith(basis+'_')}
            assert b['head_allocation'].shape==(1,7)
            assert abs((b['head_allocation'].sum()+b['extra_head_allocation'].sum())/b['head_total_power']-1)<2e-4
            assert abs((b['raw_allocation'].sum()+b['extra_raw_embedding_allocation'])/b['raw_total_power']-1)<2e-4
            assert b['projection_ledger_relative_error']<1e-4
        assert v['causal_head_relative'][0]<.015
        assert abs(v['causal_CE'][0].mean()-v['baseline_CE'].mean())<.005
        seq.append(v);metas.append(q)
    v={k:np.asarray([s[k] for s in seq],np.float64) for k in seq[0]};data[arm]=v
    row=dict(native_CE=estimate(v['baseline_CE'].mean(-1)),common_fp32_CE=estimate(v['common_fp32_CE'].mean(-1)),
             common_bf16_logits_CE=estimate(v['common_bf16_logits_CE'].mean(-1)),
             raw_RMS=float(np.sqrt(v['baseline_raw_power'].mean())),head_RMS=float(np.sqrt(v['baseline_head_power'].mean())))
    for label in ['fixed43k','own']:
        b={k[len(label)+1:]:a for k,a in v.items() if k.startswith(label+'_')}
        hp=b['head_total_power'];rp=b['raw_total_power'];ha=b['head_allocation'][:,0];ra=b['raw_allocation'][:,0]
        row[label]=dict(head_weak_power=estimate(hp),raw_weak_power=estimate(rp),
            head_weak_fraction=ratio(hp,v['baseline_head_power']),raw_centered_weak_fraction=ratio(rp,v['baseline_raw_centered_power']),
            head_weak_RMS=float(np.sqrt(hp.mean())),raw_weak_RMS=float(np.sqrt(rp.mean())),
            head_share=ratio(ha,hp),raw_share=ratio(ra,rp),head_allocation=estimate(ha),raw_allocation=estimate(ra),
            head_self_RMS=np.sqrt(b['head_self_power'].mean(0))[0].tolist(),raw_self_RMS=np.sqrt(b['raw_self_power'].mean(0))[0].tolist(),
            prefix_LN_head_share=ratio(b['extra_head_allocation'],hp),mo_head_share=ratio(ha[:,1:3].sum(-1),hp),
            mo_raw_share=ratio(ra[:,1:3].sum(-1),rp),max_ledger_error=float(np.abs(b['projection_ledger_relative_error']).max()))
    b={k[9:]:a for k,a in v.items() if k.startswith('fixed43k_')}
    row['memory_RMS']=float(np.sqrt(b['memory_token_power'].mean()))
    row['mha_Mo_RMS']=np.sqrt(b['read_head_power'].mean((0,1,3))).tolist()
    row['read_three_RMS']=np.sqrt(b['read_three_power'].mean((0,2))).tolist()
    ce=v['causal_CE'].mean(-1)
    row['causal']={name:dict(delta_CE=estimate(ce[:,i]-ce[:,0]),
        fixed43k_weak_power_ratio_to_keep=float(v['causal_head_weak_power'][:,i].mean()/v['causal_head_weak_power'][:,0].mean()))
        for i,name in enumerate(['keep','remove_mo','remove_mha'])}
    head=json.loads((root/'head_spectra/43000.json').read_text());row['head_parameters']=head
    row['own_head_fraction_over_gamma_isotropic']=row['own']['head_weak_fraction']['mean']/head['gamma_isotropic_control']
    row['small_parameters']=json.loads((root/'43000/complete.json').read_text())['final_small_parameters']
    rows[arm]=row
    verification[arm]=dict(sequences=32,max_H_relative=max(f['head_relative'] for q in metas for f in q['fidelity']),
        max_CE_difference=max(abs(f['CE_difference']) for q in metas for f in q['fidelity']),
        max_weak_power_relative=max(abs(f['weak_power_relative']) for q in metas for f in q['fidelity']))
    print('ARM',arm,'CE',row['common_fp32_CE']['mean'],'own fraction',row['own']['head_weak_fraction']['mean'],flush=True)

# Reuse the already analyzed baseline; never restore or run its model here.
task=json.loads((R/'task.json').read_text());archive=json.loads(Path(task['baseline_archive']).read_text());am=archive['manifest'];m=manifests['prenorm'];assert am['cohort_hashes']==m['cohort_hashes']
baseline=json.loads(json.dumps(archive['rows']['43000']))
baseline['native_CE']=baseline.pop('CE')
for label in ['own','fixed43k']:
    baseline[label]['raw_weak_RMS']=float(np.sqrt(baseline[label]['raw_weak_power']['mean']))
    baseline[label]['head_weak_RMS']=float(np.sqrt(baseline[label]['head_weak_power']['mean']))
control=baseline['gamma_isotropic_control']['centered_isotropic_input_expected_weak_fraction']
baseline['head_parameters']['gamma_isotropic_control']=control
baseline['own_head_fraction_over_gamma_isotropic']=baseline['own']['head_weak_fraction']['mean']/control
baseline['reused_from']=task['baseline_archive'];rows['baseline']=baseline
verification['baseline']=dict(reused=True,source=task['baseline_archive'],sequences=32,original=next(x for x in archive['verification'] if x['step']==43000))

cached=[]
for i in range(32):
    p=Path(am['local_DATA'])/'43000';q=json.loads((p/f'sequence_{i:03d}.json').read_text())
    assert q['source_commit']==am['commit'] and q['state_step']==43001 and q['checkpoint']==am['checkpoints']['43000']
    assert q['input_hash']==m['cohort_hashes'][i] and q['device_kind']==m['expected_device_kind']
    with np.load(p/f'sequence_{i:03d}.npz') as z:cached.append({k:z[k].copy() for k in ['own_head_total_power','fixed43k_head_total_power','baseline_head_power']})
vb={k:np.asarray([s[k] for s in cached],np.float64) for k in cached[0]};vn=data['prenorm']
assert abs(vb['own_head_total_power'].mean()/vb['baseline_head_power'].mean()-baseline['own']['head_weak_fraction']['mean'])<1e-8
paired={}
for label in ['own','fixed43k']:
    nh,oh=vn[label+'_head_total_power'],vb[label+'_head_total_power'];nt,ot=vn['baseline_head_power'],vb['baseline_head_power']
    delta=nh[boot].mean(1)/nt[boot].mean(1)-oh[boot].mean(1)/ot[boot].mean(1)
    paired[label+'_head_fraction_delta']=dict(mean=float(nh.mean()/nt.mean()-oh.mean()/ot.mean()),ci95=np.quantile(delta,[.025,.975]).tolist())

previous={};temporal={}
for step,date in [(13000,'20261103'),(23000,'20261103'),(33000,'20261104')]:
    report=R.parent/f'bam_hunit_prenorm_{step//1000}k_{date}'
    d=json.loads((report/'results.json').read_text());pm=d['models']['prenorm'];assert pm['cohort_hashes']==m['cohort_hashes']
    vals=[]
    for i in range(32):
        p=Path(pm['local_DATA'])/str(step);q=json.loads((p/f'sequence_{i:03d}.json').read_text());assert q['input_hash']==m['cohort_hashes'][i] and q['state_step']==step+1 and q['source_commit']==pm['commit']
        with np.load(p/f'sequence_{i:03d}.npz') as z:vals.append({k:z[k].copy() for k in ['own_head_total_power','baseline_head_power','fixed43k_raw_self_power','fixed43k_read_three_power','common_fp32_CE']})
    pv={k:np.asarray([x[k] for x in vals],np.float64) for k in vals[0]};previous[str(step)]=dict(report=str(report),rows=d['rows'],model=pm)
    delta=vn['own_head_total_power'][boot].mean(1)/vn['baseline_head_power'][boot].mean(1)-pv['own_head_total_power'][boot].mean(1)/pv['baseline_head_power'][boot].mean(1)
    a=vn['fixed43k_raw_self_power'][:,0,1];ar=vn['fixed43k_read_three_power'].mean((2,))[:,0]
    b=pv['fixed43k_raw_self_power'][:,0,1];br=pv['fixed43k_read_three_power'].mean((2,))[:,0]
    gain=np.sqrt((a[boot].mean(1)/ar[boot].mean(1))/(b[boot].mean(1)/br[boot].mean(1)))
    temporal[str(step)+'-43000']=dict(head_fraction_delta_pp=float((vn['own_head_total_power'].mean()/vn['baseline_head_power'].mean()-pv['own_head_total_power'].mean()/pv['baseline_head_power'].mean())*100),head_fraction_delta_pp_ci95=(np.quantile(delta,[.025,.975])*100).tolist(),common_fp32_CE_delta=estimate(vn['common_fp32_CE'].mean(-1)-pv['common_fp32_CE'].mean(-1)),fixed43k_dynamicMo_effective_weak_gain_ratio=float(np.sqrt((a.mean()/ar.mean())/(b.mean()/br.mean()))),effective_weak_gain_ratio_ci95=np.quantile(gain,[.025,.975]).tolist())

out=dict(complete=True,measured_only=dict(arm='prenorm',steps=[43000],sequences=32),models=manifests,rows=rows,paired=paired,temporal=temporal,previous=previous,verification=verification,baseline_manifest=am,components=['mha','mo_dynamic','mo_static','attn_bias','attn_rounding','ffn','residual_add_rounding'],training_health=json.loads((R/'tb_health.json').read_text()),scope='Only new43k model executed. All previous steps and baseline43k reused from verified same32x2048 native v5p records. Own centered effective-BF16 Wc bottom128 is primary; fixed old43k is common retrospective geometry. Activation energy is not gradient energy. Signed allocations include cross terms. Baseline archive has native CE, not a common-FP32-head measurement: no matched-precision cross-model CE claim.')
(R/'results.json').write_text(json.dumps(out,indent=2)+'\n');(R/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print('NEW43K32_VERIFIED_ALL_OLD_METRICS_REUSED',rows['prenorm']['own']['head_weak_fraction'],flush=True)

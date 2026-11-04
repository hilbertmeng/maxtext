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
for arm in ['baseline','prenorm']:
    m=json.loads((R/(arm+'.json')).read_text());manifests[arm]=m
    root=Path(m['local_DATA']);root.mkdir(parents=True,exist_ok=True)
    store=ts.KvStore.open(m['gcs_root']+'/').result()
    def fetch(key):
        q=store.read(key).result();assert q.state=='value',key
        p=root/key;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(q.value)
    keys=['complete.json','33000/complete.json']+[f'33000/sequence_{i:03d}.{ext}' for i in range(32) for ext in ['json','npz']]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(fetch,keys))
    done=json.loads((root/'complete.json').read_text());assert done['source_commit']==m['commit'] and done['cases']==32
    seq=[];metas=[]
    for i in range(32):
        p=root/'33000';q=json.loads((p/f'sequence_{i:03d}.json').read_text())
        assert q['source_commit']==m['commit'] and q['checkpoint']==m['checkpoints']['33000'] and q['state_step']==33001
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
    head=json.loads((root/'head_spectra/33000.json').read_text());row['head_parameters']=head
    row['own_head_fraction_over_gamma_isotropic']=row['own']['head_weak_fraction']['mean']/head['gamma_isotropic_control']
    row['small_parameters']=json.loads((root/'33000/complete.json').read_text())['final_small_parameters']
    rows[arm]=row
    verification[arm]=dict(sequences=32,max_H_relative=max(f['head_relative'] for q in metas for f in q['fidelity']),
        max_CE_difference=max(abs(f['CE_difference']) for q in metas for f in q['fidelity']),
        max_weak_power_relative=max(abs(f['weak_power_relative']) for q in metas for f in q['fidelity']))
    print('ARM',arm,'CE',row['common_fp32_CE']['mean'],'own fraction',row['own']['head_weak_fraction']['mean'],flush=True)

assert manifests['baseline']['cohort_hashes']==manifests['prenorm']['cohort_hashes']
vb,vn=data['baseline'],data['prenorm']
paired={k:estimate(vn[k].mean(-1)-vb[k].mean(-1)) for k in ['baseline_CE','common_fp32_CE','common_bf16_logits_CE']}
for basis in ['fixed43k','own']:
    k=basis+'_head_total_power'
    old=vb[k];new=vn[k];op=vb['baseline_head_power'];np_=vn['baseline_head_power']
    paired[basis+'_head_fraction_delta']=dict(mean=float(new.mean()/np_.mean()-old.mean()/op.mean()),
        ci95=np.quantile(new[boot].mean(1)/np_[boot].mean(1)-old[boot].mean(1)/op[boot].mean(1),[.025,.975]).tolist())
    paired[basis+'_head_weak_power_ratio']=ratio(new,old)
    paired[basis+'_raw_weak_power_ratio']=ratio(vn[basis+'_raw_total_power'],vb[basis+'_raw_total_power'])
old_manifest=json.loads((R.parent/'bam_hunit_early_weak_v5p_20261103/arms.json').read_text())
old_root=Path(old_manifest['local_DATA'])/'33000'
legacy=[]
for i in range(32):
    with np.load(old_root/f'sequence_{i:03d}.npz') as z:
        legacy.append(dict(CE_delta=float(vb['baseline_CE'][i].mean()-z['baseline_CE'].mean()),
            own_head_weak_power_relative=float(vb['own_head_total_power'][i]/z['own_head_total_power']-1)))
verification['baseline_legacy_replay']=dict(mean_CE_delta=float(np.mean([x['CE_delta'] for x in legacy])),
    max_abs_sequence_CE_delta=max(abs(x['CE_delta']) for x in legacy),
    max_abs_own_head_weak_power_relative=max(abs(x['own_head_weak_power_relative']) for x in legacy))
verification['baseline_legacy_replay']['accepted']=bool(verification['baseline_legacy_replay']['max_abs_sequence_CE_delta']<.005 and verification['baseline_legacy_replay']['max_abs_own_head_weak_power_relative']<.015)
assert verification['baseline_legacy_replay']['accepted'],('Historical baseline replay failed',verification['baseline_legacy_replay'])
uold=np.load(Path(manifests['baseline']['local_DATA'])/'head_spectra/33000.npy')
unew=np.load(Path(manifests['prenorm']['local_DATA'])/'head_spectra/33000.npy')
out=dict(complete=True,models=manifests,rows=rows,paired=paired,verification=verification,
    own_subspace_overlap=float(np.sum((uold.T@unew)**2)/128),
    components=['mha','mo_dynamic','mo_static','attn_bias','attn_rounding','ffn','residual_add_rounding'],
    scope='Same32 Pile-source2048 texts, both native v5p at33k. Own centered effective-BF16 Wc bottom128 and old fixed43k directions. Head energy is activation energy, not gradient energy. Signed allocations include cross terms. No training; configs differ in pre-norm, QK read scale, FP32 output logits.')
out['training_health_window_32901_33000']=json.loads((R/'tb_health.json').read_text())
(R/'results.json').write_text(json.dumps(out,indent=2)+'\n')
(R/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print('ALL64_PAIRED_CASES_VERIFIED',flush=True)

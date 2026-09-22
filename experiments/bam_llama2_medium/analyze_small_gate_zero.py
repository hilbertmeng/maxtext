"""Read-open AND write<=0.02 ablation, with coverage and paired uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
from analyze_write_gate_bets import holm


def analyze(root, adaptive_root, out):
    meta=json.loads((root/'metadata.json').read_text())
    refmeta=json.loads((adaptive_root/'metadata.json').read_text())
    assert meta['small_gate_cutoff']==.02
    assert meta['cohort_sha256']==refmeta['cohort_sha256'] and meta['dtype']==refmeta['dtype']
    heads=meta['protocol']['heads']
    records=[json.loads(p.read_text()) for p in sorted(root.glob('seq_*.json'))]
    assert len(records)>=2
    n=len(records);shape=(n,len(heads))
    delta=np.zeros(shape);prediction=np.zeros(shape);selected=np.zeros(shape);removed=np.zeros(shape)
    read_high_count=np.zeros(shape);read_high_mass=np.zeros(shape)
    for i,r in enumerate(records):
        rows={x['id']:x for x in r['rows']};grad={x['id']:x for x in r['gradients']}
        ref=json.loads((adaptive_root/f"seq_{r['sequence']:03d}.json").read_text());refrows={x['id']:x for x in ref['rows']}
        for control in ['baseline','zero_shift_control','terminal_control']:assert abs(rows[control]['delta'])<1e-7
        for j,h in enumerate(heads):
            a=rows[h['id']+'_shift-1.0'];b=refrows[h['id']+'_shift-1.0']
            assert a['clipped']==0 and abs(a['gate_removed']-a['gate_before'])<1e-6
            delta[i,j]=a['delta'];prediction[i,j]=-grad[h['id']]['derivative']
            selected[i,j]=a['selected'];removed[i,j]=a['gate_removed']
            read_high_count[i,j]=b['selected'];read_high_mass[i,j]=b['gate_before']
    weights=np.random.default_rng(9202203).multinomial(n,np.full(n,1/n),size=20000)/n
    boot=weights@delta;ci=np.quantile(boot,[.025,.975],axis=0)
    p=stats.ttest_1samp(delta,0,axis=0).pvalue
    p=np.where(np.isnan(p)&(np.max(abs(delta),axis=0)==0),1.,p);adjusted=holm(p)
    result=[]
    for j,h in enumerate(heads):
        active=int(np.count_nonzero(selected[:,j]))
        result.append(dict(id=h['id'],groups=h['groups'],mean_delta=float(delta[:,j].mean()),ci95=ci[:,j].tolist(),
                           holm_p=float(adjusted[j]),gradient_prediction=float(prediction[:,j].mean()),
                           selected_tokens=int(selected[:,j].sum()),active_sequences=active,
                           fraction_of_read_high_tokens=float(selected[:,j].sum()/max(1,read_high_count[:,j].sum())),
                           fraction_of_read_high_gate_mass=float(removed[:,j].sum()/max(1e-20,read_high_mass[:,j].sum())),
                           no_eligible_positions=active==0))
    groups={}
    for g in meta['protocol']['predictions']:
        members=[j for j,h in enumerate(heads) if g in h['groups']]
        groups[g]=dict(mean_delta=float(delta[:,members].mean()),ci95=np.quantile(boot[:,members].mean(1),[.025,.975]).tolist(),
                       heads_with_any_eligible_positions=sum(result[j]['active_sequences']>0 for j in members),
                       nominal_positive=sum(result[j]['ci95'][0]>0 for j in members),nominal_negative=sum(result[j]['ci95'][1]<0 for j in members),
                       holm_positive=sum(adjusted[j]<.05 and delta[:,j].mean()>0 for j in members),
                       holm_negative=sum(adjusted[j]<.05 and delta[:,j].mean()<0 for j in members))
    summary=dict(n_sequences=n,sequences=[r['sequence'] for r in records],complete=[r['sequence'] for r in records]==list(range(32,128)),
                 dtype=meta['dtype'],cutoff=.02,heads=result,groups=groups)
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
    np.savez_compressed(out/'paired.npz',delta=delta,prediction=prediction,selected=selected,removed=removed)
    print(json.dumps(dict(n=n,groups=groups),indent=2,default=lambda x:x.item()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('adaptive_root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();analyze(a.root,a.adaptive_root,a.out)

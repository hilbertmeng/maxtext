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
    specs=[('zero','_shift-1.0'),('double','_shift1.0'),('fivefold','_shift4.0'),('open_p005','_open_p005')]
    all_delta=np.zeros(shape+(len(specs),))
    delta=np.zeros(shape);prediction=np.zeros(shape);selected=np.zeros(shape);removed=np.zeros(shape)
    read_high_count=np.zeros(shape);read_high_mass=np.zeros(shape)
    for i,r in enumerate(records):
        rows={x['id']:x for x in r['rows']};grad={x['id']:x for x in r['gradients']}
        context=adaptive_root/f"gradient_{r['sequence']:03d}.npz"
        if context.exists():
            with np.load(context) as f:
                gates, reads = f['write_gate'], f['read_gate']
                refrows={}
                for h in heads:
                    g=gates[h['layer'],:,h['head']]
                    # Match the intervention's gate budget, including positions
                    # whose target loss is masked. Gradient-sign coverage uses
                    # valid target positions separately.
                    mask=reads[h['layer'],:,h['head']]>=h['threshold']
                    refrows[h['id']+'_shift-1.0']=dict(selected=int(mask.sum()),gate_before=float(g[mask].sum(dtype=np.float64)))
        else:
            ref=json.loads((adaptive_root/f"seq_{r['sequence']:03d}.json").read_text());refrows={x['id']:x for x in ref['rows']}
        for control in ['baseline','zero_shift_control','terminal_control']:assert abs(rows[control]['delta'])<1e-7
        for j,h in enumerate(heads):
            a=rows[h['id']+'_shift-1.0'];b=refrows[h['id']+'_shift-1.0']
            assert a['clipped']==0 and abs(a['gate_removed']-a['gate_before'])<1e-6
            delta[i,j]=a['delta'];prediction[i,j]=-grad[h['id']]['derivative']
            selected[i,j]=a['selected'];removed[i,j]=a['gate_removed']
            read_high_count[i,j]=b['selected'];read_high_mass[i,j]=b['gate_before']
            for k,(_,suffix) in enumerate(specs):
                arm=rows[h['id']+suffix];assert arm['selected']==a['selected'] and arm['clipped']==0
                all_delta[i,j,k]=arm['delta']
    weights=np.random.default_rng(9202203).multinomial(n,np.full(n,1/n),size=20000)/n
    boot=weights@delta;ci=np.quantile(boot,[.025,.975],axis=0)
    p=stats.ttest_1samp(delta,0,axis=0).pvalue
    p=np.where(np.isnan(p)&(np.max(abs(delta),axis=0)==0),1.,p);adjusted=holm(p)
    all_boot=(weights@all_delta.reshape(n,-1)).reshape(len(weights),len(heads),len(specs))
    all_ci=np.quantile(all_boot,[.025,.975],axis=0)
    all_p=stats.ttest_1samp(all_delta,0,axis=0).pvalue
    all_p=np.where(np.isnan(all_p)&(np.max(abs(all_delta),axis=0)==0),1.,all_p)
    all_adjusted=holm(all_p.ravel()).reshape(len(heads),len(specs))
    result=[]
    for j,h in enumerate(heads):
        active=int(np.count_nonzero(selected[:,j]))
        result.append(dict(id=h['id'],groups=h['groups'],mean_delta=float(delta[:,j].mean()),ci95=ci[:,j].tolist(),
                           holm_p=float(adjusted[j]),gradient_prediction=float(prediction[:,j].mean()),
                           selected_tokens=int(selected[:,j].sum()),active_sequences=active,
                           fraction_of_read_high_tokens=float(selected[:,j].sum()/max(1,read_high_count[:,j].sum())),
                           fraction_of_read_high_gate_mass=float(removed[:,j].sum()/max(1e-20,read_high_mass[:,j].sum())),
                           no_eligible_positions=active==0,
                           treatments={name:dict(mean_delta=float(all_delta[:,j,k].mean()),ci95=all_ci[:,j,k].tolist(),holm_all_arms_p=float(all_adjusted[j,k])) for k,(name,_) in enumerate(specs)}))
    groups={}
    for g in meta['protocol']['predictions']:
        members=[j for j,h in enumerate(heads) if g in h['groups']]
        groups[g]=dict(mean_delta=float(delta[:,members].mean()),ci95=np.quantile(boot[:,members].mean(1),[.025,.975]).tolist(),
                       heads_with_any_eligible_positions=sum(result[j]['active_sequences']>0 for j in members),
                       nominal_positive=sum(result[j]['ci95'][0]>0 for j in members),nominal_negative=sum(result[j]['ci95'][1]<0 for j in members),
                       holm_positive=sum(adjusted[j]<.05 and delta[:,j].mean()>0 for j in members),
                       holm_negative=sum(adjusted[j]<.05 and delta[:,j].mean()<0 for j in members),
                       treatments={name:dict(mean_delta=float(all_delta[:,members,k].mean()),ci95=np.quantile(all_boot[:,members,k].mean(1),[.025,.975]).tolist()) for k,(name,_) in enumerate(specs)})
    summary=dict(n_sequences=n,sequences=[r['sequence'] for r in records],complete=[r['sequence'] for r in records]==list(range(32,128)),
                 dtype=meta['dtype'],cutoff=.02,arms=[name for name,_ in specs],holm_all_arms_family_size=len(heads)*len(specs),heads=result,groups=groups)
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
    np.savez_compressed(out/'paired.npz',delta=delta,all_delta=all_delta,prediction=prediction,selected=selected,removed=removed)
    plot(summary,out)
    print(json.dumps(dict(n=n,groups=groups),indent=2,default=lambda x:x.item()))


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,12),sharey=True);y=np.arange(len(summary['heads']))
    for ax,names in zip(axes,[['zero','double'],['fivefold','open_p005']]):
        for name,offset,color in zip(names,[-.12,.12],['tab:blue','tab:orange']):
            means=np.array([h['treatments'][name]['mean_delta'] for h in summary['heads']])*1e6
            ci=np.array([h['treatments'][name]['ci95'] for h in summary['heads']])*1e6
            ax.errorbar(means,y+offset,xerr=[means-ci[:,0],ci[:,1]-means],fmt='o',color=color,capsize=2,label=name)
        ax.axvline(0,color='black',linewidth=.8);ax.grid(axis='x',alpha=.2);ax.legend()
        ax.set_xlabel('Paired delta loss (micro-nats/token); negative = improvement')
    axes[0].set_yticks(y,[h['id']+f" ({100*h['fraction_of_read_high_tokens']:.1f}%)" for h in summary['heads']]);axes[0].invert_yaxis()
    fig.suptitle(f"Read-high AND original write gate <=0.02; n={summary['n_sequences']}\nLabels: eligible fraction of read-high positions; 95% sequence bootstrap; separate x scales")
    fig.tight_layout();fig.savefig(out/'small_gate_responses.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('adaptive_root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();analyze(a.root,a.adaptive_root,a.out)

"""Actual bidirectional effects: head-by-bin maps and within-head comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
from analyze_write_gate_bets import analyze, holm


def run(root,out):
    analyze(root,out,plot_outputs=False)
    s=json.loads((out/'summary.json').read_text())
    p=np.load(out/'paired.npz');delta=p['delta'];heads=s['heads'];n=len(delta)
    down=s['shifts'].index(-1.);up=s['shifts'].index(1.)
    rawp=stats.ttest_1samp(delta,0,axis=0).pvalue
    rawp=np.where(np.isnan(rawp),1.,rawp)
    allp=holm(rawp.ravel()).reshape(rawp.shape)
    base_ids=list(dict.fromkeys(h['base_head_id'] for h in heads))
    lookup={(h['base_head_id'],h['bin_index']):j for j,h in enumerate(heads)}
    slope=(delta[:,:,up]-delta[:,:,down])/2
    step_mass=(p['gate_change'][:,:,up]-p['gate_change'][:,:,down])/2
    total_slope=slope*p['valid_tokens'][:,None]
    mean_step=step_mass.mean(0)
    normalized=np.divide(total_slope.mean(0),mean_step,out=np.full(len(heads),np.nan),where=mean_step>0)
    pairs=[];contrasts=[]
    for name in base_ids:
        for b in range(6):
            if (name,b) in lookup and (name,b+1) in lookup:
                j,k=lookup[name,b],lookup[name,b+1]
                pairs.append(dict(head=name,lower_bin=b,upper_bin=b+1))
                contrasts.append(slope[:,k]-slope[:,j])
    contrasts=np.stack(contrasts,axis=1)
    weights=np.random.default_rng(9202221).multinomial(n,np.full(n,1/n),size=20000)/n
    denominator=weights@step_mass
    normalized_boot=np.divide(weights@total_slope,denominator,out=np.full_like(denominator,np.nan),where=denominator>0)
    ci=np.quantile(weights@contrasts,[.025,.975],axis=0)
    adjusted=holm(np.nan_to_num(stats.ttest_1samp(contrasts,0,axis=0).pvalue,nan=1.))
    for j,row in enumerate(pairs):
        row.update(mean_slope_difference=float(contrasts[:,j].mean()),ci95=ci[:,j].tolist(),holm_p=float(adjusted[j]))
        a,b=lookup[row['head'],row['lower_bin']],lookup[row['head'],row['upper_bin']]
        difference=normalized[b]-normalized[a]
        influence=((total_slope[:,b]-normalized[b]*step_mass[:,b])/mean_step[b]
                   -(total_slope[:,a]-normalized[a]*step_mass[:,a])/mean_step[a])
        se=np.std(influence,ddof=1)/np.sqrt(n)
        row.update(normalized_difference=float(difference),normalized_ci95=np.nanquantile(normalized_boot[:,b]-normalized_boot[:,a],[.025,.975]).tolist(),
                   normalized_raw_p=float(2*stats.t.sf(abs(difference)/se,n-1)) if se>0 else 1.)
    normalized_p=holm(np.array([r['normalized_raw_p'] for r in pairs]))
    for row,pvalue in zip(pairs,normalized_p):row['normalized_holm_p']=float(pvalue)
    results=[]
    for j,h in enumerate(heads):
        signs=np.sign(delta[:,j].mean(0))
        reliable=(allp[j]<.05).all()
        pattern=('lower_better' if signs[down]<0 and signs[up]>0 else
                 'higher_better' if signs[down]>0 and signs[up]<0 else
                 'both_worse' if (signs>0).all() else 'both_better' if (signs<0).all() else 'unresolved')
        results.append(dict(id=h['id'],head=h['base_head_id'],bin=h['bin_index'],
                            mean_delta=delta[:,j].mean(0).tolist(),holm_both_arms=allp[j].tolist(),
                            point_pattern=pattern,pattern_supported_both_arms=bool(reliable),
                            total_loss_slope_per_unit_gate_mass=float(normalized[j]),
                            normalized_slope_ci95=np.nanquantile(normalized_boot[:,j],[.025,.975]).tolist()))
    audit=dict(n_sequences=n,complete=s['complete'],cells=results,adjacent_within_head=pairs,
               arm_family_size=int(allp.size),adjacent_family_size=len(pairs),
               note='Bins have different head membership and intervention counts. Compare within heads; normalized slope divides total target-loss change by actual gate-probability change mass, with sequence bootstrap and delta-method t/Holm tests. Raw contrasts alone do not distinguish frequency from sensitivity.')
    (out/'binned_summary.json').write_text(json.dumps(audit,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,max(8,len(base_ids)*.22)),sharey=True,layout='constrained')
    bound=max(abs(delta.mean(0)).max()*1e6,1e-6)
    for ax,arm,title in zip(axes,[down,up],['Decrease write probability','Increase write probability']):
        a=np.full((len(base_ids),7),np.nan)
        for row,name in enumerate(base_ids):
            for b in range(7):
                if (name,b) in lookup:
                    j=lookup[name,b];a[row,b]=delta[:,j,arm].mean()*1e6
                    if allp[j,arm]<.05:ax.text(b,row,'*',ha='center',va='center',color='black',fontsize=9)
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d8d8d8')
        im=ax.imshow(a,cmap=cmap,vmin=-bound,vmax=bound,aspect='auto')
        ax.set_title(title);ax.set_xticks(range(7),['0–.02','.02–.05','.05–.1','.1–.2','.2–.4','.4–.6','.6–1'],rotation=45,ha='right')
        ax.set_xlabel('Original write-gate probability bin');ax.set_yticks(range(len(base_ids)),base_ids,fontsize=7)
    fig.colorbar(im,ax=axes,label='Paired delta loss (micro-nats/token); blue improves',shrink=.6)
    fig.suptitle(f"Read-open, strongly negative heads; n={n} paired sequences\nGray: unavailable bins; * Holm across all head-bin × direction tests")
    fig.savefig(out/'binned_actual_effects.png',dpi=180);plt.close(fig)
    print('BOTH_DIRECTION_SUPPORTED',[(r['id'],r['point_pattern']) for r in results if r['pattern_supported_both_arms']])
    print('WITHIN_HEAD_ADJACENT_SUPPORTED',[r for r in pairs if r['holm_p']<.05])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.root,a.out)

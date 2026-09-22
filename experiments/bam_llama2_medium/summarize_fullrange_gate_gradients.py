"""Sequence uncertainty for full-range read-open gradient predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
from analyze_write_gate_bets import holm


def run(root):
    protocol=json.loads((root/'negative_fullrange_protocol.json').read_text())
    a=np.load(root/'fullrange_binned_gradients.npz')['values']
    down=-a[:,:,:,3];full=down.sum(2);n=len(a)
    w=np.random.default_rng(9202230).multinomial(n,np.full(n,1/n),size=20000)/n
    boot=w@full;ci=np.quantile(boot,[.025,.975],axis=0)
    raw=stats.ttest_1samp(full,0,axis=0).pvalue
    adj=holm(np.nan_to_num(raw,nan=1.))
    rows=[]
    for j,h in enumerate(protocol['heads']):
        rows.append(dict(id=h['id'],rho=h['read_write_rho'],predicted_down=float(full[:,j].mean()),
                         ci95=ci[:,j].tolist(),holm_p=float(adj[j]),bins=down[:,j].mean(0).tolist(),
                         selected=int(a[:,j,:,0].sum())))
    groups={}
    for name,lo,hi in [('early',2,7),('middle',8,16),('late',17,22)]:
        ix=[j for j,h in enumerate(protocol['heads']) if lo<=h['layer']<=hi]
        b=boot[:,ix].mean(1)
        groups[name]=dict(heads=len(ix),predicted_down_mean=float(full[:,ix].mean()),
                          ci95=np.quantile(b,[.025,.975]).tolist(),
                          decrease_point_count=int((full[:,ix].mean(0)<0).sum()),
                          holm_decrease=int(sum(adj[j]<.05 and full[:,j].mean()<0 for j in ix)),
                          holm_increase=int(sum(adj[j]<.05 and full[:,j].mean()>0 for j in ix)))
    result=dict(n_sequences=n,head_count=len(rows),predicted_down_mean=float(full.mean()),
                ci95=np.quantile(boot.mean(1),[.025,.975]).tolist(),
                decrease_point_count=int((full.mean(0)<0).sum()),increase_point_count=int((full.mean(0)>0).sum()),
                holm_significant=[r for r in rows if r['holm_p']<.05],groups=groups,heads=rows,
                scope='Read-open valid positions; gradient prediction for g-h, h=min(.01,.1g,.1(1-g)); no write upper cutoff')
    (root/'negative_fullrange_gradient_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2))
    print('Largest predictions:',sorted(rows,key=lambda x:abs(x['predicted_down']),reverse=True)[:8])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    run(p.parse_args().root)

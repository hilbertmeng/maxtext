"""Summarize LocalO-only writeback gradients and compare other write components."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp
STATES=['read_high_write_low','read_low_write_high','both_low','both_high']

def one(task):
    path,prior,heads=task;i=int(path.stem.split('_')[-1]);ls=[h['layer'] for h in heads];hs=[h['head'] for h in heads]
    with np.load(path) as f:
        d=f['gradient'][ls,:,hs].astype(np.float64);g=f['write_gate'][ls,:,hs].astype(np.float64);r=f['read_gate'][ls,:,hs].astype(np.float64);v=f['valid'];primal=float(f['primal_delta'])
    with np.load(Path(prior)/'position_gradient_worker'/f'gradient_{i:03d}.npz') as f:
        whole=f['gradient'][ls,:,hs].astype(np.float64)*g
        gate_error=float(np.max(abs(g-f['write_gate'][ls,:,hs])))
        read_error=float(np.max(abs(r-f['read_gate'][ls,:,hs])))
        np.testing.assert_array_equal(v,f['valid'])
    assert gate_error<1e-6 and read_error<1e-6,(gate_error,read_error)
    sums=[];counts=[];positive=[]
    for t in [.05,.1,.2]:
        masks=[v&(r>=t)&(g<t),v&(r<t)&(g>=t),v&(r<t)&(g<t),v&(r>=t)&(g>=t)]
        sums.append(np.stack([np.stack([np.where(m,z,0).sum(-1) for z in [d,whole,whole-d]],axis=-1) for m in masks],axis=1))
        counts.append(np.stack([m.sum(-1) for m in masks],axis=1));positive.append(np.stack([(m&(d>0)).sum(-1) for m in masks],axis=1))
    return np.stack(sums,axis=1),np.stack(counts,axis=1),np.stack(positive,axis=1),primal,gate_error,read_error

def holm(p):
    p=np.nan_to_num(np.array(p),nan=1);ix=np.argsort(p);a=np.zeros_like(p);a[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*np.arange(len(p),0,-1)));return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('prior',type=Path);ap.add_argument('--partial',action='store_true');a=ap.parse_args();out=a.root/'analysis';out.mkdir(exist_ok=True)
    heads=json.loads((a.root/'protocol.json').read_text())['heads'];paths=sorted(a.root.glob('shard*/gradient_*.npz'))
    ids=[int(p.stem.split('_')[-1]) for p in paths];assert len(ids)==len(set(ids))
    if not a.partial:assert sorted(ids)==list(range(32,128)),ids
    with concurrent.futures.ProcessPoolExecutor(8) as pool:rows=list(pool.map(one,[(p,str(a.prior),heads) for p in paths]))
    sums=np.stack([x[0] for x in rows]);counts=np.stack([x[1] for x in rows]);pos=np.stack([x[2] for x in rows]);n=len(rows)
    np.savez_compressed(out/'statistics.npz',sums=sums,counts=counts,positive=pos,sequences=ids)
    weights=np.random.default_rng(9876).multinomial(n,np.full(n,1/n),size=20000)/n
    pooled=-.1*sums.mean(1);ci=np.quantile(weights@pooled.reshape(n,-1),[.025,.975],axis=0).reshape(2,3,4,3)
    hp=holm(ttest_1samp(sums[:,:,1,:,0],0,axis=0).pvalue.reshape(-1)).reshape(len(heads),4)
    gp=holm(ttest_1samp(pooled[:,1,:,0],0,axis=0).pvalue)
    result=dict(n_sequences=n,sequences=ids,heads=heads,states=STATES,components=['localO_only','whole_write_same_fraction','nonLocalO_by_gradient_difference'],thresholds=[],head_primary=[],max_primal_drift=max(abs(x[3]) for x in rows),max_gate_context_error=max(x[4] for x in rows),max_read_context_error=max(x[5] for x in rows))
    for k,t in enumerate([.05,.1,.2]):
        states=[]
        for j,name in enumerate(STATES):
            direction=sums[:,:,k,j,0].mean(0);c=counts[:,:,k,j].sum();component=[]
            for b in range(3):component.append(dict(predicted_down_mean=float(pooled[:,k,j,b].mean()),ci95=ci[:,k,j,b].tolist()))
            states.append(dict(state=name,count=int(c),fraction=float(c/counts[:,:,k].sum()),positive_gradient_fraction=float(pos[:,:,k,j].sum()/c),components=component,improve_heads=int((direction>0).sum()),worsen_heads=int((direction<0).sum()),zero_heads=int((direction==0).sum()),holm_p_four_states=float(gp[j]) if k==1 else None))
        result['thresholds'].append(dict(threshold=t,states=states))
    for h,head in enumerate(heads):result['head_primary'].append(dict(id=f"L{head['layer']}H{head['head']}",predicted_down=(-.1*sums[:,h,1,:,0].mean(0)).tolist(),holm_p_348=hp[h].tolist(),counts=counts[:,h,1].sum(0).tolist()))
    # Paired ±10% calibration; reconstruct masks on every position just as in
    # the finite runner, rather than silently excluding target-masked positions.
    finite=[]
    byid=dict(zip(ids,paths))
    for path in sorted(a.root.glob('shard*/finite_*.npz')):
        i=int(path.stem.split('_')[-1])
        if i not in byid:continue
        with np.load(byid[i]) as f:d=f['gradient'].astype(np.float64);g=f['write_gate'];r=f['read_gate']
        with np.load(path) as f:arms=f['ids'];deltas=f['deltas'];assert abs(deltas[:2]).max()<1e-7
        secants=[];pred=[]
        for k in range(2,len(arms),2):
            l,h,st,_=arms[k];l,h,st=map(int,[l,h,st]);rh=r[l,:,h]>=.1;wh=g[l,:,h]>=.1;category=np.where(rh,np.where(wh,3,0),np.where(wh,1,2));pred.append(d[l,category==st,h].sum());secants.append((deltas[k+1]-deltas[k])/.2)
        finite.append((i,pred,secants,deltas[2:]))
    if finite:
        pred=np.array([v[1] for v in finite]);actual=np.array([v[2] for v in finite]);pm=pred.mean(0);am=actual.mean(0);active=(abs(pm)+abs(am))>1e-10
        result['calibration']=dict(n_sequences=len(finite),cells=int(active.sum()),mean_sign_agreement=float((np.sign(pm[active])==np.sign(am[active])).mean()),magnitude_weighted_relative_error=float(abs(pm-am).sum()/abs(pm).sum()),mean_absolute_error=float(abs(pm-am).mean()),predicted_mean=pm.tolist(),finite_mean=am.tolist(),sequences=[v[0] for v in finite])
        np.savez_compressed(out/'calibration.npz',predicted=pred,finite=actual,deltas=np.array([v[3] for v in finite]))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['heads','head_primary','sequences']},indent=2))

if __name__=='__main__':main()

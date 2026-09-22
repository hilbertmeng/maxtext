"""Connect joint continuous-gate states to existing FP32 loss gradients."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,time
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

STATES=['read_high_write_low','read_low_write_high','both_low','both_high']
THRESHOLDS=[.05,.1,.2]
METRICS=['count','positive','negative','zero','sum_gradient','sum_adaptive_gradient','step_mass']

def one(task):
    path,heads=task
    ls=[h['layer'] for h in heads];hs=[h['head'] for h in heads]
    with np.load(path) as f:
        g=f['write_gate'][ls,:,hs].astype(np.float64);r=f['read_gate'][ls,:,hs].astype(np.float64)
        d=f['gradient'][ls,:,hs].astype(np.float64);valid=f['valid']
    step=np.minimum(.01,np.minimum(.1*g,.1*(1-g)))
    vals=np.stack([np.ones_like(d),d>0,d<0,d==0,d,d*step,step],axis=-1)
    out=[]
    for t in THRESHOLDS:
        masks=[(r>=t)&(g<t),(r<t)&(g>=t),(r<t)&(g<t),(r>=t)&(g>=t)]
        out.append(np.stack([np.where((m&valid)[...,None],vals,0).sum(axis=1) for m in masks],axis=1))
    out=np.stack(out,axis=1)
    np.testing.assert_allclose(out[:,:, :,4].sum(-1),np.broadcast_to((d*valid).sum(1)[:,None],(len(heads),3)),atol=1e-13)
    return out

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.zeros_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)));return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('prior',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    heads=[h for h in json.loads((args.prior/'protocol.json').read_text())['all_head_stats'] if 3<=h['layer']<=16 and h['read_write_rho']<=-.3]
    tasks=[(args.prior/'position_gradient_worker'/f'gradient_{i:03d}.npz',heads) for i in range(32,128)]
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(8) as pool:values=np.stack(list(pool.map(one,tasks)))
    np.testing.assert_array_equal(one(tasks[0]),values[0])
    np.savez_compressed(args.out/'statistics.npz',values=values,metrics=METRICS,thresholds=THRESHOLDS,states=STATES)
    weights=np.random.default_rng(9876).multinomial(96,np.full(96,1/96),size=20000)/96
    total=values.sum((0,1));deriv=values[...,5];headmean=deriv.mean(0)
    headp=np.nan_to_num(ttest_1samp(deriv[:,:,1,:],0,axis=0).pvalue,nan=1)
    headadjusted=holm(headp.reshape(-1)).reshape(headp.shape)
    pooled=deriv.mean(1);flat=pooled.reshape(96,-1);ci=np.quantile(weights@flat,[.025,.975],axis=0).reshape(2,3,4)
    pooledp=np.nan_to_num(ttest_1samp(pooled[:,1,:],0,axis=0).pvalue,nan=1);pooledadjusted=holm(pooledp)
    rows=[]
    for k,t in enumerate(THRESHOLDS):
        states=[]
        for j,name in enumerate(STATES):
            a=total[k,j];n=a[0];down=-pooled[:,k,j].mean()
            states.append(dict(state=name,count=int(n),fraction=n/total[k,:,0].sum(),positive_gradient_fraction=a[1]/n if n else None,
                predicted_down_mean=down,predicted_down_ci95=[float(-ci[1,k,j]),float(-ci[0,k,j])],predicted_up_mean=-down,
                mean_raw_gradient=a[4]/n if n else None,down_improves_heads=int((headmean[:,k,j]>0).sum()),down_worsens_heads=int((headmean[:,k,j]<0).sum()),zero_heads=int((headmean[:,k,j]==0).sum()),
                holm_p_four_states=float(pooledadjusted[j]) if k==1 else None))
        rows.append(dict(threshold=t,states=states))
    result=dict(n_sequences=96,heads=heads,definition='FP32 baseline gates; valid positions; h=min(.01,.1*g,.1*(1-g)); mean of independent head responses; first-order approximation, not finite measured loss',rows=rows,per_head_primary=[],seconds=time.perf_counter()-start)
    for i,h in enumerate(heads):
        result['per_head_primary'].append(dict(id=f"L{h['layer']}H{h['head']}",predicted_down=(-headmean[i,1]).tolist(),holm_p_348=headadjusted[i].tolist(),counts=values[:,i,1,:,0].sum(0).astype(int).tolist()))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(rows=rows,seconds=result['seconds']),indent=2))

if __name__=='__main__':main()

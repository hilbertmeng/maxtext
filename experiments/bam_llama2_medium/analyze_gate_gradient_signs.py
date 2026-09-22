"""Per-position probability-gradient signs; sequence is the uncertainty unit."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np

SCOPES=['valid_all','read_high','read_high_write_small']


def one(task):
    path,heads,adaptive_root=task
    with np.load(path) as f:
        grad=f['gradient'];gate=f['write_gate'];read=f['read_gate'];valid=f['valid']
    counts=np.zeros((len(heads),3,4),np.int64);sums=np.zeros((len(heads),3),np.float64)
    sequence=int(path.stem.split('_')[-1]);ref=json.loads((adaptive_root/f'seq_{sequence:03d}.json').read_text())
    old={x['id']:x['derivative'] for x in ref['gradients']};errors=[]
    for j,h in enumerate(heads):
        l,n=h['layer'],h['head'];g=gate[l,:,n];r=read[l,:,n];v=grad[l,:,n]
        high=valid&(r>=h['threshold']);small=high&(g<=.02)
        for k,mask in enumerate([valid,high,small]):
            x=v[mask];counts[j,k]=[(x>0).sum(),(x<0).sum(),(x==0).sum(),len(x)];sums[j,k]=x.sum(dtype=np.float64)
        step=np.minimum(.01,np.minimum(.1*g,.1*(1-g)))
        reconstructed=float(np.sum(v.astype(np.float64)*step*(r>=h['threshold'])))
        errors.append(reconstructed-old[h['id']] if h['id'] in old else np.nan)
    return sequence,counts,sums,np.array(errors)


def analyze(root,adaptive_root,out,workers,protocol_path=None):
    meta=json.loads((root/'metadata.json').read_text());protocol=json.loads(protocol_path.read_text()) if protocol_path else meta['protocol'];heads=protocol['heads']
    paths=sorted(root.glob('gradient_*.npz'));assert len(paths)>=2
    tasks=[(p,heads,adaptive_root) for p in paths]
    # Check numerical equivalence and measured parallel throughput on a small subset.
    start=time.perf_counter();serial=[one(t) for t in tasks[:min(8,len(tasks))]];serial_time=time.perf_counter()-start
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:parallel=list(pool.map(one,tasks[:len(serial)]))
    parallel_time=time.perf_counter()-start
    for a,b in zip(serial,parallel):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    remaining=tasks[len(serial):]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:records=serial+list(pool.map(one,remaining))
    ids=[r[0] for r in records];counts=np.stack([r[1] for r in records]);sums=np.stack([r[2] for r in records]);errors=np.stack([r[3] for r in records])
    total=counts.sum(0);n=len(records)
    weights=np.random.default_rng(9202204).multinomial(n,np.full(n,1/n),size=10000)
    positives=weights@counts[:,:,:,0].reshape(n,-1)
    nonzero=weights@(counts[:,:,:,0]+counts[:,:,:,1]).reshape(n,-1)
    proportions=np.divide(positives,nonzero,out=np.full_like(positives,np.nan,dtype=float),where=nonzero>0).reshape(-1,len(heads),3)
    result=[]
    for j,h in enumerate(heads):
        scopes={}
        for k,name in enumerate(SCOPES):
            pos,neg,zero,ntotal=map(int,total[j,k]);nonzero_count=pos+neg
            bootstrap=proportions[:,j,k];bootstrap=bootstrap[np.isfinite(bootstrap)]
            scopes[name]=dict(positive=pos,negative=neg,zero=zero,total=ntotal,
                              positive_fraction_nonzero=pos/nonzero_count if nonzero_count else None,
                              positive_fraction_ci95=np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap) else None,
                              mean_probability_gradient=float(sums[:,j,k].sum()/ntotal) if ntotal else None,
                              mean_probability_gradient_sign=int(np.sign(sums[:,j,k].sum())) if ntotal else None,
                              active_sequences=int(np.count_nonzero(counts[:,j,k,3])))
        result.append(dict(id=h['id'],groups=h['groups'],scopes=scopes))
    summary=dict(n_sequences=n,sequences=ids,complete=ids==list(range(32,128)),heads=result,
                 directional_reconstruction_mean_abs_error=float(np.nanmean(abs(errors))),directional_reconstruction_max_abs_error=float(np.nanmax(abs(errors))),
                 directional_reconstruction_comparable_heads=int(np.isfinite(errors).any(0).sum()),
                 cpu_check=dict(workers=workers,serial_seconds=serial_time,parallel_seconds=parallel_time,speedup=serial_time/parallel_time,numerically_equal=True))
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'counts.npz',counts=counts,sums=sums,reconstruction_errors=errors)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['heads','sequences']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('adaptive_root',type=Path);p.add_argument('out',type=Path);p.add_argument('--workers',type=int,default=8);p.add_argument('--protocol',type=Path)
    a=p.parse_args();analyze(a.root,a.adaptive_root,a.out,a.workers,a.protocol)

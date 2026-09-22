"""Address and read/write-content angle distributions, retaining all gate states."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,time
from pathlib import Path
import numpy as np

STATES=['read_high_write_low','read_low_write_high','both_low','both_high']
KINDS=['address','read_vs_total_write','read_vs_nonlocalO_write']
QS=[.01,.05,.1,.25,.5,.75,.9,.95,.99]

def one(task):
    root,seq,heads=task
    x=np.load(Path(root)/f'sample_{seq:03d}.npy',mmap_mode='r')[[h['layer'] for h in heads],:,[h['head'] for h in heads]].astype(np.float64)
    # Gram order 00,01,02,03,11,12,13,22,23,33.
    nonlocal2=x[...,0]+x[...,4]+x[...,7]+2*(x[...,1]+x[...,2]+x[...,5])
    cross=x[...,3]+x[...,6]+x[...,8];o2=x[...,9]
    total2=nonlocal2+o2+2*cross
    def divide(a,b):return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    cos=np.stack([x[...,12],divide(cross+o2,np.sqrt(np.maximum(o2*total2,0))),divide(cross,np.sqrt(np.maximum(o2*nonlocal2,0)))],axis=-1)
    angles=np.degrees(np.arccos(np.clip(cos,-1,1))).astype(np.float32)
    return angles,x[...,26].astype(np.float32),x[...,10].astype(np.float32)

def summary(a):
    a=a[np.isfinite(a)]
    if not len(a):return dict(n=0,quantiles=None,mean=None,obtuse_fraction=None)
    return dict(n=len(a),quantiles=np.quantile(a,QS).tolist(),mean=float(a.mean(dtype=np.float64)),obtuse_fraction=float((a>90).mean()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('geometry',type=Path);ap.add_argument('prior',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    heads=[h for h in json.loads((a.prior/'protocol.json').read_text())['all_head_stats'] if 3<=h['layer']<=16 and h['read_write_rho']<=-.3]
    tasks=[(str(a.geometry),i,heads) for i in range(32,128)];start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(8) as pool:records=list(pool.map(one,tasks))
    angles=np.stack([r[0] for r in records]);read=np.stack([r[1] for r in records]);write=np.stack([r[2] for r in records]);del records
    valid=np.load(a.geometry/'valid.npy')[32:128,None,:]
    result=dict(heads=heads,states=STATES,kinds=KINDS,quantile_levels=QS,n_sequences=96,thresholds=[],angle_units='degrees',read_content='actual LocalO, whose positive scalar gate does not change direction',write_content='sum of four components in 48-D data coordinates; scalar write RMS and gate do not change direction',address='C8 read key lifted by abs_v_cache_projection into full M address coordinates, versus actual normalized P_loc',per_head=[])
    hist=np.zeros((3,4,3,180),dtype=np.int64);phist=np.zeros((len(heads),4,3,180),dtype=np.int64)
    # Sequence/head/state/kind counts, sums and obtuse counts for cluster bootstrap.
    moments=np.zeros((96,len(heads),4,3,3))
    for ti,t in enumerate([.05,.1,.2]):
        masks=[valid&(read>=t)&(write<t),valid&(read<t)&(write>=t),valid&(read<t)&(write<t),valid&(read>=t)&(write>=t)]
        pooled=[]
        for st,mask in enumerate(masks):
            metrics=[]
            for k in range(3):
                z=angles[...,k];picked=z[mask];metrics.append(summary(picked));hist[ti,st,k]=np.histogram(picked[np.isfinite(picked)],bins=np.arange(181))[0]
                if ti==1:
                    ok=mask&np.isfinite(z);moments[:,:,st,k,0]=ok.sum(-1);moments[:,:,st,k,1]=np.where(ok,z,0).sum(-1,dtype=np.float64);moments[:,:,st,k,2]=(ok&(z>90)).sum(-1)
            pooled.append(dict(state=STATES[st],positions=int(mask.sum()),metrics=metrics))
        result['thresholds'].append(dict(threshold=t,states=pooled))
        if ti==1:
            for j,h in enumerate(heads):
                stats=[]
                for st,mask in enumerate(masks):
                    metrics=[]
                    for k in range(3):
                        z=angles[:,j,:,k][mask[:,j]];metrics.append(summary(z));phist[j,st,k]=np.histogram(z[np.isfinite(z)],bins=np.arange(181))[0]
                    stats.append(metrics)
                result['per_head'].append(dict(id=f"L{h['layer']}H{h['head']}",states=stats))
    # Compare HH minus LH within each head, with equal head weights to avoid pooling confounds.
    weights=np.random.default_rng(9876).multinomial(96,np.full(96,1/96),size=4000)
    comparisons=[]
    for k in range(3):
        count=moments[:,:,:,k,0];sums=moments[:,:,:,k,1]
        active=(count[:,:,1].sum(0)>=256)&(count[:,:,3].sum(0)>=256)&((count[:,:,1]>0).sum(0)>=8)&((count[:,:,3]>0).sum(0)>=8)
        ids=np.flatnonzero(active);mean=sums.sum(0)/np.maximum(count.sum(0),1)
        delta=mean[active,3]-mean[active,1]
        boot=[]
        for st in [1,3]:boot.append((weights@sums[:,active,st])/np.maximum(weights@count[:,active,st],1))
        bd=(boot[1]-boot[0]).mean(-1)
        comparisons.append(dict(kind=KINDS[k],eligible_heads=ids.tolist(),n_heads=len(ids),HH_minus_LH_mean_angle=float(delta.mean()),ci95=np.quantile(bd,[.025,.975]).tolist(),HH_smaller_heads=int((delta<0).sum()),HH_larger_heads=int((delta>0).sum()),per_head_deltas=delta.tolist()))
    result['within_head_comparison']=comparisons;result['seconds']=time.perf_counter()-start
    np.savez_compressed(a.out/'distributions.npz',hist=hist,per_head_hist=phist,moments=moments,bin_edges=np.arange(181))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(primary=result['thresholds'][1],comparison=comparisons,seconds=result['seconds']),indent=2))

if __name__=='__main__':main()

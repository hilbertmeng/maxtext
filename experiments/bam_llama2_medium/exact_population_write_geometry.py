"""Exact pooled quantiles from all active tokens (not averages of head quantiles)."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,concurrent.futures,time
from pathlib import Path
import numpy as np
from analyze_write_geometry import transform,FN
from explore_write_geometry import EX
r=Path(sys.argv[1]);m=json.loads((r/'metadata.json').read_text());ix={k:i for i,k in enumerate(m['fields'])};valid=np.load(r/'valid.npy');arrays=[];start=time.perf_counter()
for i in range(m['n']):
 raw=np.load(r/f'sample_{i:03d}.npy',mmap_mode='r')[2:23]
 mask=valid[i][None,:,None]&(raw[...,ix['read_o_gate']]>=.1)&(raw[...,ix['write_gate']]>=.1)&(raw[...,ix['gram_33']]>1e-20)
 v=raw[mask];f,_=transform(v,m['fields']);arrays.append(np.concatenate([f,v[:,[ix[k] for k in EX]]],-1))
z=np.concatenate(arrays);del arrays
q=np.unique(np.r_[np.linspace(0,1,1001),[.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999]])
def calc(j):return np.nanquantile(z[:,j],q)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:qs=np.stack(list(p.map(calc,range(z.shape[1]))))
np.savez_compressed(r/'pooled_exact.npz',names=FN+EX,quantile_levels=q,quantiles=qs,count=len(z),layers=np.arange(2,23))
print(json.dumps(dict(count=len(z),seconds=time.perf_counter()-start,quantiles={name:qs[j,np.searchsorted(q,[.05,.25,.5,.75,.95])].tolist() for j,name in enumerate(FN+EX)})),flush=True)

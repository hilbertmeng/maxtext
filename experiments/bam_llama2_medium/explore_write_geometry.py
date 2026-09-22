"""Read/write full distributions and grouped, sequence-held-out gate models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,argparse,time,concurrent.futures
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits
from analyze_write_geometry import transform,spearman,FN
threadpool_limits(1)
QN=[0,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1]
EX=['key_cos','content_cos','old_at_unit_p_norm','o_write_at_unit_p_norm','write_ratio','relative_parallel','read_o_gate','read_v_gate','write_gate']

def run(task):
 root,layer=task;root=Path(root);start=time.perf_counter();meta=json.loads((root/'metadata.json').read_text());n=meta['n'];ix={s:i for i,s in enumerate(meta['fields'])};valid=np.load(root/'valid.npy');pos=np.load(root/'positions.npy')
 raw=np.stack([np.load(root/f'sample_{i:03d}.npy',mmap_mode='r')[layer] for i in range(n)]);f,norms=transform(raw,meta['fields']);go=raw[...,ix['read_o_gate']];gv=raw[...,ix['read_v_gate']];gw=raw[...,ix['write_gate']]
 thresholds=[.05,.1,.2];q=np.full((3,16,len(EX),len(QN)),np.nan);hist=np.zeros((3,16,len(EX),128),np.int64);counts=np.zeros((3,n,16,9),np.int64);largest=np.zeros((3,16,4),np.int64);joint=np.zeros((3,16,64,64),np.int64)
 geom_edges=np.stack([np.linspace(-10,3,513) if name.startswith('lognorm') else np.linspace(-1,1,513) if name.startswith('cos') else np.linspace(0,1,513) for name in FN]);geom_hist=np.zeros((3,len(FN),512),np.int64);pooled_q=np.full((3,len(FN)+len(EX),len(QN)),np.nan)
 edges=np.stack([np.linspace(-1,1,129) if 'cos' in s else np.linspace(0,1,129) if 'gate' in s else np.linspace(-5,5,129) if s=='relative_parallel' else np.linspace(-6,6,129) for s in EX]);eb=np.linspace(-1,1,65);rb=np.linspace(-6,6,65)
 for ti,t in enumerate(thresholds):
  mask=valid[...,None]&(go>=t)&(gw>=t)&(norms[...,3]>1e-10)
  for j,name in enumerate(FN+EX):
   z=(f[...,j] if j<len(FN) else raw[...,ix[name]])[mask];z=z[np.isfinite(z)]
   if len(z):
    pooled_q[ti,j]=np.quantile(z,QN)
    if j<len(FN):geom_hist[ti,j]=np.histogram(z,geom_edges[j])[0]
  for h in range(16):
   m=mask[:,:,h]
   if not m.any():continue
   for j,s in enumerate(EX):
    z=raw[:,:,h,ix[s]][m];z=z[np.isfinite(z)];q[ti,h,j]=np.quantile(z,QN) if len(z) else np.nan
    if not ('cos' in s or 'gate' in s or s=='relative_parallel'):z=np.log10(np.maximum(z,1e-30))
    hist[ti,h,j]=np.histogram(z,edges[j])[0]
   for c in range(4):largest[ti,h,c]=np.sum(np.argmax(norms[:,:,h],axis=-1)[m]==c)
   cc=raw[:,:,h,ix['content_cos']];a=raw[:,:,h,ix['relative_parallel']];rr=raw[:,:,h,ix['write_ratio']]
   good=m&np.isfinite(cc)&np.isfinite(rr)
   conditions=[m,good,good&(cc<0),good&(a < -1),good&(a>0),good&(a<=0)&(a>=-1),good&(rr>1),good&(raw[:,:,h,ix['key_cos']]*cc<0),good&(np.abs(raw[:,:,h,ix['key_cos']])<.2)]
   for j,c in enumerate(conditions):counts[ti,:,h,j]=c.sum(axis=1)
   joint[ti,h]=np.histogram2d(cc[good],np.log10(np.maximum(rr[good],1e-30)),bins=[eb,rb])[0]
 # Geometry/gate model decomposition. Gates are added controls, never write-gated output magnitudes.
 rng=np.random.default_rng(9876+layer);xx=[];yy=[];split=[];sp=[]
 mask=valid[...,None]&(go>=.1)&(gw>=.1)&(norms[...,3]>1e-10)
 for i in range(n):
  for h in range(16):
   ids=np.flatnonzero(mask[i,:,h]);ids=rng.choice(ids,min(128,len(ids)),replace=False)
   if not len(ids):continue
   z=f[i,ids,h].copy();z[:,[8,11,12]]=0
   xx.append(np.column_stack([np.full(len(ids),h),np.log1p(pos[i,ids]),go[i,ids,h],gv[i,ids,h],z,np.log10(np.maximum(raw[i,ids,h,ix['sum_norm']],1e-20))]));yy.append(gw[i,ids,h]);split.extend([i<32]*len(ids));sp.extend([i]*len(ids))
 scores={};partial=np.full((2,16,len(FN)),np.nan)
 if xx:
  X=np.nan_to_num(np.concatenate(xx),nan=0);Y=np.concatenate(yy);tr=np.array(split);seq=np.array(sp)
  groups={'head_position':[0,1],'read_gate_controls':[0,1,2,3],'total_amplitude':[0,1,20],'amplitude':[0,1]+list(range(4,8)),'composition':[0,1]+list(range(8,12)),'angles':[0,1]+list(range(12,18)),'geometry':[0,1]+list(range(4,20)),'geometry_and_read_gates':list(range(20))}
  preds={}
  if tr.sum()>=1000 and (~tr).sum()>=1000:
   for name,cols in groups.items():
    m=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=10,min_samples_leaf=100,learning_rate=.08,early_stopping=False,random_state=9876)
    m.fit(X[tr][:,cols],Y[tr]);p=m.predict(X[~tr][:,cols]);preds[name]=p;scores[name]=float(mean_squared_error(Y[~tr],p))
   # Sequence SSE permits uncertainty without pretending tokens independent.
   mse_seq=np.full((n-32,len(groups),2),np.nan)
   for i in range(32,n):
    take=seq[~tr]==i
    for j,name in enumerate(groups):mse_seq[i-32,j]=[np.square(Y[~tr][take]-preds[name][take]).sum(),take.sum()]
  else:mse_seq=np.zeros((n-32,0,2))
  # Position-stratified rank relations: demean ranks within 8 fixed positional bins and head.
  from scipy.stats import rankdata
  for si,ss in enumerate([tr,~tr]):
   for h in range(16):
    take=ss&(X[:,0]==h);z=X[take];y=Y[take]
    if len(y)<100:continue
    bins=np.minimum((np.expm1(z[:,1])/256).astype(int),7);yr=rankdata(y)
    for b in range(8):
     sel=bins==b
     if sel.any():yr[sel]-=yr[sel].mean()
    for j in range(len(FN)):
     xr=rankdata(z[:,4+j]);
     for b in range(8):
      sel=bins==b
      if sel.any():xr[sel]-=xr[sel].mean()
     den=np.linalg.norm(xr)*np.linalg.norm(yr)
     if den>0:partial[si,h,j]=xr@yr/den
 else:mse_seq=np.zeros((n-32,0,2));groups={}
 dest=root/'exploration';dest.mkdir(exist_ok=True)
 np.savez_compressed(dest/f'layer_{layer:02d}.npz',quantiles=q,quantile_levels=QN,geometry_hist=geom_hist,geometry_edges=geom_edges,pooled_quantiles=pooled_q,pooled_names=FN+EX,hist=hist,edges=edges,names=EX,counts=counts,largest=largest,joint=joint,joint_cos_edges=eb,joint_logratio_edges=rb,partial_correlations=partial,mse_sequence=mse_seq,model_names=list(groups))
 result=dict(layer=layer,seconds=time.perf_counter()-start,mse=scores)
 (dest/f'layer_{layer:02d}.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=24);a=p.parse_args()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  for x in pool.map(run,[(a.root,l) for l in range(24)]):print(json.dumps(x),flush=True)

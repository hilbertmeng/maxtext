"""Multicore exhaustive distributions and sequence-held-out gate discovery."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits
threadpool_limits(1)
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
COMP=['sv','bam_self','bam_other','local_o']
FN=[f'lognorm_{s}' for s in COMP]+[f'share_{s}' for s in COMP]+[f'cos_{COMP[i]}__{COMP[j]}' for i,j in PAIRS]+['coherence','self_alpha']

def transform(raw,fields):
 ix={s:i for i,s in enumerate(fields)};norms=np.sqrt(np.maximum(np.stack([raw[...,ix[f'gram_{i}{i}']] for i in range(4)],-1),0));total=norms.sum(-1)
 cos=[]
 for i,j in PAIRS:
  den=norms[...,i]*norms[...,j];cos.append(np.divide(raw[...,ix[f'gram_{i}{j}']],den,out=np.full_like(den,np.nan),where=den>1e-20).clip(-1,1))
 normsum2=sum(raw[...,ix[f'gram_{i}{j}']]*(1 if i==j else 2) for i in range(4) for j in range(i,4))
 coherence=np.sqrt(np.maximum(normsum2,0))/np.maximum(total,1e-20)
 feat=np.concatenate([np.log10(np.maximum(norms,1e-10)),norms/np.maximum(total[...,None],1e-20),np.stack(cos,-1),coherence[...,None],raw[...,ix['self_alpha'],None]],-1)
 return feat,norms

def spearman(x,y):
 good=np.isfinite(x)&np.isfinite(y)
 if good.sum()<20 or np.ptp(x[good])==0 or np.ptp(y[good])==0:return np.nan
 a=rankdata(x[good]);b=rankdata(y[good]);a-=a.mean();b-=b.mean();return float(a@b/np.sqrt((a@a)*(b@b)))

def analyze_layer(task):
 root,layer,n,threshold,descriptive=task;root=Path(root);start=time.perf_counter();meta=json.loads((root/'metadata.json').read_text());fields=meta['fields'];ix={s:i for i,s in enumerate(fields)}
 valid=np.load(root/'valid.npy')[:n];position=np.load(root/'positions.npy')[:n]
 raw=np.stack([np.load(root/f'sample_{i:03d}.npy',mmap_mode='r')[layer] for i in range(n)])
 feat,norms=transform(raw,fields);gate=raw[...,ix['write_gate']];allmask=np.broadcast_to(valid[...,None],gate.shape)
 mask=allmask&(raw[...,ix['read_o_gate']]>=threshold)&(gate>=threshold)
 if threshold>0:mask &= norms[...,3]>1e-10
 coverage=mask.sum(axis=1);coverage_sequences=(coverage>0).sum(axis=0)
 quantiles=np.array([0,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1.]);q=np.full((16,len(FN),len(quantiles)),np.nan);gq=np.full((16,len(quantiles)),np.nan)
 hist=np.zeros((16,len(FN),32,32),np.int64);curves=np.full((16,len(FN),8,7),np.nan)
 edges=[]
 for name in FN:
  edges.append(np.linspace(-10,3,33) if name.startswith('lognorm') else np.linspace(-1,1,33) if name.startswith('cos') else np.linspace(0,1,33))
 seq_curves=np.full((n,16,len(FN),8,2),np.nan,np.float32)
 for h in range(16):
  hv=mask[:,:,h];y=gate[:,:,h][hv]
  if not len(y):continue
  gq[h]=np.quantile(y,quantiles)
  for j,name in enumerate(FN):
   x=feat[:,:,h,j][hv];good=np.isfinite(x);z=x[good];gy=y[good]
   if not len(z):continue
   q[h,j]=np.quantile(z,quantiles);hist[h,j]=np.histogram2d(z,gy,bins=[edges[j],np.linspace(0,1,33)])[0]
   # Freeze exploratory quantile bins on the first32 sequences, validate on next32.
   trainx=feat[:32,:,h,j][hv[:32]];trainx=trainx[np.isfinite(trainx)]
   if not len(trainx):continue
   bins=np.quantile(trainx,np.linspace(0,1,9));bins[0]=-np.inf;bins[-1]=np.inf
   for i in range(n):
    xx=feat[i,:,h,j];yy=gate[i,:,h];ok=hv[i]&np.isfinite(xx);bi=np.searchsorted(bins[1:-1],xx,side='right')
    for b in range(8):
     take=ok&(bi==b);seq_curves[i,h,j,b]=[take.sum(),yy[take].sum()]
   for b in range(8):
    take=np.isfinite(x)&(np.searchsorted(bins[1:-1],x,side='right')==b)
    if take.sum():curves[h,j,b]=np.quantile(y[take],[.05,.1,.25,.5,.75,.9,.95])
 # Deterministic balanced token sample for exploratory modeling and rank relations.
 rng=np.random.default_rng(9876+layer)
 xx=np.full((n,128,16,len(FN)),np.nan,np.float32);yy=np.full((n,128,16),np.nan,np.float32);pp=np.zeros((n,128,16),np.float32)
 for i in range(n):
  for h in range(16):
   candidates=np.flatnonzero(mask[i,:,h]);count=min(128,len(candidates))
   if count:
    take=rng.choice(candidates,count,replace=False);xx[i,:count,h]=feat[i,take,h];yy[i,:count,h]=gate[i,take,h];pp[i,:count,h]=position[i,take]
 cor=np.full((2,16,len(FN)),np.nan)
 for split,sl in enumerate([slice(0,32),slice(32,n)]):
  for h in range(16):
   for j in range(len(FN)):cor[split,h,j]=spearman(xx[sl,:,h,j].ravel(),yy[sl,:,h].ravel())
 head=np.broadcast_to(np.arange(16)[None,None,:],yy.shape);pos=np.log1p(pp)
 control=np.stack([head,pos],-1).reshape(-1,2);X=np.concatenate([control,np.nan_to_num(xx.reshape(-1,len(FN)),nan=0)],-1);Y=yy.ravel();cut0=32*yy.shape[1]*16;finite=np.isfinite(Y);cut=int(finite[:cut0].sum());X=X[finite];Y=Y[finite]
 # Structural self-V / local-O collinearity: do not learn rounding-noise proxies.
 X[:,14]=0.
 def fit(cols):
  m=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=10,min_samples_leaf=100,learning_rate=.08,early_stopping=False,random_state=9876)
  m.fit(X[:cut,cols],Y[:cut]);pred=m.predict(X[cut:,cols]);return m,mean_squared_error(Y[cut:],pred)
 if not descriptive and cut>=1000 and len(Y)-cut>=1000:
  baseline,base_mse=fit(slice(0,2));full,full_mse=fit(slice(None));groups={'amplitude':list(range(2,6)),'shares':list(range(6,10)),'angles':list(range(10,16)),'coherence':[16],'self_alpha':[17]};perm={}
  for name,cols in groups.items():
   xp=X[cut:].copy()
   for h in range(16):
    where=np.flatnonzero(xp[:,0]==h);order=rng.permutation(where);xp[np.ix_(where,cols)]=xp[np.ix_(order,cols)]
   perm[name]=float((mean_squared_error(Y[cut:],full.predict(xp))-full_mse)/base_mse)
 else:base_mse=full_mse=float("nan");perm={}
 # Actual O update vs content at the same unit write address. No extra ablation.
 ratio=raw[...,ix['write_ratio']];parallel=raw[...,ix['relative_parallel']];cc=raw[...,ix['content_cos']]
 write=[]
 for h in range(16):
  good=mask[:,:,h]&np.isfinite(ratio[:,:,h])&np.isfinite(cc[:,:,h]);v=ratio[:,:,h][good];c=cc[:,:,h][good];a=parallel[:,:,h][good]
  write.append(dict(valid=int(good.sum()),ratio_quantiles=np.quantile(v,quantiles).tolist() if len(v) else None,cos_quantiles=np.quantile(c,quantiles).tolist() if len(v) else None,opposing=float(np.mean(c<0)) if len(v) else None,over_erase=float(np.mean(a < -1)) if len(v) else None,reinforce=float(np.mean(c>0)) if len(v) else None))
 result=dict(layer=layer,n=n,threshold=threshold,eligible_tokens_per_head=coverage.sum(axis=0).tolist(),eligible_sequences_per_head=coverage_sequences.tolist(),eligible_sequences_discovery=(coverage[:32]>0).sum(axis=0).tolist(),eligible_sequences_validation=(coverage[32:]>0).sum(axis=0).tolist(),sampled_train=cut,sampled_validation=len(Y)-cut,seconds=time.perf_counter()-start,baseline_mse=float(base_mse),full_mse=float(full_mse),heldout_incremental_r2=float(1-full_mse/base_mse),group_permutation_importance=perm,write=write,reconstruction_quantiles=np.quantile(raw[...,ix['reconstruction_relative']][allmask],quantiles).tolist(),key_reconstruction_quantiles=np.nanquantile(raw[...,ix['key_content_relative_error']][allmask],quantiles).tolist())
 dest=root/f'analysis_gate{round(threshold*100):03d}';dest.mkdir(exist_ok=True);(dest/f'layer_{layer:02d}.json').write_text(json.dumps(result,indent=2));np.savez_compressed(dest/f'layer_{layer:02d}.npz',quantile_levels=quantiles,feature_quantiles=q,gate_quantiles=gq,hist=hist,curves=curves,sequence_curves=seq_curves,correlations=cor,feature_names=FN,edges=np.stack(edges),sample_features=xx,sample_gate=yy,sample_positions=pp,eligible_counts=coverage)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--n',type=int,default=64);p.add_argument('--workers',type=int,default=8);p.add_argument('--layers',default='');p.add_argument('--threshold',type=float,default=.1);p.add_argument('--descriptive',action='store_true');a=p.parse_args();layers=list(map(int,a.layers.split(','))) if a.layers else list(range(24))
 print('CPU_ENV',os.cpu_count(),'workers',a.workers,flush=True)
 tasks=[(a.root,l,a.n,a.threshold,a.descriptive) for l in layers]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  for r in pool.map(analyze_layer,tasks):print('LAYER_ANALYZED',r['layer'],r['seconds'],r['heldout_incremental_r2'],r['group_permutation_importance'],flush=True)
 print('GEOMETRY_ANALYSIS_DONE',flush=True)
if __name__=='__main__':main()

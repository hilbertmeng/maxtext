"""Seal head groups using first32 observational sequences; never use loss."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,concurrent.futures
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def layer_stats(task):
 root,l=task;root=Path(root);meta=json.loads((root/'metadata.json').read_text());ix={k:i for i,k in enumerate(meta['fields'])}
 a=np.stack([np.load(root/f'sample_{i:03d}.npy',mmap_mode='r')[l] for i in range(32)])
 valid=np.load(root/'valid.npy')[:32];rows=[]
 for h in range(16):
  r=a[:,:,h];norms=np.sqrt(np.maximum(r[...,[ix[f'gram_{j}{j}'] for j in range(4)]],0));s=norms[...,3]/np.maximum(norms.sum(-1),1e-20);g=r[...,ix['read_o_gate']];w=r[...,ix['write_gate']]
  threshold=max(.1,float(np.quantile(g[valid],.75)));take=valid&(g>=threshold)
  rho=float(spearmanr(g[valid],w[valid]).statistic);rs=float(spearmanr(s[valid],w[valid]).statistic)
  pp=r[...,ix['relative_parallel']][take];cc=r[...,ix['content_cos']][take];rr=r[...,ix['write_ratio']][take]
  rows.append(dict(layer=l,head=h,read_write_rho=rho,share_write_rho=rs,threshold=threshold,selected_tokens=int(take.sum()),selected_sequences=int((take.sum(1)>0).sum()),opposing_fraction=float(np.nanmean(cc<0)),ratio_median=float(np.nanmedian(rr)),opposing_strength=float(np.nanmean(np.maximum(-pp,0)))))
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args()
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:rows=sum(list(pool.map(layer_stats,[(a.root,l) for l in range(2,23)])),[])
 eligible=[r for r in rows if r['selected_sequences']>=16 and r['selected_tokens']>=512]
 neg=[r for r in eligible if 3<=r['layer']<=16 and r['read_write_rho']<=-.3 and r['share_write_rho']<=-.3]
 weak=[r for r in eligible if 17<=r['layer']<=22 and abs(r['read_write_rho'])<=.1 and abs(r['share_write_rho'])<=.1]
 pos=[r for r in eligible if 17<=r['layer']<=22 and r['read_write_rho']>=.2 and r['share_write_rho']>=.2]
 anti=sorted([r for r in eligible if r['read_write_rho']<=-.3 and r['opposing_fraction']>=.5],key=lambda r:-r['opposing_strength'])[:6]
 rng=np.random.default_rng(9202201)
 def choose(rs,n):return [rs[i] for i in sorted(rng.choice(len(rs),min(n,len(rs)),replace=False))]
 groups={'negative_middle':choose(neg,12),'weak_late':choose(weak,6),'positive_late':choose(pos,6),'opposing_large':anti}
 selected={}
 for name,rs in groups.items():
  for r in rs:
   key=f"L{r['layer']}H{r['head']}"
   if key not in selected:selected[key]={**r,'id':key,'groups':[]}
   selected[key]['groups'].append(name)
 result=dict(discovery_sequences=[0,32],evaluation_sequences=[32,128],shifts=[-2.,-1.,1.,2.],shift_units='g_prime = g + alpha*h; h=min(0.01,0.1*g,0.1*(1-g)), fixed at baseline; gradient dloss/dalpha at alpha=0',primary_shift=-1.,rule='read_gate >= max(0.1, head-specific discovery P75); no write gate filtering',predictions={'negative_middle':'lowering: majority loss increases','weak_late':'lowering: approximately zero; equivalence margin 0.0001 nats/token','positive_late':'lowering: small damage or improvement; equivalence margin 0.0001','opposing_large':'lowering: some loss improvement expected'},pool_sizes={k:len(v) for k,v in [('negative_middle',neg),('weak_late',weak),('positive_late',pos)]},heads=list(selected.values()),all_head_stats=rows)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2));print('SEALED',result['pool_sizes'],{k:len(v) for k,v in groups.items()},'unique',len(selected),flush=True)

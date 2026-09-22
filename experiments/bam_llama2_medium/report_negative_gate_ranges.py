import concurrent.futures,json,time
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
P=parser.parse_args().root
heads=[h for h in json.loads((P/'selection_waterfall.json').read_text()) if h['read_write_rho']<=-.2]
ls=np.array([h['layer'] for h in heads]);hs=np.array([h['head'] for h in heads]);thresholds=np.array([h['threshold'] for h in heads]);qs=[0,.01,.1,.25,.5,.75,.9,.99,1.]
def one(path):
 with np.load(path) as f:
  g=f['write_gate'];r=f['read_gate'];v=f['valid']
 return g[ls,:,hs],r[ls,:,hs],v
paths=sorted((P/'position_gradient_worker').glob('gradient_*.npz'));start=time.perf_counter()
with concurrent.futures.ProcessPoolExecutor(8) as pool:records=list(pool.map(one,paths))
g=np.stack([x[0] for x in records],axis=1).reshape(len(heads),-1);r=np.stack([x[1] for x in records],axis=1).reshape(len(heads),-1);valid=np.stack([x[2] for x in records]).reshape(-1)
rows=[]
for j,h in enumerate(heads):
 masks={'all_valid':valid,'read_high':valid&(r[j]>=thresholds[j])};scopes={}
 for scope,mask in masks.items():
  x=g[j,mask];scopes[scope]=dict(n=len(x),quantiles=np.quantile(x,qs).tolist() if len(x) else None,fractions_below={str(c):float((x<=c).mean()) for c in [.02,.03,.05,.1,.2,.5]} if len(x) else None)
 rows.append(dict(id=h['id'],rho=h['read_write_rho'],discovery_no_write_le002=h['write_small']==0,read_threshold=h['threshold'],scopes=scopes))
summary=dict(sequences=[32,128],dtype='FP32',quantile_levels=qs,heads=rows)
sub=np.array([h['write_small']==0 for h in heads]);summary['discovery_88_head_group']={}
for scope in ['all_valid','read_high']:
 x=g[sub];mask=np.broadcast_to(valid,x.shape).copy()
 if scope=='read_high':mask&=r[sub]>=thresholds[sub,None]
 y=x[mask];q=np.quantile(y,qs).tolist();summary['discovery_88_head_group'][scope]=dict(n=len(y),quantiles=q,fractions_below={str(c):float((y<=c).mean()) for c in [.02,.03,.05,.1,.2,.5]});print(scope,summary['discovery_88_head_group'][scope])
selected=[x for x in rows if x['discovery_no_write_le002']]
print('88 group eval heads still no <=.02',sum(x['scopes']['all_valid']['quantiles'][0]>.02 for x in selected))
for scope in ['all_valid','read_high']:
 q=np.array([x['scopes'][scope]['quantiles'] for x in selected if x['scopes'][scope]['n']]);print('HEAD_QUANTILES',scope,'min median max of head medians',np.quantile(q[:,4],[0,.1,.25,.5,.75,.9,1]).tolist());print('head medians bins',np.histogram(q[:,4],bins=[0,.02,.05,.1,.2,.5,1])[0].tolist())
 print('examples',[(x['id'],[round(v,4) for v in x['scopes'][scope]['quantiles']]) for x in sorted(selected,key=lambda x:(x['scopes'][scope]['quantiles'] or [0]*9)[4])[::12] if x['scopes'][scope]['n']])
(P/'negative_head_gate_ranges.json').write_text(json.dumps(summary,indent=2));print('elapsed',time.perf_counter()-start)

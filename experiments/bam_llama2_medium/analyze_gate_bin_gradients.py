import concurrent.futures,json
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path)
P=parser.parse_args().root
heads=json.loads((P/'negative_fullrange_protocol.json').read_text())['heads'];edges=np.array([0,.02,.05,.1,.2,.4,.6,1.000001]);ls=np.array([h['layer'] for h in heads]);hs=np.array([h['head'] for h in heads]);threshold=np.array([h['threshold'] for h in heads])
def one(path):
 with np.load(path) as f:g=f['write_gate'][ls,:,hs];r=f['read_gate'][ls,:,hs];v=f['gradient'][ls,:,hs];valid=f['valid']
 selected=(r>=threshold[:,None])&valid;step=np.minimum(.01,np.minimum(.1*g,.1*(1-g)));out=np.zeros((len(heads),len(edges)-1,5))
 for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
  m=selected&(g>=lo)&(g<hi)
  out[:,k,0]=m.sum(1);out[:,k,1]=(m&(v>0)).sum(1);out[:,k,2]=np.where(m,v,0).sum(1,dtype=np.float64);out[:,k,3]=np.where(m,v*step,0).sum(1,dtype=np.float64);out[:,k,4]=np.where(m,g,0).sum(1,dtype=np.float64)
 return out
paths=sorted((P/'position_gradient_worker').glob('gradient_*.npz'))
with concurrent.futures.ProcessPoolExecutor(8) as pool:a=np.stack(list(pool.map(one,paths)))
np.savez_compressed(P/'fullrange_binned_gradients.npz',values=a,edges=edges)
w=np.random.default_rng(9202220).multinomial(len(a),np.full(len(a),1/len(a)),size=10000)/len(a)
pred=-a[:,:,:,3];boot=(w@pred.reshape(len(a),-1)).reshape(len(w),len(heads),-1);ci=np.quantile(boot,[.025,.975],axis=0)
rows=[]
for k in range(len(edges)-1):
 total=a[:,:,k,0].sum();gb=boot[:,:,k].mean(1);row=dict(bin=[float(edges[k]),float(min(edges[k+1],1))],positions=int(total),positive_gradient_fraction=float(a[:,:,k,1].sum()/total),head_averaged_predicted_down_delta=float(pred[:,:,k].mean()),ci95=np.quantile(gb,[.025,.975]).tolist());rows.append(row);print(row)
result=dict(heads=[dict(id=h['id'],predicted_down=pred[:,j].mean(0).tolist(),ci95=ci[:,j].T.tolist(),counts=a[:,j,:,0].sum(0).astype(int).tolist()) for j,h in enumerate(heads)],bin_summary=rows,scope='70 strong negative heads with sufficient read-high coverage; no write upper cutoff; gradients only, not finite ablations')
(P/'fullrange_binned_gradients.json').write_text(json.dumps(result,indent=2))

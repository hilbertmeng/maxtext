#!/usr/bin/env python3
"""CPU-only scale calibration, without changing any training RUN's initialization."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D,N=1024,16;width=3*N;base_std=.006

def gelu(z):return .5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
# Numerical moments for the same tanh-approximate GELU used by the model.
nodes,weights=np.polynomial.hermite.hermgauss(96);h=gelu(np.sqrt(2)*nodes)
mean=float(weights@h/np.sqrt(np.pi));second=float(weights@(h*h)/np.sqrt(np.pi));variance=second-mean**2
up_std=np.sqrt(D*base_std**2/(width*variance))
rows=[]
for seed in range(5):
 rng=np.random.default_rng(seed+420);x=rng.normal(size=(4096,D)).astype(np.float32)
 down=rng.normal(size=(D,width)).astype(np.float32);up=rng.normal(size=(width,N)).astype(np.float32)
 linear=x@(rng.normal(size=(D,N)).astype(np.float32)*base_std)
 legacy=gelu(x@(down*base_std))@(up*base_std)
 calibrated=gelu(x@(down/np.sqrt(D)))@(up*up_std)
 out={'seed':seed+420}
 for name,z in [('linear',linear),('legacy',legacy),('calibrated',calibrated)]:
  out[name]={'token_std_rms':float(np.sqrt(np.mean(np.var(z,axis=0)))),
             'token_mean_rms':float(np.sqrt(np.mean(np.mean(z,axis=0)**2))),
             'logit_rms':float(np.sqrt(np.mean(z*z)))}
 rows.append(out)
result={'fixture':'Independent Gaussian unit-RMS tokens; D1024 N16, shared width48. CPU scale check, not a training result.',
        'gelu_mean':mean,'gelu_second_moment':second,'gelu_variance':variance,
        'calibrated_down_std':1/np.sqrt(D),'calibrated_up_std':float(up_std),'rows':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n')
for row in rows:print(row['seed'],{name:round(row[name]['token_std_rms'],6) for name in ['linear','legacy','calibrated']})

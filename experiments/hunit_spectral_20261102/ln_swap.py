"""Frozen-data2x2 LN swaps separate raw-residual and learned-affine changes."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent;m=json.loads((R/'arms_primary.json').read_text());C=Path(m['capture_DATA']);D=Path(m['local_DATA']);steps=[33000,43000];params=[];bases=[]
for t in steps:
 with np.load(D/'head_spectra'/f'{t}.npz') as z:params.append((z['gamma'].astype(np.float64),z['bias'].astype(np.float64)));bases.append(z['centered_basis'][:,:128])
def bf16(x):
 b=np.asarray(x,np.float32).view(np.uint32);return ((b+np.uint32(0x7fff)+((b>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
power=np.zeros((2,2,2,32,2),np.float64);errors=np.zeros((2,32),np.float64);actual=np.zeros((2,2,32,2),np.float64)
for ri,t in enumerate(steps):
 for i in range(32):
  f=C/'results'/str(t)/f'sequence_{i:03d}.npz'
  with np.load(f) as z:raw=z['raw'][0].astype(np.float64);h=z['h'][0].copy()
  center=raw-raw.mean(-1,keepdims=True);n=center/np.sqrt(np.mean(center*center,-1,keepdims=True)+1e-5)
  for li,(g,b) in enumerate(params):
   swapped=bf16(n*g+b)
   if ri==li:errors[ri,i]=np.linalg.norm(swapped.astype(np.float64)-h)/np.linalg.norm(h.astype(np.float64))
   for bi,u in enumerate(bases):
    pc=swapped@u;power[bi,ri,li,i]=[np.mean(swapped.astype(np.float64)**2),np.mean(np.sum(pc.astype(np.float64)**2,-1)/2560)]
  for bi,u in enumerate(bases):
   pc=h@u;actual[bi,ri,i]=[np.mean(h.astype(np.float64)**2),np.mean(np.sum(pc.astype(np.float64)**2,-1)/2560)]
 print('NATIVE_LN_SWAP_RAW_CHECKPOINT_COMPLETE',t,flush=True)
assert errors.max()<.015,('LN functional reconstruction failed; no factor attribution accepted',float(errors.max()))
rng=np.random.default_rng(20261102);boot=rng.integers(0,32,(5000,32));out=dict(scope='32x2048 native captured raw vectors, saved final-LN parameters; two-factor symmetric Shapley of mean-square power, frozen-data intervention, not retraining attribution. Each basis fixed to one checkpoint; BF16 output RNE. LN includes scale and bias jointly.',source_commit=m['capture_commit'],checkpoint_paths=m['checkpoints'],own_LN_max_relative_RMS=float(errors.max()),basis={})
for bi,t in enumerate(steps):
 cells=power[bi];p=cells.mean(2);b=cells[:,:,boot,:].mean(3)
 row=dict(cell_mean_power=p.tolist(),actual_native_endpoint_power=actual[bi].mean(1).tolist())
 for mi,name in enumerate(['total_power','weak5_power']):
  raw=.5*((p[1,0,mi]-p[0,0,mi])+(p[1,1,mi]-p[0,1,mi]));ln=.5*((p[0,1,mi]-p[0,0,mi])+(p[1,1,mi]-p[1,0,mi]));delta=p[1,1,mi]-p[0,0,mi];np.testing.assert_allclose(raw+ln,delta,rtol=1e-12,atol=1e-12)
  rb=.5*((b[1,0,:,mi]-b[0,0,:,mi])+(b[1,1,:,mi]-b[0,1,:,mi]));lb=.5*((b[0,1,:,mi]-b[0,0,:,mi])+(b[1,1,:,mi]-b[1,0,:,mi]));db=b[1,1,:,mi]-b[0,0,:,mi]
  row[name]=dict(growth=float(delta),raw_residual_share=float(raw/delta),LN_parameter_share=float(ln/delta),LN_parameter_share_ci95=np.quantile(lb/db,[.025,.975]).tolist(),raw_residual_share_ci95=np.quantile(rb/db,[.025,.975]).tolist())
 out['basis'][str(t)]=row
(R/'ln_swap.json').write_text(json.dumps(out,indent=2)+'\n');print('NATIVE_LN_SWAP_COMPLETE',json.dumps({t:v['weak5_power'] for t,v in out['basis'].items()}),flush=True)

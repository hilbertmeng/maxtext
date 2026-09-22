"""Four-part write composition from immutable TPU Gram captures; CPU only.

Discovery 0:32 selects heads; evaluation 32:128 is never used for selection.
Energy shares of components exclude interference; signed projection attribution
and cross energy are saved separately. No interpretation as retraining benefit.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np

SCOPES = ['all', 'write_ge005', 'write_ge010', 'write_ge020']
NAMES = ['count'] + [f'{kind}_{i}' for kind in ('norm', 'ungated_norm', 'energy', 'ungated_energy', 'signed_energy') for i in range(4)] + ['total_energy', 'nonlocal_norm', 'local_norm', 'nonlocal_energy', 'local_energy', 'cross_energy', 'ungated_nonlocal_norm', 'ungated_local_norm', 'ungated_nonlocal_energy', 'ungated_local_energy', 'matrix_nonlocal_norm', 'matrix_local_norm', 'matrix_nonlocal_energy', 'matrix_local_energy']
GRAM = [(i,j) for i in range(4) for j in range(i,4)]

def one(args):
    root, seq, heads = args
    raw = np.load(Path(root)/f'sample_{seq:03d}.npy', mmap_mode='r')
    valid = np.load(Path(root)/'valid.npy', mmap_mode='r')[seq]
    x = raw[[h['layer'] for h in heads], :, [h['head'] for h in heads]].astype(np.float64)
    gram = np.zeros(x.shape[:2]+(4,4))
    for k,(i,j) in enumerate(GRAM):
        gram[...,i,j] = gram[...,j,i] = x[...,k]
    diag = np.maximum(np.diagonal(gram,axis1=-2,axis2=-1),0)
    norms = np.sqrt(diag)
    gate, read = x[...,10], x[...,26]
    # sum_norm is the actual pre-normalization 48-dimensional write data.
    denom = np.sqrt(x[...,22]**2/48 + 1e-6)
    scale = gate/denom
    base = 1/denom
    energy = diag*scale[...,None]**2
    norm = norms*scale[...,None]
    nonlocal2 = np.maximum(gram[...,:3,:3].sum(axis=(-1,-2)),0)
    local2 = diag[...,3]
    cross = 2*gram[...,:3,3].sum(axis=-1)
    signed = gram.sum(axis=-1)*scale[...,None]**2
    pn = x[...,18]
    vals = np.concatenate([np.ones(gate.shape+(1,)),norm,norms*base[...,None],energy,diag*base[...,None]**2,signed,
        np.stack([(nonlocal2+local2+cross)*scale**2,np.sqrt(nonlocal2)*scale,np.sqrt(local2)*scale,nonlocal2*scale**2,local2*scale**2,cross*scale**2,np.sqrt(nonlocal2)*base,np.sqrt(local2)*base,nonlocal2*base**2,local2*base**2,np.sqrt(nonlocal2)*scale*pn,np.sqrt(local2)*scale*pn,nonlocal2*scale**2*pn**2,local2*scale**2*pn**2],axis=-1)],axis=-1)
    masks = [np.broadcast_to(valid,gate.shape)] + [valid[None,:]&(gate>=t) for t in (.05,.1,.2)]
    sums = np.stack([np.where(m[...,None],vals,0).sum(axis=1) for m in masks],axis=1)
    quadrants=[]
    for t in (.05,.1,.2):
        quadrants.append(np.stack([((read>=t)&(gate<t)&valid).sum(1),((read<t)&(gate>=t)&valid).sum(1),((read<t)&(gate<t)&valid).sum(1),((read>=t)&(gate>=t)&valid).sum(1)],axis=-1))
    hist = np.stack([np.histogram2d(read[j,valid],gate[j,valid],bins=50,range=((0,1),(0,1)))[0] for j in range(len(heads))])
    shares = norms / np.maximum(norms.sum(-1,keepdims=True),1e-30)
    share_hist = np.stack([np.stack([np.stack([np.histogram(shares[j,m[j],k],bins=100,range=(0,1))[0] for k in range(4)]) for m in masks]) for j in range(len(heads))])
    return sums,np.stack(quadrants,axis=1),hist,share_hist,float(np.max(x[...,23][np.broadcast_to(valid,gate.shape)]))

def fractions(s):
    get=lambda key:s[...,NAMES.index(key)]
    groups={}
    for kind in ('norm','ungated_norm','energy','ungated_energy','signed_energy'):
        a=np.stack([get(f'{kind}_{i}') for i in range(4)],axis=-1)
        den=get('total_energy') if kind=='signed_energy' else a.sum(-1)
        groups[kind+'_shares']=(a/np.maximum(den[...,None],1e-30)).tolist()
    for kind in ('norm','energy'):
        a,b=get('nonlocal_'+kind),get('local_'+kind)
        c,d=get('ungated_nonlocal_'+kind),get('ungated_local_'+kind)
        groups['nonlocal_'+kind+'_share']=(a/np.maximum(a+b,1e-30)).tolist()
        groups['ungated_nonlocal_'+kind+'_share']=(c/np.maximum(c+d,1e-30)).tolist()
        groups['local_'+kind+'_suppression_pp']=(100*(d/np.maximum(c+d,1e-30)-b/np.maximum(a+b,1e-30))).tolist()
        a,b=get('matrix_nonlocal_'+kind),get('matrix_local_'+kind)
        groups['matrix_nonlocal_'+kind+'_share']=(a/np.maximum(a+b,1e-30)).tolist()
    groups['cross_energy_over_total']=(get('cross_energy')/np.maximum(get('total_energy'),1e-30)).tolist()
    return groups

def main():
    ap=argparse.ArgumentParser();ap.add_argument('geometry',type=Path);ap.add_argument('prior',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    heads=[h for h in json.loads((a.prior/'protocol.json').read_text())['all_head_stats'] if 3<=h['layer']<=16 and h['read_write_rho']<=-.3]
    args=[(str(a.geometry),i,heads) for i in range(32,128)]
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(one,args))
    sums=np.stack([r[0] for r in rows]);quad=np.stack([r[1] for r in rows]);hist=sum(r[2] for r in rows);share_hist=sum(r[3] for r in rows)
    # Reproducibility check on one independently repeated serial sample.
    check=one(args[0]);np.testing.assert_array_equal(check[0],rows[0][0]);np.testing.assert_array_equal(check[1],rows[0][1])
    np.savez_compressed(a.out/'statistics.npz',sums=sums,quadrants=quad,hist=hist,share_hist=share_hist,names=NAMES,scopes=SCOPES)
    total=sums.sum((0,1));perhead=sums.sum(0)
    report=dict(heads=heads,selection='discovery 0:32; L3-L16; read/write Spearman <= -0.3; no read-coverage exclusion',evaluation=[32,128],thresholds=[.05,.1,.2],scopes=SCOPES,components=['sv','sv_bam_self','sv_bam_other','localO'],pooled=fractions(total),per_head=fractions(perhead),quadrant_order=['read_high_write_low','read_low_write_high','both_low','both_high'],quadrants=(quad.sum((0,1))/quad.sum((0,1)).sum(-1,keepdims=True)).tolist(),per_head_quadrants=(quad.sum(0)/quad.sum(0).sum(-1,keepdims=True)).tolist(),counts=total[:,0].tolist(),max_reconstruction_relative=max(r[4] for r in rows),seconds=time.perf_counter()-start,workers=a.workers)
    # Sequence-cluster bootstrap of pooled cumulative ratios, not token resampling.
    rng=np.random.default_rng(9876);byseq=sums.sum(1);boot=byseq[rng.integers(0,96,(4000,96))].sum(1);bf=fractions(boot)
    report['bootstrap_95']={k:np.quantile(v,[.025,.975],axis=0).tolist() for k,v in bf.items()}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('selection','counts','pooled','quadrants','seconds','max_reconstruction_relative')},indent=2))

if __name__=='__main__':main()

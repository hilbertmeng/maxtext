"""Shared-input-basis design space from immutable checkpoint projections."""
import argparse
import json
from pathlib import Path
import numpy as np
from bam_cross_layer_geometry import GROUPS, eig_basis, spectrum

PARTITIONS = {
    'blocks': [list(range(i,i+3)) for i in range(0,18,3)],
    'thirds': [list(range(i,i+6)) for i in range(0,18,6)],
    'halves': [list(range(9)),list(range(9,18))],
    'global': [list(range(18))],
    # Same basis count as thirds, but share by layer role instead of depth.
    'roles': [list(range(i,18,3)) for i in range(3)],
}

def main():
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument('geometry_dir');args=ap.parse_args();out=Path(args.geometry_dir)
  arrays=np.load(out/'weights.npz')
  groups={g:next(arrays[n] for n in arrays.files if '/'+g+'/' in n) for g in GROUPS}
  weights=[groups[g][:,b].astype(np.float64) for b in range(6) for g in GROUPS]
  d,r=weights[0].shape;norms=[np.linalg.norm(w) for w in weights]
  q=[np.linalg.svd(w,full_matrices=False)[0] for w in weights]
  pairs=[]
  for i in range(18):
    for j in range(i+1,18):
      pairs.append(dict(layers=[i,j],same_block=i//3==j//3,distance=j-i,
          full_subspace_overlap=float(np.sum((q[i].T@q[j])**2)/r),
          top64_overlap=float(np.sum((q[i][:,:64].T@q[j][:,:64])**2)/64)))
  report=dict(shape=[d,r],pairs=pairs,random_full_overlap=r/d,
              random_top64_overlap=64/d,partitions={},variants={})
  bases={}
  for name,partition in PARTITIONS.items():
    assert sorted(sum(partition,[]))==list(range(18))
    stats=[]
    for gi,layers in enumerate(partition):
      entry=dict(layers=layers,joint={})
      for mode in ('raw','equal_layer_energy'):
        joined=np.concatenate([weights[i] if mode=='raw' else weights[i]/norms[i] for i in layers],axis=1)
        e,u=eig_basis(joined);entry['joint'][mode]=dict(spectrum=spectrum(e),ranks={})
        for rank in (256,384,512,768):
          a=u[:,:rank]
          entry['joint'][mode]['ranks'][str(rank)]=dict(
              energy_retained=float(e[:rank].sum()/e.sum()),
              per_layer_energy_retained=[float(np.sum((a.T@weights[i])**2)/norms[i]**2) for i in layers])
          if mode=='raw':bases[f'{name}_group{gi}_rank{rank}']=a.astype(np.float32)
      stats.append(entry)
    report['partitions'][name]=stats
    # Two common ranks and one parameter-budget matched rank: no automatic retraining claim.
    ranks={384,512}
    target_cost=384*(6*d+18*r)  # blockwise R384 basis+coeff budget
    budget_rank=min(1200,int(target_cost//(len(partition)*d+18*r)))
    ranks.add(budget_rank)
    for rank in sorted(ranks):
      for gi,layers in enumerate(partition):
        key=f'{name}_group{gi}_rank{rank}'
        if key not in bases:
          _,u=eig_basis(np.concatenate([weights[i] for i in layers],axis=1))
          bases[key]=u[:,:rank].astype(np.float32)
      retained=[0.]*18
      for gi,layers in enumerate(partition):
        a=bases[f'{name}_group{gi}_rank{rank}'].astype(np.float64)
        for i in layers:retained[i]=float(np.sum((a.T@weights[i])**2)/norms[i]**2)
      report['variants'][f'{name}_r{rank}']=dict(partition=partition,rank=rank,
          basis_keys=[f'{name}_group{gi}_rank{rank}' for gi in range(len(partition))],
          per_layer_energy_retained=retained,
          parameter_savings=18*d*r-rank*(len(partition)*d+18*r),
          budget_matched_to_blocks_r384=(rank==budget_rank))
    print('PARTITION_DONE',name,flush=True)
  np.savez_compressed(out/'global-bases.npz',**bases)
  (out/'global-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
  print('GLOBAL_GEOMETRY_DONE',args.geometry_dir,flush=True)
if __name__=='__main__':main()

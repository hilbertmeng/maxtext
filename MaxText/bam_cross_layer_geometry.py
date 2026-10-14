"""Read-only scan-aware geometry of three shared down projections per block.

CPU partial Orbax restore. A common input basis A and per-layer coefficients C
approximate W_l=A C_l BEFORE GELU; this is not a claim that GELU units can be
rotated or tied without changing the function.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

GROUPS = ('local_0', 'local_1', 'fetch_2')
RANKS = (128, 256, 384, 512, 768)

def spectrum(e):
  e = np.maximum(e, 0)
  p = e / max(float(e.sum()), 1e-30)
  nz = p[p > 0]
  return dict(stable_rank=float(1 / max(p[0], 1e-30)),
              effective_rank=float(np.exp(-np.sum(nz * np.log(nz)))),
              **{f'rank_{int(f*100)}': int(np.searchsorted(np.cumsum(p), f)+1)
                 for f in (.90, .95, .99)})

def eig_basis(w):
  e, u = np.linalg.eigh(w @ w.T)
  return np.maximum(e[::-1], 0), u[:, ::-1]

def analyze_block(block, matrices):
  start = time.monotonic()
  d, r = matrices[0].shape
  assert all(w.shape == (d, r) and np.isfinite(w).all() for w in matrices)
  norms = [float(np.linalg.norm(w)) for w in matrices]
  q = [np.linalg.svd(w, full_matrices=False)[0] for w in matrices]
  result = dict(block=block, layers=[3*block+i for i in range(3)], shape=[d,r],
                norms=norms, pairs=[], joint={},
                basis_parameter_break_even_rank=3*d*r/(d+3*r))
  for i,j in ((0,1),(1,2),(0,2)):
    cosine = np.linalg.svd(q[i].T @ q[j], compute_uv=False)
    result['pairs'].append(dict(layers=[3*block+i,3*block+j],
        raw_weight_cosine=float(np.vdot(matrices[i], matrices[j])/(norms[i]*norms[j])),
        mean_squared_principal_cosine=float(np.mean(cosine**2)),
        random_subspace_expectation=r/d,
        principal_cosine_quantiles=np.quantile(cosine,[0,.25,.5,.75,1]).tolist(),
        top_subspace_overlap={str(k):float(np.sum((q[i][:,:k].T@q[j][:,:k])**2)/k)
                              for k in (32,64,128) if k <= r}))
  bases = {}
  for mode in ('raw', 'equal_layer_energy'):
    joined = np.concatenate([w if mode=='raw' else w/n for w,n in zip(matrices,norms)],axis=1)
    e, u = eig_basis(joined)
    result['joint'][mode] = dict(spectrum=spectrum(e), ranks={})
    for k in RANKS:
      if k > min(d,3*r): continue
      a = u[:,:k]
      result['joint'][mode]['ranks'][str(k)] = dict(
          joint_energy_retained=float(e[:k].sum()/e.sum()),
          per_layer_energy_retained=[float(np.sum((a.T@w)**2)/(n*n)) for w,n in zip(matrices,norms)],
          original_parameters=3*d*r, basis_and_coefficients_parameters=k*(d+3*r),
          parameter_savings=3*d*r-k*(d+3*r))
      if mode=='raw' and k in (256,384): bases[f'block{block}_rank{k}'] = a.astype(np.float32)
  result['seconds'] = time.monotonic()-start
  return result, bases

def restore(checkpoint):
  import jax
  import orbax.checkpoint as ocp
  from flax.traverse_util import flatten_dict, unflatten_dict
  cp = ocp.PyTreeCheckpointer()
  tree = cp.metadata(checkpoint)
  tree = getattr(tree, 'item_metadata', tree)
  tree = getattr(tree, 'tree', tree)
  selected = {p:v for p,v in flatten_dict(tree).items()
              if p[:2]==('params','params') and p[-2:]==('full_m_read_down','kernel')}
  if len(selected)!=3: raise ValueError(list(selected))
  sharding = jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
  items = {p:jax.ShapeDtypeStruct(v.shape,v.dtype) for p,v in selected.items()}
  args = {p:ocp.ArrayRestoreArgs(sharding=sharding,global_shape=v.shape,dtype=v.dtype) for p,v in selected.items()}
  restored = cp.restore(checkpoint,args=ocp.args.PyTreeRestore(item=unflatten_dict(items),
                    restore_args=unflatten_dict(args),partial_restore=True))
  return {'/'.join(p):np.asarray(v,dtype=np.float32) for p,v in flatten_dict(restored).items()}

def main():
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument('checkpoint');ap.add_argument('output');ap.add_argument('--workers',type=int,default=2)
  args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
  arrays=restore(args.checkpoint)
  np.savez_compressed(out/'weights.npz',**arrays)
  bygroup={g:next(a for n,a in arrays.items() if '/'+g+'/' in n) for g in GROUPS}
  assert all(a.shape[:2]==(1200,6) for a in bygroup.values())
  blocks=[[bygroup[g][:,b].astype(np.float64) for g in GROUPS] for b in range(6)]
  t=time.monotonic()
  reference,_=analyze_block(0,blocks[0])
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    results=list(pool.map(lambda item:analyze_block(*item),enumerate(blocks)))
  # Parallel execution must preserve numerical results.
  for mode in reference['joint']:
    for k,v in reference['joint'][mode]['ranks'].items():
      np.testing.assert_allclose(v['joint_energy_retained'],results[0][0]['joint'][mode]['ranks'][k]['joint_energy_retained'],rtol=1e-12)
  report=dict(checkpoint=args.checkpoint,scope='within each consecutive three-layer block',
              arrays={n:list(a.shape) for n,a in arrays.items()},blocks=[x[0] for x in results],
              elapsed_seconds=time.monotonic()-t,workers=args.workers,
              interpretation='Common input basis plus per-layer pre-GELU coefficients; not direct nonlinear feature tying.')
  np.savez_compressed(out/'bases.npz',**{k:v for _,b in results for k,v in b.items()})
  (out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
  print('GEOMETRY_DONE',args.output,report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()

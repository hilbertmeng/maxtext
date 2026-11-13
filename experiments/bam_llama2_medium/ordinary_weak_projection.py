"""Healthy ordinary-residual energy bases and frozen-reference vector surgery.

Use uncentered calibration/held-out second moments over representative tokens.
Low energy is descriptive; held-out validation and causal controls are separate.
"""
import numpy as np


def calibrate(calibration,held_out,thresholds=(.99,.95)):
  c=np.asarray(calibration,np.float64);v=np.asarray(held_out,np.float64)
  assert c.shape==v.shape and c.ndim>=2 and c.shape[-1]==c.shape[-2]
  assert np.isfinite(c).all() and np.isfinite(v).all()
  dim=c.shape[-1];prefix=c.shape[:-2];bases=[[] for _ in thresholds];layers=[]
  for idx,(a,b) in enumerate(zip(c.reshape((-1,dim,dim)),v.reshape((-1,dim,dim)))):
    assert np.linalg.norm(a-a.T)<=1e-6*max(np.linalg.norm(a),1e-30)
    w,q=np.linalg.eigh((a+a.T)/2);assert w[0]>=-1e-5*max(float(w[-1]),1e-30)
    w=np.maximum(w,0)[::-1];q=q[:,::-1];total=float(w.sum());validation_total=float(np.trace(b));assert validation_total>=0
    row=dict(index=idx,calibration_energy=total,held_out_energy=validation_total,eigenvalues_descending=w.tolist(),thresholds=[])
    for ti,t in enumerate(thresholds):
      assert 0<t<=1
      rank=min(dim,int(np.searchsorted(np.cumsum(w),t*total)+1)) if total>0 else 0
      p=np.eye(dim)-q[:,:rank]@q[:,:rank].T
      bases[ti].append(p.astype(np.float32))
      row['thresholds'].append(dict(retained_energy=t,rank=rank,weak_calibration_fraction=float(w[rank:].sum()/total) if total>0 else None,weak_held_out_fraction=float(np.trace(p@b)/validation_total) if validation_total>0 else None))
    layers.append(row)
  return np.asarray(bases,np.float32).reshape((len(thresholds),*prefix,dim,dim)),dict(thresholds=list(thresholds),layers=layers,scope='Uncentered healthy ordinary residual second moments; weak complement includes null space. Held-out tokens do not imply independent-batch validation.')


def patch_write(peak,healthy,projector,mode):
  import jax
  import jax.numpy as jnp
  p=peak.astype(jnp.float32);old=healthy.astype(jnp.float32);basis=projector.astype(jnp.float32)
  mm=lambda x,y:jnp.matmul(x,y,precision=jax.lax.Precision.HIGHEST)
  weak=mm(p,basis);strong=p-weak
  permutation=jnp.asarray(np.random.default_rng(9876).permutation(p.shape[-1]));random_basis=jnp.take(jnp.take(basis,permutation,axis=-2),permutation,axis=-1);random=mm(p,random_basis)
  norm=lambda x:jnp.linalg.norm(x,axis=-1,keepdims=True)
  matched=lambda x:x*norm(weak)/jnp.maximum(norm(x),1e-30)
  return jax.lax.switch(mode,(lambda:peak,lambda:(p-weak).astype(peak.dtype),lambda:(p+mm(old-p,basis)).astype(peak.dtype),lambda:(p-matched(strong)).astype(peak.dtype),lambda:(p-matched(random)).astype(peak.dtype),lambda:healthy.astype(peak.dtype)))

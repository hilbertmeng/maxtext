"""Paired host analysis of fidelity-gated pattern/write captures (no TPU use)."""
import argparse
import json
from pathlib import Path
import numpy as np


def energy(x):
  return float(np.sum(np.asarray(x,dtype=np.float64)**2))


def relative_error(actual,reference):
  return np.sqrt(energy(np.asarray(actual)-reference)/max(energy(reference),1e-30))


def component_projection(component,total):
  """Signed projection, not a causal share; values need not lie in [0,1]."""
  denom=energy(total)
  return float(np.sum(np.asarray(component,dtype=np.float64)*total)/denom) if denom>1e-30 else None


def route_difference(old,new):
  """Last axis is keys; zero/empty causal rows are excluded explicitly."""
  old=np.asarray(old,dtype=np.float64);new=np.asarray(new,dtype=np.float64)
  valid=(old.sum(-1)>.5)&(new.sum(-1)>.5)
  mid=(old+new)/2
  kl=lambda p:np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)/np.maximum(mid,1e-300)),0),-1)
  return dict(valid=valid,tv=.5*np.sum(abs(new-old),-1),js=.5*(kl(old)+kl(new)),
              top_key_changed=np.argmax(old,-1)!=np.argmax(new,-1),
              entropy_old=-np.sum(old*np.log(np.maximum(old,1e-300)),-1),
              entropy_new=-np.sum(new*np.log(np.maximum(new,1e-300)),-1),
              first_mass_old=old[...,0],first_mass_new=new[...,0],
              prefix8_mass_old=old[...,:8].sum(-1),prefix8_mass_new=new[...,:8].sum(-1))


def weak_write(matrix,delta,retained_energy=.99):
  """Both axes: write energy outside old matrix's 99%-energy singular subspace.

  This includes the old null space. It is a descriptive projection of sampled
  matrices, not evidence that a weak direction is harmful or causally active.
  """
  matrix=np.asarray(matrix,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
  u,s,vh=np.linalg.svd(matrix,full_matrices=False)
  cumulative=np.cumsum(s*s);total=float(cumulative[-1])
  rank=int(np.searchsorted(cumulative,retained_energy*total)+1) if total>0 else 0
  denom=energy(delta)
  content=delta@vh[:rank].T;address=u[:,:rank].T@delta
  return dict(rank99=rank,old_energy=total,write_energy=energy(delta),
              matrix_stable_rank=total/max(float(s[0]**2),1e-30),
              matrix_top4_energy_fraction=float(np.sum(s[:4]**2)/max(total,1e-30)),
              weak_content_fraction=float(np.clip(1-energy(content)/denom,0,1)) if denom>1e-30 else None,
              weak_address_fraction=float(np.clip(1-energy(address)/denom,0,1)) if denom>1e-30 else None)


def describe(values,mask=None):
  values=np.asarray(values,dtype=np.float64)
  if mask is not None:values=values[np.asarray(mask)]
  values=values[np.isfinite(values)]
  if not values.size:return None
  return dict(mean=float(values.mean()),median=float(np.median(values)),
              p95=float(np.percentile(values,95)),max=float(values.max()))


def cosine(a,b):
  dot=np.sum(a.astype(np.float64)*b,-1)
  denom=np.sqrt(np.sum(a.astype(np.float64)**2,-1)*np.sum(b.astype(np.float64)**2,-1))
  return np.divide(dot,denom,out=np.full_like(dot,np.nan),where=denom>1e-20)


def analyze(root):
  metadata=json.loads((root/'pattern-write.json').read_text())
  cases={c['name']:c for c in metadata['cases']}
  assert all(cases[n]['fidelity_passed'] for n in ('old_control','peak_control'))
  old=dict(np.load(root/'old_control-pattern-write.npz',allow_pickle=False))
  peak=dict(np.load(root/'peak_control-pattern-write.npz',allow_pickle=False))
  for values in (old,peak):
    assert all(np.isfinite(v).all() for v in values.values()),'Nonfinite capture'
  result=dict(scope=metadata['pattern_scope'],cautions=[
    'FP32 reconstructed probabilities, not exact Splash weights.',
    'Signed component projections are algebraic descriptions, not additive causal shares.',
    'Weak-subspace energy is sampled M geometry, not proof of an instability.',
    'Peak-batch matched states do not establish chronological onset.',
    'Fullbatch JIT addition-rounding metrics disagree with materialized snapshots; excluded. Use sampled host metrics.'],layers=[],rollback_cases=[])
  diff=route_difference(old['route'],peak['route'])
  for layer in range(6):
    entry=dict(layer=layer,route_by_head=[],context={},write={},memory=[])
    for head in range(old['route'].shape[2]):
      valid=diff['valid'][layer,:,head]
      entry['route_by_head'].append(dict(head=head,**{
        k:describe(v[layer,:,head],valid) for k,v in diff.items() if k!='valid'}))
    old_y=old['context_reconstructed'][layer];new_y=peak['context_reconstructed'][layer]
    fixed=peak['context_fixed_oldP'][layer];delta=new_y-old_y
    entry['context']=dict(old_native_reconstruction_error=relative_error(old_y,old['context_native'][layer]),
      peak_native_reconstruction_error=relative_error(new_y,peak['context_native'][layer]),
      delta_rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
      route_delta_rms=float(np.sqrt(np.mean((new_y-fixed).astype(np.float64)**2))),
      content_delta_rms=float(np.sqrt(np.mean((fixed-old_y).astype(np.float64)**2))),
      route_signed_projection=component_projection(new_y-fixed,delta),
      content_signed_projection=component_projection(fixed-old_y,delta),
      decomposition_error=relative_error((new_y-fixed)+(fixed-old_y),delta))
    before=old['projection_oldW'][layer];after=peak['projection_peakW'][layer]
    fixed_w=peak['projection_oldW'][layer];delta=after-before
    entry['projection']=dict(head_input_signed_projection=component_projection(fixed_w-before,delta),
      kernel_signed_projection=component_projection(after-fixed_w,delta),
      old_native_reconstruction_error=relative_error(before,old['projection_native'][layer]),
      peak_native_reconstruction_error=relative_error(after,peak['projection_native'][layer]),
      old_peak_write_cosine=describe(cosine(old['projection_native'][layer],peak['projection_native'][layer])))
    for label,values in (('old',old),('peak',peak)):
      x=values['residual_in'][layer];w=values['projection_native'][layer]
      x2=np.sum(x.astype(np.float64)**2,-1);w2=np.sum(w.astype(np.float64)**2,-1)
      dot=np.sum(x.astype(np.float64)*w,-1)
      fullbatch=dict(zip(metadata['residual_write_stats'],cases[label+'_control']['residual_write_stats'][layer]))
      for key in ('bf16_addition_error_relative_energy_sqrt','nonzero_write_elements_rounded_away_fraction'):
        fullbatch.pop(key,None)
      post=values['residual_after_attention'][layer]
      entry['write'][label]=dict(sampled_write_over_residual=describe(np.sqrt(w2/np.maximum(x2,1e-30))),
        sampled_write_residual_cosine=describe(cosine(x,w)),
        sampled_energy_growth_fraction=describe((2*dot+w2)/np.maximum(x2,1e-30)),
        sampled_materialized_addition_error=relative_error(post.astype(np.float64)-x,w),
        sampled_materialized_rounded_away_fraction=float(np.mean((post==x)&(w!=0))),
        fullbatch_stats=fullbatch)
      for seq_index,seq_id in enumerate(metadata['sequences']):
        for query_index,query in enumerate(metadata['queries']):
          matrix=values['M_in'][layer,seq_index,query_index]
          actual=values['M_out'][layer,seq_index,query_index]-matrix
          attention=values['attention_dM_reference'][layer,seq_index,query_index]
          mlp=values['memory_mlp_dM_reference'][layer,seq_index,query_index]
          entry['memory'].append(dict(state=label,sequence=seq_id,query=query,
            total_reference_error=relative_error(attention+mlp,actual),
            actual=weak_write(matrix,actual),attention=weak_write(matrix,attention),
            memory_mlp=weak_write(matrix,mlp),
            actual_in_fixed_old_basis=weak_write(old['M_in'][layer,seq_index,query_index],actual),
            attention_in_fixed_old_basis=weak_write(old['M_in'][layer,seq_index,query_index],attention)))
    result['layers'].append(entry)
  baseline=cases['peak_control']['loss']-cases['old_control']['loss']
  for name,case in cases.items():
    path=root/(name+'-pattern-write.npz')
    if name in ('old_control','peak_control') or not path.exists():continue
    assert case['fidelity_passed'],case
    values=dict(np.load(path,allow_pickle=False));route=route_difference(old['route'],values['route'])
    result['rollback_cases'].append(dict(name=name,loss=case['loss'],
      conditional_loss_rescue=(cases['peak_control']['loss']-case['loss'])/baseline,
      route_tv_to_old=[describe(route['tv'][i],route['valid'][i]) for i in range(6)],
      write_error_to_old=[relative_error(values['projection_native'][i],old['projection_native'][i]) for i in range(6)]))
  return result


if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
  args=parser.parse_args();result=analyze(args.root)
  dest=args.root/'paired-pattern-write-analysis.json';dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(dest)

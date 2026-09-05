"""Aggregate immutable row-cross probe artifacts; no TPU access required."""
import argparse
import json
from pathlib import Path
import numpy as np


def join(root, pattern, keys=None):
  arrays={}
  for p in sorted(Path(root).glob(pattern)):
    with np.load(p) as x:
      for k in keys or x.files: arrays.setdefault(k,[]).append(x[k])
  return {k:np.concatenate(v) for k,v in arrays.items()}


def describe(x):
  x=np.asarray(x,dtype=float)
  return dict(mean=float(x.mean()),ci95=float(1.96*x.std(ddof=1)/len(x)**.5),
              median=float(np.median(x)),positive=int((x>0).sum()),n=len(x))


def analyze(root, reference=None):
  root=Path(root)
  summary=json.loads((root/'summary.json').read_text())
  raw=join(root,'batch_*.npz',('loss','sequence_hashes')); metrics=join(root,'metrics_*.npz')
  if not np.array_equal(raw['sequence_hashes'],metrics['sequence_hashes']):
    raise ValueError('loss and metric cohorts differ')
  if len(set(raw['sequence_hashes']))!=128: raise ValueError('incomplete or repeated cohort')
  result=dict(metadata=summary, metric_forward_loss_drift=describe(metrics['loss']-raw['loss'][:,0]),
      interventions=[dict(arm=name,**describe(raw['loss'][:,i]-raw['loss'][:,0]))
                     for i,name in enumerate(summary['arms'])],layers=[])
  for i,layer in enumerate(summary['metric_layers']):
    a=metrics['alpha_sums'][:,i].astype(float); total=a.sum(0)
    v=metrics['contribution_normalized'][:,i]
    entry=dict(layer=layer,
       negative_count_fraction=float(total[1]/total[2]),
       negative_abs_mass_fraction=float(total[4]/(total[3]+total[4])),
       energy_mean=metrics['energy'][:,i].mean(0).tolist(),
       contribution=[describe(v[:,k]) for k in range(4)],
       cross_contribution=describe(v[:,3]-v[:,0]),
       positive_negative_cosine=describe(metrics['pos_neg_cosine'][:,i]),
       cancellation_ratio=describe(metrics['cancellation_ratio'][:,i]),
       decomposition_relative_error=describe(metrics['decomposition_relative_error'][:,i]))
    arm=f'L{layer}_no_negative'
    if arm in summary['arms']:
      harm=raw['loss'][:,summary['arms'].index(arm)]-raw['loss'][:,0]
      negative_mass=a[:,4]/(a[:,3]+a[:,4])
      entry['sample_correlations']=dict(
          negative_mass_fraction_vs_removal_loss=float(np.corrcoef(negative_mass,harm)[0,1]),
          negative_part_direct_ig_vs_removal_loss=float(np.corrcoef(v[:,2],harm)[0,1]))
    result['layers'].append(entry)
  if reference:
    old=join(reference,'residual_attribution_batch_*.npz',
             ('sequence_hashes','production_loss','contribution_normalized'))
    if not np.array_equal(old['sequence_hashes'],raw['sequence_hashes']):
      raise ValueError('historical cohort differs')
    result['historical_loss_drift']=describe(raw['loss'][:,0]-old['production_loss'])
    result['historical_row_ig_drift']=[]
    for i,layer in enumerate(summary['metric_layers']):
      result['historical_row_ig_drift'].append(dict(layer=layer,**describe(
          metrics['contribution_normalized'][:,i,3]-old['contribution_normalized'][:,layer,4:6].sum(1))))
  (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  return result


if __name__=='__main__':
  p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--reference')
  a=p.parse_args(); print(json.dumps(analyze(a.root,a.reference),indent=2))

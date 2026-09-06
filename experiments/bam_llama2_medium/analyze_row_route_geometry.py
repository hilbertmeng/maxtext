"""Read-only scalar routing/row-geometry analysis, preserving averaging units."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_row_mediation import stats


def sign_summary(root):
  root=Path(root);m=json.loads((root/'summary.json').read_text())
  arrays=[];counts=[]
  for p in sorted(root.glob('metrics_*.npz')):
    with np.load(p) as x:arrays.append(x['alpha_sums'])
    with np.load(root/p.name.replace('metrics_','batch_')) as x:counts.append(x['valid'].sum(-1))
  a=np.concatenate(arrays);n=np.concatenate(counts)
  rows=[]
  for i,l in enumerate(m['metric_layers']):
    positive=a[:,i,3]/n;negative=a[:,i,4]/n;coefficient=1+positive-negative
    rows.append(dict(layer=l,positive_cross_mass=stats(positive),negative_cross_mass=stats(negative),
        total_coefficient_sequence_mean=stats(coefficient),
        sequence_mean_quantiles=np.quantile(coefficient,[.1,.5,.9]).tolist()))
  return dict(root=str(root),metadata=m,n=len(n),layers=rows,
              limitation='These are within-sequence means, not per-token coefficient distributions.')


def consumer_geometry(root):
  root=Path(root);m=json.loads((root/'summary.json').read_text())
  geometry=[];coefficients=[];per_sequence=[];hashes=[]
  for p in sorted(root.glob('batch_*.npz')):
    with np.load(p) as x:
      for key in ['null_max_error','source_scope_error','unused_reference_error','immediate_cut_error']:
        if np.any(x[key]!=0):raise ValueError((str(p),key))
      hashes.extend(x['sequence_hashes'].tolist())
      g=x['source_geometry_token'];a=x['alpha_coefficient_sum'];valid=x['valid']
      for i in range(len(valid)):
        v=valid[i];c=a[i,v];z=g[i,v];coefficients.append(c);geometry.append(z)
        per_sequence.append([c.mean(),np.mean(abs(c)<.1),np.mean(abs(c)<.25),
                             np.mean(c<0),*z.mean(0)])
  if len(hashes)!=m['requested_sequences'] or len(set(hashes))!=len(hashes):
    raise ValueError('incomplete or duplicate cohort')
  a=np.concatenate(coefficients);g=np.concatenate(geometry);s=np.asarray(per_sequence)
  names=['coefficient_mean','fraction_abs_coefficient_lt_0.1',
         'fraction_abs_coefficient_lt_0.25','fraction_coefficient_negative',*m['source_geometry_columns']]
  return dict(root=str(root),metadata=m,n=len(s),sequence_means={k:stats(s[:,i]) for i,k in enumerate(names)},
      token_coefficient_quantiles=np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist(),
      token_self_cross_cosine_quantiles=np.quantile(g[:,3],[0,.05,.25,.5,.75,.95,1]).tolist(),
      token_cosine_coefficient_correlation=float(np.corrcoef(a,g[:,3])[0,1]))


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--sign-root',action='append',default=[])
  parser.add_argument('--consumer-root');parser.add_argument('--output',required=True);args=parser.parse_args()
  result=dict(sign_runs=[sign_summary(p) for p in args.sign_root])
  if args.consumer_root:result['consumer']=consumer_geometry(args.consumer_root)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print('ROUTE_GEOMETRY_ANALYSIS_READY '+args.output)

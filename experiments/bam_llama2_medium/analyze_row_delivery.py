"""Validate complete delivery probes and contrast each policy with its cutoff control."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_row_mediation import stats


def analyze(root):
  root = Path(root)
  meta = json.loads((root/'summary.json').read_text())
  rows, hashes = [], []
  for path in sorted(root.glob('batch_*.npz')):
    with np.load(path, allow_pickle=False) as data:
      if data['checks'].shape != (len(meta['checks']),) or np.any(data['checks'] != 0):
        raise ValueError(f'invalid endpoint controls: {path}')
      if not np.isfinite(data['token_loss']).all() or not np.isfinite(data['loss']).all():
        raise ValueError(f'nonfinite: {path}')
      hashes.extend(map(str, data['sequence_hashes']))
      rows.append(data['loss'].astype(float))
  a = np.concatenate(rows)
  if len(a) != meta['requested_sequences'] or len(set(hashes)) != len(a):
    raise ValueError('requires the complete unique cohort')
  names = {name: i for i,name in enumerate(meta['arms'])}
  results = []
  for name,i in names.items():
    if not name.startswith('keep_'):
      continue
    policy,end = name.removeprefix('keep_').rsplit('_through_L',1)
    control = names[f'keep_all_through_L{end}']
    results.append(dict(policy=policy,end=int(end),
        vs_clean=stats(a[:,i]-a[:,0]), vs_deleted=stats(a[:,i]-a[:,1]),
        vs_all_at_same_cutoff=stats(a[:,i]-a[:,control])))
  return dict(root=str(root),metadata=meta,n=len(a),complete=True,
      row_deletion=stats(a[:,1]-a[:,0]),results=results)


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root'); parser.add_argument('--output',required=True)
  args=parser.parse_args()
  result=analyze(args.root)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print('DELIVERY_ANALYSIS_READY',args.output,'n=',result['n'])

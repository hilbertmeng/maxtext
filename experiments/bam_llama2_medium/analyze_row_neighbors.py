"""Validate the matched cohort and report row deletion/joint contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_row_mediation import stats


def load(root):
  root=Path(root);meta=json.loads((root/'summary.json').read_text())
  hashes=[];values=[]
  for p in sorted(root.glob('batch_*.npz')):
    with np.load(p) as d:
      hashes.extend(d['sequence_hashes'].tolist());values.append(d['loss'])
  if len(set(hashes))!=len(hashes):raise ValueError('duplicate cohort sequences')
  if len(hashes)!=meta['requested_sequences']:raise ValueError('incomplete neighbor sweep')
  return meta,np.concatenate(values).astype(float),hashes


def analyze(root):
  meta,loss,hashes=load(root);arms=meta['arms'];rows=[]
  for layer in meta['layers']:
    ix=[arms.index(f'L{layer}_{c}') for c in ['cross','self','both']]
    rows.append(dict(layer=layer,**{c:stats(loss[:,i]-loss[:,0]) for c,i in zip(['cross','self','both'],ix)},
        interaction=stats(loss[:,ix[2]]-loss[:,ix[0]]-loss[:,ix[1]]+loss[:,0])))
  rank={c:sorted(meta['layers'],key=lambda l:loss[:,arms.index(f'L{l}_{c}')].mean(),reverse=True)
        for c in ['cross','self','both']}
  # Pairwise L11 contrasts preserve the covariance/outlier structure of the cohort.
  contrasts={c:{str(l):stats(loss[:,arms.index(f'L11_{c}')]-loss[:,arms.index(f'L{l}_{c}')])
                   for l in meta['layers'] if l!=11} for c in rank}
  return dict(metadata=meta,rows=rows,rank=rank,L11_minus_neighbor=contrasts)


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
  args=parser.parse_args();result=analyze(args.root)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print('NEIGHBOR_ANALYSIS_READY '+args.output)

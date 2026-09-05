"""Aggregate paired probe losses and verify the historical same-cohort baseline."""
import argparse
import json
from pathlib import Path
import numpy as np


def joined(directory, pattern, loss_key):
  losses, hashes = [], []
  for file in sorted(Path(directory).glob(pattern)):
    with np.load(file) as data:
      losses.append(data[loss_key])
      hashes.extend(map(str, data['sequence_hashes']))
  if not losses:
    raise ValueError(f'no batches in {directory}')
  return np.concatenate(losses).astype(np.float64), hashes


def analyze(directory, reference):
  losses, hashes = joined(directory, 'batch_*.npz', 'loss')
  old, old_hashes = joined(reference, 'residual_attribution_batch_*.npz', 'production_loss')
  if hashes != old_hashes:
    raise ValueError('historical and current cohort order differ')
  delta = losses - losses[:, :1]
  error = losses[:, 0] - old
  output = dict(sequences=len(hashes), unique_sequences=len(set(hashes)),
                historical_baseline_mean=float(old.mean()),
                historical_baseline_max_error=float(np.abs(error).max()),
                historical_baseline_mean_error=float(error.mean()), results=[])
  for i, scale in enumerate((1., .5, 0.)):
    d = delta[:, i]
    output['results'].append(dict(cross_scale=scale, loss=float(losses[:, i].mean()),
        delta_mean=float(d.mean()), ci95_halfwidth=float(1.96*d.std(ddof=1)/len(d)**.5),
        quantiles=dict(zip(('min','p05','p25','median','p75','p95','max'),
                          map(float,np.quantile(d,[0,.05,.25,.5,.75,.95,1])))),
        harmed_count=int((d>0).sum()), helped_count=int((d<0).sum())))
  output['full_removal_over_half_removal_delta'] = float(delta[:,2].mean()/delta[:,1].mean())
  return output


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('directory')
  parser.add_argument('reference')
  args = parser.parse_args()
  result = analyze(args.directory, args.reference)
  (Path(args.directory)/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))

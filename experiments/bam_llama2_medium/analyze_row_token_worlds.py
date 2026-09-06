"""Audit all-origin own-loss probes; retain conditional, non-additive semantics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_row_mediation import stats


def analyze(root):
  root = Path(root)
  meta = json.loads((root / 'summary.json').read_text())
  if hashlib.sha256((root / 'cohort.npz').read_bytes()).hexdigest() != meta['cohort_sha256']:
    raise ValueError('cohort file checksum mismatch')
  names = meta['arms']
  losses, hashes = [], []
  for path in sorted(root.glob('batch_*.npz')):
    with np.load(path, allow_pickle=False) as d:
      if d['checks'].shape != (len(meta['checks']),) or np.any(d['checks']):
        raise ValueError(f'failed exact controls: {path}')
      token = np.asarray(d['token_loss'], dtype=float)
      valid = np.asarray(d['valid'], dtype=bool)
      loss = np.asarray(d['loss'], dtype=float)
      if not np.isfinite(token).all() or not np.isfinite(loss).all():
        raise ValueError(f'nonfinite losses: {path}')
      if token.shape[:2] != loss.shape or loss.shape[1] != len(names):
        raise ValueError(f'wrong arm dimensions: {path}')
      recomputed = (token * valid[:, None]).sum(-1) / valid.sum(-1)[:, None]
      np.testing.assert_allclose(loss, recomputed, atol=2e-6, rtol=2e-6)
      hashes.extend(map(str, d['sequence_hashes']))
      losses.append(loss)
  a = np.concatenate(losses)
  if len(a) != meta['requested_sequences'] or len(set(hashes)) != len(a):
    raise ValueError('requires complete unique cohort')
  with np.load(root / 'cohort.npz', allow_pickle=False) as cohort:
    if hashes != list(map(str, cohort['sequence_hashes'])):
      raise ValueError('cohort ordering/hash mismatch')
  i = {name: n for n, name in enumerate(names)}
  clean, joint, own = (a[:, i[k]] for k in
      ('clean', 'all_origins_deleted', 'own_origin_only_deleted'))
  return dict(metadata=meta, exact_controls=True, n=len(a),
      collective_deletion=stats(joint-clean), own_origin_deletion=stats(own-clean),
      own_consumers={name:stats(a[:,index]-clean) for index,name in enumerate(names)
          if name.startswith(('deny_', 'joint_', 'cut_'))},
      interpretation=('Own-origin deletion measures that origin prediction only. '
          'Collective minus own is conditional on own deletion, not an additive '
          'fraction of transported benefit; receiver consumers are outside scope.'))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('root'); p.add_argument('--output', required=True)
  args = p.parse_args()
  result = analyze(args.root)
  Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({k: v for k, v in result.items() if k != 'metadata'}, indent=2))

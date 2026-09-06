"""Paired downstream mediation contrasts, with tokenwise historical anchors."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_row_mediation import stats


def analyze(root, anchor_root=None, allow_partial=False):
  root = Path(root)
  meta = json.loads((root/'summary.json').read_text())
  records = []
  for path in sorted(root.glob('batch_*.npz')):
    with np.load(path) as x:
      if np.any(x['checks'] != 0): raise ValueError(f'failed exact controls: {path}')
      for i, h in enumerate(x['sequence_hashes']):
        records.append((str(h), x['loss'][i], x['token_loss'][i], x['valid'][i]))
  if len(set(r[0] for r in records)) != len(records): raise ValueError('duplicate sequences')
  if not records: raise ValueError('no completed batches')
  if not allow_partial and len(records) != meta['requested_sequences']:
    raise ValueError(f'incomplete cohort {len(records)}/{meta["requested_sequences"]}')
  a = np.stack([r[1] for r in records]).astype(float)
  result = dict(root=str(root), metadata=meta, n=len(a), complete=len(a)==meta['requested_sequences'],
                row_deletion=stats(a[:, 1]-a[:, 0]), early_v_denial=stats(a[:, 2]-a[:, 0]),
                recipients=[])
  indices = {name: i for i, name in enumerate(meta['arms'])}
  for name, i in indices.items():
    if not name.startswith('block_'): continue
    j = indices['rescue_' + name.removeprefix('block_')]
    result['recipients'].append(dict(
        recipient=name.removeprefix('block_'),
        block_vs_clean=stats(a[:, i]-a[:, 0]),
        rescue_vs_denied=stats(a[:, j]-a[:, 2]),
        remaining_after_rescue_vs_clean=stats(a[:, j]-a[:, 0])))
  if anchor_root:
    anchor_root = Path(anchor_root)
    am = json.loads((anchor_root/'summary.json').read_text())
    if am['checkpoint'] != meta['checkpoint']: raise ValueError('different checkpoints')
    column = am['arms'].index(f'L{meta["source_layer"]}_{meta["source_component"]}')
    old = {}
    for path in sorted(anchor_root.glob('batch_*.npz')):
      with np.load(path) as x:
        for i, h in enumerate(x['sequence_hashes']):
          old[str(h)] = x['token_loss'][i, [0, column]]
    maxima = np.zeros(2); shifts = []
    for h, _, tokens, valid in records:
      delta = tokens[:2].astype(float)-old[h].astype(float)
      maxima = np.maximum(maxima, abs(delta).max(-1))
      shifts.append(delta[:, valid].mean(-1))
    shifts = np.asarray(shifts)
    result['historical_anchor'] = dict(root=str(anchor_root),
        clean_token_max_error=float(maxima[0]), row_deleted_token_max_error=float(maxima[1]),
        clean_mean_shift=stats(shifts[:, 0]), deleted_mean_shift=stats(shifts[:, 1]),
        deletion_effect_shift=stats(shifts[:, 1]-shifts[:, 0]))
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root'); parser.add_argument('--anchor-root')
  parser.add_argument('--allow-partial', action='store_true')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  result = analyze(args.root, args.anchor_root, args.allow_partial)
  Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
  print('V_EXPORT_ANALYSIS_READY', args.output, 'n=', result['n'], 'complete=', result['complete'])

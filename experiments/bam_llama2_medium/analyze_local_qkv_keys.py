"""Aggregate saved per-sequence key statistics without rerunning the checkpoint."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', type=Path)
  args = parser.parse_args()
  files = sorted(args.directory.glob('batch_*.npz'))
  if len(files) != 32:
    raise ValueError(f'Expected 32 complete batches, found {len(files)}')
  values = {}
  for path in files:
    with np.load(path) as batch:
      for name in batch.files:
        values.setdefault(name, []).append(batch[name])
  values = {name: np.concatenate(v, 0) for name, v in values.items()}
  assert all(v.shape[0] == 128 for v in values.values())
  summary = {name: dict(mean=np.nanmean(v, axis=0).tolist(),
                       standard_error=(np.nanstd(v, axis=0, ddof=1)/np.sqrt(np.isfinite(v).sum(0))).tolist())
             for name, v in values.items()}
  (args.directory/'summary.json').write_text(json.dumps(summary, indent=2))
  lines = ['# LocalQ/K/V runtime key similarity', '',
           '128 sequences; equal-sequence aggregation. Absolute cosine ignores the sign that signed head mixing can absorb.', '',
           'R2 energy uses four unit-normalized Q1/K1/V2 bases. It is a per-token adaptive rank bound, not a globally fixed sharing scheme.', '']
  for side in ('row', 'col'):
    lines += [f'## {side}', '',
        '| Layer | signed QK | abs QK | centered abs QK | Q in V span | K in V span | joint R2 energy | effective QV cos² | effective KV cos² |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for l in range(24):
      p = f'L{l:02d}_{side}'
      get = lambda k: np.asarray(summary[p+'_'+k]['mean'])
      common = [get('key_uncentered_cos')[0,1], get('key_uncentered_abs_cos')[0,1],
                get('key_centered_abs_cos')[0,1]]
      extra = [get('q_in_v_span')[0], get('k_in_v_span')[0], get('key_uncentered_rank_energy')[1],
               get('effective_qv_cos2'), get('effective_kv_cos2')] if l % 3 != 2 else [None]*5
      cells = ['—' if v is None else f'{float(v):.4f}' for v in common+extra]
      lines.append(f'| {l} | '+ ' | '.join(cells)+' |')
  (args.directory/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
  main()

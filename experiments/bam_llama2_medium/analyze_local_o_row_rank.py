"""Aggregate per-sequence spectra without losing layer or L/F distinctions."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', type=Path)
  args = parser.parse_args()
  files = sorted(args.directory.glob('sample_*.npz'))
  if not files:
    raise SystemExit('No completed samples')
  values = {}
  for path in files:
    with np.load(path) as data:
      for k in data.files:
        values.setdefault(k, []).append(np.asarray(data[k]))
  mean = {k: np.nanmean(v, axis=0) for k,v in values.items()}
  np.savez_compressed(args.directory/'aggregate.npz', **mean)
  lines = [f'# LocalO row rank: {len(files)}/128 sequences', '',
           'All ranks are per-token head-matrix ranks; L0 is structurally zero.', '',
           '| Layer | Type | key r95 | output r95 | key E2/E4/E8 | output E2/E4/E6 | key-SVD output squared error r2/r4/r8 |',
           '|---|---|---:|---:|---|---|---|']
  for l in range(24):
    prefix=f'L{l:02d}_'
    def get(k): return mean[prefix+k]
    def selected(k, ranks): return '/'.join(f'{get(k)[r-1]:.4f}' for r in ranks)
    lines.append(f'| {l} | {"F" if l%3==2 else "L"} | {get("key_amplitude_r95"):.2f} | '
                 f'{get("out_amplitude_r95"):.2f} | {selected("key_amplitude_retained_mean",[2,4,8])} | '
                 f'{selected("out_amplitude_retained_mean",[2,4,6])} | '
                 f'{selected("key_svd_read_relative_squared_error",[2,4,8])} |')
  lines += ['', '## Head correlations and direction-only spectra', '',
            'Cosines exclude diagonal pairs; signed and absolute values are both reported.',
            'Direction spectra normalize each head vector before SVD, removing head-amplitude dominance.', '',
            '| Layer | Type | key mean cos | key mean abs cos | output mean cos | output mean abs cos | key direction E4 | output direction E4 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
  off = ~np.eye(16, dtype=bool)
  for l in range(1,24):
    p=f'L{l:02d}_'
    vals=[np.mean(mean[p+k][off]) for k in ('key_cos','key_abs_cos','out_cos','out_abs_cos')]
    vals += [mean[p+k+'_direction_retained_mean'][3] for k in ('key','out')]
    lines.append(f'| {l} | {"F" if l%3==2 else "L"} | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
  lines += ['', '## Full rank curves', '',
            'Output ranks above8 add no native-space capacity. Key ranks span1–16.', '']
  for l in range(1,24):
    lines += [f'### L{l} ({"F" if l%3==2 else "L"})', '']
    for stage in ('raw','norm','key','out'):
      a=mean[f'L{l:02d}_{stage}_amplitude_retained_mean']
      lines.append(f'- {stage} energy rank1–{len(a)}: '+', '.join(f'{v:.5f}' for v in a))
    lines.append('')
  (args.directory/'summary.md').write_text('\n'.join(lines)+'\n')
  print('\n'.join(lines[:30]))

if __name__ == '__main__':
  main()

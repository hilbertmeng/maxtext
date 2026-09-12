"""Aggregate raw Q/K basis geometry; no checkpoint rerun required."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', type=Path)
  parser.add_argument('--allow-partial', action='store_true')
  args = parser.parse_args()
  files = sorted(args.directory.glob('batch_*.npz'))
  if not files or (not args.allow_partial and len(files) != 128):
    raise ValueError(f'Expected 128 samples, found {len(files)}')
  values = {}
  for path in files:
    with np.load(path) as batch:
      for name in batch.files:
        values.setdefault(name, []).append(batch[name])
  values = {k: np.concatenate(v, axis=0) for k, v in values.items()}
  summary = {}
  for k, v in values.items():
    count = np.isfinite(v).sum(0)
    mean = np.divide(np.nansum(v, 0), count, out=np.full(v.shape[1:], np.nan), where=count>0)
    summary[k] = dict(mean=mean.tolist(), valid_samples=count.tolist())
  (args.directory/'summary.json').write_text(json.dumps(summary, indent=2))
  lines = ['# XL QK rank-4 raw basis geometry', '',
      f'{len(files)} sequences, equal-sequence means, all valid tokens. Projection plus bias only.',
      'Joint rank-4 retained energy is the per-token optimal SVD bound, not proof of a realizable shared projection.',
      'Span coverage uses unit-normalized source bases; signed/absolute cosines average all 4×4 Q–K pairs.',
      'L0 zero bases are undefined and excluded. R4 energy is squared singular-value energy, not vector norm.', '']
  for side in ('row', 'col'):
    lines += [f'## {side}', '',
        '| Layer | mode | signed QK | abs QK | Q in K span | K in Q span | Q R2 energy | K R2 energy | joint R4 energy | joint unit R4 energy |',
        '|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for layer in range(24):
      p = f'L{layer:02d}_{side}_'
      get = lambda k: np.asarray(summary[p+k]['mean'])
      cells = [get('raw_uncentered_cos')[:4,4:].mean(),
          get('raw_uncentered_abs_cos')[:4,4:].mean(),
          get('q_in_k_span').mean(), get('k_in_q_span').mean(),
          get('q_rank_energy')[1], get('k_rank_energy')[1],
          get('raw_uncentered_amplitude_weighted_rank_energy')[3],
          get('raw_uncentered_rank_energy')[3]]
      text = ['—' if layer == 0 or not np.isfinite(v) else f'{v:.4f}' for v in cells]
      lines.append(f'| {layer} | {"F" if layer%3==2 else "L"} | '+' | '.join(text)+' |')
  if 'dynamic__L01_row_q_rank_energy' in summary:
    lines += ['', '## Bias decomposition', '',
        'Ratios are ratios of mean per-token norms; they are not additive energy shares.',
        'Static b is token-independent; its centered geometry is undefined and unused.', '']
    for side in ('row', 'col'):
      lines += [f'### {side}', '',
          '| Layer | Q bias/dynamic norm | K bias/dynamic norm | Q dynamic–bias cosine | K dynamic–bias cosine | joint R4 Wx+b | joint R4 Wx | joint R4 b |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
      for layer in range(1,24):
        key = f'L{layer:02d}_{side}_raw_uncentered_amplitude_weighted_rank_energy'
        get = lambda k: np.asarray(summary[k]['mean'])
        cells = []
        for arm in ('q','k'):
          p = f'L{layer:02d}_{arm}_{side}_'
          cells.append(get(p+'bias_norm').mean()/max(get(p+'dynamic_norm').mean(),1e-30))
        cells += [get(f'L{layer:02d}_{arm}_{side}_dynamic_bias_cos').mean() for arm in ('q','k')]
        cells += [get(stage+key)[3] for stage in ('','dynamic__','bias__')]
        lines.append(f'| {layer} | '+' | '.join(f'{v:.4f}' for v in cells)+' |')
  (args.directory/'summary.md').write_text('\n'.join(lines)+'\n')
  detail = ['# Per-layer, per-side Q–K basis details', '',
      f'{len(files)} sequences. Rows Q0–Q3; columns K0–K3. Spectra are cumulative squared-energy fractions.', '']
  stages = ('', 'dynamic__', 'bias__') if 'dynamic__L01_row_q_rank_energy' in summary else ('',)
  for stage in stages:
    detail += ['## '+({'':'Wx+b','dynamic__':'Wx','bias__':'b'}[stage]), '']
    for side in ('row','col'):
      detail += [f'### {side}', '']
      for layer in range(1,24):
        p = stage+f'L{layer:02d}_{side}_'
        get = lambda k: np.asarray(summary[p+k]['mean'])
        detail += [f'#### L{layer}', '', '```text']
        for metric in ('cos','abs_cos'):
          matrix = get('raw_uncentered_'+metric)[:4,4:]
          detail += [metric+'       K0       K1       K2       K3']
          detail += [f'Q{i}    '+' '.join(f'{v:+.5f}' for v in row) for i,row in enumerate(matrix)]
        for label,key in (('Q spectrum','q_rank_energy'),('K spectrum','k_rank_energy'),
                          ('joint spectrum','raw_uncentered_amplitude_weighted_rank_energy'),
                          ('Q in K span','q_in_k_span'),('K in Q span','k_in_q_span')):
          detail += [label+': '+' '.join(f'{v:.5f}' for v in get(key))]
        detail += ['```','']
  (args.directory/'layer_details.md').write_text('\n'.join(detail)+'\n')


if __name__ == '__main__':
  main()

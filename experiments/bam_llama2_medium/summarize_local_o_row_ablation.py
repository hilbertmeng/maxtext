"""Incremental paired summaries; run after each 32-sequence milestone."""
import argparse
import json
import contextlib
import io
from pathlib import Path
import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('directory', type=Path)
  parser.add_argument('--limit', type=int, default=128)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  scenarios = json.loads((args.directory / 'ablation_groups_scenarios.json').read_text())
  rows = []
  for i in range(args.limit):
    path = args.directory / f'ablation_groups_{i:03d}.npz'
    if not path.exists():
      break
    with np.load(path) as data:
      rows.append(data['gap'].reshape(len(scenarios)))
  if not rows:
    raise SystemExit('No completed samples')
  gaps = np.stack(rows)
  assert np.isfinite(gaps).all()
  stream = io.StringIO()
  with contextlib.redirect_stdout(stream):
    report(gaps, scenarios)
  result = stream.getvalue()
  if args.output:
    args.output.write_text(result)
  print(result, end='')


def report(gaps, scenarios):
  print(f'Contiguous paired sequences: {len(gaps)}; positive gap = harm.')
  print('Intervals are descriptive mean ± 1.96 SE, not multiple-test-corrected.')
  print('| Group | Oracle | Rank | Mean gap | ±1.96 SE | Positive fraction |')
  print('|---|---|---:|---:|---:|---:|')
  for j, (group, mask, rank, mode) in enumerate(scenarios):
    x = gaps[:, j]
    if (mode == 1 and rank == 16) or (mode == 2 and rank == 8):
      np.testing.assert_array_equal(x, np.zeros_like(x))
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    print(f'| {group} | {"key" if mode == 1 else "output"} | {rank} | {x.mean():+.6f} | {1.96*se:.6f} | {(x>0).mean():.3f} |')


if __name__ == '__main__':
  main()

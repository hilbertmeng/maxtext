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
  parser.add_argument('--focus', action='store_true', help='Merge original and focused samples for ranks4/6.')
  args = parser.parse_args()
  scenarios = json.loads((args.directory / 'ablation_groups_scenarios.json').read_text())
  original_scenarios = scenarios
  if args.focus:
    scenarios = json.loads((args.directory / 'ablation_groups_focus_scenarios.json').read_text())
    original_indices = [original_scenarios.index(s) for s in scenarios]
    old_meta = json.loads((args.directory / 'ablation_groups_metadata.json').read_text())
    new_meta = json.loads((args.directory / 'ablation_groups_focus_metadata.json').read_text())
    assert old_meta['sequence_hashes'] == new_meta['sequence_hashes']
    assert old_meta['checkpoint'] == new_meta['checkpoint']
  rows = []
  for i in range(args.limit):
    path = args.directory / f'ablation_groups_{i:03d}.npz'
    original = path.exists()
    if not original and args.focus:
      path = args.directory / f'ablation_groups_focus_{i:03d}.npz'
    if not path.exists():
      break
    with np.load(path) as data:
      values = data['gap'].reshape(-1)
      rows.append(values[original_indices] if args.focus and original else values)
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

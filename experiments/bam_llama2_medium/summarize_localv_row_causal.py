"""Aggregate paired per-sequence losses; keep layer and side distinctions."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--limit', type=int, default=128)
args = p.parse_args()
for stage in ('dose', 'route', 'qk', 'focus'):
  path = args.directory / f'{stage}_scenarios.json'
  if not path.exists():
    continue
  scenarios = json.loads(path.read_text())
  samples = []
  for i in range(args.limit):
    file = args.directory / f'{stage}_{i:03d}.npz'
    if not file.exists():
      break
    with np.load(file) as data:
      samples.append(data['gap'].reshape(len(scenarios)))
  if not samples:
    continue
  x = np.stack(samples)
  assert np.max(np.abs(x[:,0])) == 0
  mean = x.mean(0)
  se = x.std(0,ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.full(len(mean),np.nan)
  print(f'\n## {stage}: {len(x)} paired sequences; positive=higher loss\n')
  print('| Intervention | Mean gap | ±1.96 SE (descriptive) | Fraction gap>0 |')
  print('|---|---:|---:|---:|')
  for c,m,s,f in zip(scenarios,mean,se,(x>0).mean(0)):
    print(f'| {c["name"]} | {m:+.7f} | {1.96*s:.7f} | {f:.3f} |')

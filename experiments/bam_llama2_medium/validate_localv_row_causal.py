"""Verify complete paired artifacts and independent intervention endpoints."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--count', type=int, default=64)
a = p.parse_args()
stages = {}
cohort_hash = None
for stage in ('dose', 'qk', 'route', 'focus'):
  metadata = json.loads((a.directory / f'{stage}_metadata.json').read_text())
  assert metadata['checkpoint_step'] == 5250
  if cohort_hash is None:
    cohort_hash = metadata['cohort_sha256']
  assert metadata['cohort_sha256'] == cohort_hash
  cases = json.loads((a.directory / f'{stage}_scenarios.json').read_text())
  losses = []
  for i in range(a.count):
    with np.load(a.directory / f'{stage}_{i:03d}.npz') as data:
      assert str(data['sequence_hash']) == metadata['sequence_hashes'][i]['inputs']
      assert data['loss'].shape == (len(cases), 1) and np.isfinite(data['loss']).all()
      np.testing.assert_array_equal(data['gap'], data['loss'] - data['baseline'])
      assert data['gap'][0,0] == 0
      losses.append(data['loss'][:,0])
  stages[stage] = {c['name']: np.stack(losses)[:,j] for j,c in enumerate(cases)}
  print(f'{stage}: {a.count}/{a.count} valid, commit={metadata["diagnostic_commit"]}')
for stage, cases in stages.items():
  np.testing.assert_array_equal(cases['native'], stages['dose']['native'])
for name, values in stages['dose'].items():
  if name.endswith('_dose0'):
    np.testing.assert_array_equal(values, stages['route'][name.replace('_dose0', '_both_off')])
for focus, stage, name in [('V_all_off','dose','all_L_dose0'),
                          ('V_L1_off','dose','L01_dose0'), ('QK_all_off','qk','all_QK_dose0')]:
  np.testing.assert_array_equal(stages['focus'][focus], stages[stage][name])
print('VALIDATED: sequence hashes, native losses, all17 transport endpoints, three focus overlaps exact')

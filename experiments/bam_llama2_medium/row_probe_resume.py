"""Validated scalar-artifact resume; no model state or activation references."""
import json
from pathlib import Path

import numpy as np


def resume_batches(output, metadata, cohort, previous_commit=None):
  output = Path(output)
  files = sorted(output.glob('batch_*.npz'))
  if not files:
    return {}
  old = json.loads((output / 'summary.json').read_text())
  allowed = {metadata['diagnostic_commit']}
  if previous_commit:
    allowed.add(previous_commit)
  if old['diagnostic_commit'] not in allowed:
    raise ValueError('resume requires the same runtime or an explicitly audited previous commit')
  for key in ('base_config_class', 'checkpoint', 'trainer_commit', 'cohort_sha256',
              'source_layer', 'source_component', 'source_positions',
              'early_v_cross_layers', 'batch_size', 'requested_sequences', 'arms', 'checks'):
    if (key in old) != (key in metadata) or old.get(key) != metadata.get(key):
      raise ValueError(f'incompatible resume metadata: {key}')
  for key in ('controls', 'arm_set', 'own_only'):
    if (key in old) != (key in metadata) or old.get(key) != metadata.get(key):
      raise ValueError(f'incompatible resume metadata: {key}')
  saved = {}
  size = metadata['batch_size']
  for path in files:
    offset = int(path.stem.split('_')[1])
    if offset % size or not 0 <= offset < len(cohort['inputs']):
      raise ValueError(f'invalid batch offset: {path}')
    expected = cohort['sequence_hashes'][offset:offset + size]
    with np.load(path, allow_pickle=False) as data:
      if not np.array_equal(data['sequence_hashes'], expected):
        raise ValueError(f'cohort mismatch: {path}')
      valid = cohort['targets_segmentation'][offset:offset + size] != 0
      if not np.array_equal(data['valid'], valid):
        raise ValueError(f'valid-mask mismatch: {path}')
      loss = data['loss']
      tokens = data['token_loss']
      if loss.shape != (len(expected), len(metadata['arms'])):
        raise ValueError(f'arm shape mismatch: {path}')
      if tokens.shape != loss.shape + (valid.shape[-1],):
        raise ValueError(f'token shape mismatch: {path}')
      if data['checks'].shape != (len(metadata['checks']),) or np.any(data['checks'] != 0):
        raise ValueError(f'failed numerical audits: {path}')
      if not np.isfinite(loss).all() or not np.isfinite(tokens).all():
        raise ValueError(f'nonfinite saved result: {path}')
      saved[offset] = loss.copy()
  metadata['resume'] = dict(previous_commit=old['diagnostic_commit'],
                           batch_offsets=sorted(saved),
                           previous_resume=old.get('resume'))
  return saved


def save_batch(path, **arrays):
  path = Path(path)
  pending = path.with_suffix('.pending')
  with pending.open('wb') as stream:
    np.savez_compressed(stream, **arrays)
  pending.replace(path)


def save_summary(path, metadata):
  path = Path(path)
  pending = path.with_suffix('.pending')
  pending.write_text(json.dumps(metadata, indent=2) + '\n')
  pending.replace(path)

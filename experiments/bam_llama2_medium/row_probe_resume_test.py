import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from row_probe_resume import resume_batches, save_batch, save_summary


class ResumeTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.path = Path(self.tmp.name)
    self.meta = dict(diagnostic_commit='old', base_config_class='model', checkpoint='ckpt',
        trainer_commit='trainer', cohort_sha256='cohort', source_layer=11,
        source_component='both', source_positions='all valid', early_v_cross_layers=[12],
        batch_size=1, requested_sequences=2, arms=['clean', 'deleted'], checks=['null'])
    self.cohort = dict(inputs=np.ones((2, 3)), targets_segmentation=np.ones((2, 3)),
                       sequence_hashes=np.asarray(['a', 'b']))
    self.arrays = dict(loss=np.zeros((1, 2)), token_loss=np.zeros((1, 2, 3)),
        valid=np.ones((1, 3), bool), checks=np.zeros(1), sequence_hashes=np.asarray(['a']))
    save_batch(self.path / 'batch_000.npz', **self.arrays)
    save_summary(self.path / 'summary.json', self.meta)

  def test_resume_and_atomic_publication(self):
    self.assertEqual(list(resume_batches(self.path, self.meta, self.cohort)), [0])
    self.assertFalse(list(self.path.glob('*.pending')))

  def test_runtime_requires_explicit_audit(self):
    new = dict(self.meta, diagnostic_commit='new')
    with self.assertRaises(ValueError):
      resume_batches(self.path, new, self.cohort)
    self.assertEqual(list(resume_batches(self.path, new, self.cohort, 'old')), [0])
    self.assertEqual(new['resume']['previous_commit'], 'old')

  def test_configuration_mismatch(self):
    for key, value in [('source_component', 'self'), ('early_v_cross_layers', [12, 13]),
                       ('checkpoint', 'other')]:
      with self.subTest(key=key), self.assertRaises(ValueError):
        resume_batches(self.path, dict(self.meta, **{key: value}), self.cohort)

  def test_bad_batch(self):
    for key, value in [('checks', np.ones(1)), ('sequence_hashes', np.asarray(['x'])),
                       ('valid', np.zeros((1, 3), bool)), ('loss', np.full((1, 2), np.nan))]:
      save_batch(self.path / 'batch_000.npz', **dict(self.arrays, **{key: value}))
      with self.subTest(key=key), self.assertRaises(ValueError):
        resume_batches(self.path, self.meta, self.cohort)

  def test_own_world_metadata_without_export_fields(self):
    meta = dict(self.meta)
    del meta['early_v_cross_layers']
    save_summary(self.path / 'summary.json', meta)
    self.assertEqual(list(resume_batches(self.path, meta, self.cohort)), [0])
    with self.assertRaises(ValueError):
      resume_batches(self.path, dict(meta, early_v_cross_layers=[]), self.cohort)
    with self.assertRaises(ValueError):
      resume_batches(self.path, dict(meta, controls={'clean': [0]}), self.cohort)


if __name__ == '__main__':
  unittest.main()

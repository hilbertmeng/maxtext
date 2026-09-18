"""Regression for truncated TPU kernel buffers with intact step markers."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from analyze_bam_xplane import summarize, classify_local


class CoverageTest(unittest.TestCase):
  def test_packed_projection_outside_read_scope_is_bam(self):
    events = [
        dict(ph='M', name='process_name', pid=1, args={'name': '/device:TPU:0'}),
        dict(ph='X', pid=1, name='jit_train_step(test)', ts=0, dur=1000),
        dict(ph='X', pid=1, name='dot.1', ts=0, dur=1000,
             args={'tf_op': 'bam/local_packed_projection/W_local_packed/dot_general'}),
    ]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'trace.json.gz'
      with gzip.open(path, 'wt') as stream:
        json.dump({'traceEvents': events}, stream)
      _, buckets = summarize(path)[1]
    self.assertEqual(buckets['local_packed_projection'][0], 1.)
    self.assertEqual(buckets['bam_total'][0], 1.)
    self.assertEqual(buckets['non_bam_xla_ops'][0], 0.)

  def test_direct_column_contraction_is_not_other(self):
    self.assertEqual(classify_local(
        'bam/read_local_m_for_qk/bam/contract_1a_col/dot_general'), 'read_m')

  def test_independent_local_v_counts_once(self):
    events = [
        dict(ph='M', name='process_name', pid=1, args={'name': '/device:TPU:0'}),
        dict(ph='X', pid=1, name='jit_train_step(test)', ts=0, dur=1000),
        dict(ph='X', pid=1, name='fusion.1', ts=0, dur=1000,
             args={'tf_op': 'bam/read_local_m_for_v/bam/read_m_contract/mul'}),
    ]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'trace.json.gz'
      with gzip.open(path, 'wt') as stream:
        json.dump({'traceEvents': events}, stream)
      _, buckets = summarize(path)[1]
    self.assertEqual(buckets['local_v.read_m'][0], 1.)
    self.assertEqual(buckets['bam_total'][0], 1.)
    self.assertEqual(buckets['non_bam_xla_ops'][0], 0.)

  def test_partial_second_step_does_not_dilute_scopes(self):
    events = [dict(ph='M', name='process_name', pid=1,
                   args={'name': '/device:TPU:0'})]
    for start, covered in ((0, 999), (2000, 500)):
      events.extend([
          dict(ph='X', pid=1, name='jit_train_step(test)', ts=start, dur=1000),
          dict(ph='X', pid=1, name='1', ts=start, dur=1000),
          dict(ph='X', pid=1, name='while.1', ts=start, dur=1000),
          dict(ph='X', pid=1, name='fusion.1', ts=start, dur=covered,
               args={'tf_op': 'bam/write_m/P_loc_up/dot_general:'}),
      ])
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'trace.json.gz'
      with gzip.open(path, 'wt') as stream:
        json.dump({'traceEvents': events}, stream)
      step_ms, buckets = summarize(path)[1]
    self.assertEqual(step_ms, 1.)
    self.assertAlmostEqual(buckets['write_m'][0], .999)
    self.assertAlmostEqual(buckets['all_xla_ops'][0], .999)


if __name__ == '__main__':
  unittest.main()

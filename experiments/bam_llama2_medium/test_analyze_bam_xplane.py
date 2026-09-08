"""Regression for truncated TPU kernel buffers with intact step markers."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from analyze_bam_xplane import summarize


class CoverageTest(unittest.TestCase):
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

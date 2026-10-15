"""Coverage and masking regressions for independently packed validation data."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pile_validation_data import read_records, rewindow, write_record


class ValidationDataTest(unittest.TestCase):
  def test_roundtrip_across_source_cuts_and_short_tail(self):
    spec = importlib.util.spec_from_file_location('repack', Path(__file__).resolve().parents[2] / 'scripts/repack_pile_validation.py')
    repack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repack)
    tokens = np.array([11, 0, 22, 33, 0, 44, 55, 66], dtype='<u2')
    with tempfile.TemporaryDirectory() as root:
      source, output = Path(root) / 'source.tf', Path(root) / 'output.tf'
      with source.open('wb') as stream:
        write_record(stream, tokens[:3])
        write_record(stream, tokens[3:])
      manifest = repack.build(source, output, 3, hashlib.sha256(tokens.tobytes()).hexdigest())
      actual = list(read_records(output))
      self.assertEqual([x.tolist() for x in actual], [[11,0,22,33], [33,0,44,55], [55,66]])
      self.assertEqual(manifest['target_tokens_including_eos'], 7)
      self.assertEqual(manifest['target_tokens_excluding_eos'], 5)
      self.assertEqual(manifest['record_length_counts'], {4: 2, 2: 1})

  def test_wrong_source_digest_does_not_publish_dataset(self):
    spec = importlib.util.spec_from_file_location('repack', Path(__file__).resolve().parents[2] / 'scripts/repack_pile_validation.py')
    repack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repack)
    with tempfile.TemporaryDirectory() as root:
      source, output = Path(root) / 'source.tf', Path(root) / 'output.tf'
      with source.open('wb') as stream:
        write_record(stream, [11, 0, 22, 33])
      with self.assertRaisesRegex(ValueError, 'audited source'):
        repack.build(source, output, 3, 'incorrect digest')
      self.assertFalse(output.exists())
      self.assertFalse(Path(str(output) + '.manifest.json').exists())

  def test_crc_corruption_is_fatal(self):
    with tempfile.TemporaryDirectory() as root:
      path = Path(root) / 'source.tf'
      with path.open('wb') as stream:
        write_record(stream, [1,2,3])
      payload = bytearray(path.read_bytes())
      payload[13] ^= 1
      path.write_bytes(payload)
      with self.assertRaisesRegex(ValueError, 'CRC mismatch'):
        list(read_records(path))

  def test_exact_boundary_has_no_targetless_tail(self):
    actual = list(rewindow([np.arange(7, dtype='<u2')], 3))
    self.assertEqual([r.tolist() for r in actual], [[0,1,2,3], [3,4,5,6]])



if __name__ == '__main__':
  unittest.main()

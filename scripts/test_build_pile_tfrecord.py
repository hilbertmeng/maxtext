"""Check contiguous source windows and transactional dataset publication."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location('builder', Path(__file__).with_name('build_pile_tfrecord_4096.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class PileBuildTest(unittest.TestCase):
    def test_true_windows_and_overlap_across_document_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'docs'
            tokens = np.arange(20000, dtype=np.uint16)
            sizes = np.array([6000, 7000, 7000], dtype=np.int32)
            pointers = np.array([0, 12000, 26000], dtype=np.int64)
            prefix.with_suffix('.bin').write_bytes(tokens.tobytes())
            prefix.with_suffix('.idx').write_bytes(
                b'MMIDIDX\x00\x00' + struct.pack('<QBQQ', 1, 8, 3, 3)
                + sizes.tobytes() + pointers.tobytes())
            boundaries = np.arange(0, 18433, 2048)
            docs = np.searchsorted(np.cumsum(sizes), boundaries, side='right')
            starts = np.array([0, 6000, 13000])[docs]
            sample_idx = np.stack([docs, boundaries - starts], axis=1)
            dataset = b.MMapDocumentDataset(str(prefix))
            for length in (2048, 4096, 8192):
                stride = length // 2048
                a = b.materialize_sample(dataset, np.arange(3), sample_idx, 0, stride, length + 1)
                c = b.materialize_sample(dataset, np.arange(3), sample_idx, 1, stride, length + 1)
                np.testing.assert_array_equal(a, tokens[:length+1])
                np.testing.assert_array_equal(c, tokens[length:2*length+1])
                self.assertEqual(a[-1], c[0])
                self.assertEqual(np.intersect1d(a[1:], c[1:]).size, 0)

    def run_fake_build(self, directory, fail=False):
        args = argparse.Namespace(work_dir=directory, data_prefix='/cache/docs',
            doc_idx='/cache/doc_idx.npy', sample_idx='/cache/sample_idx.npy',
            output_uri='gs://bucket/pile8192', seq_length=8192, base_seq_length=2048,
            global_batch_size=64, steps=1000, steps_per_shard=500, seed=9876,
            workers=2, validate_only=False, boundary_stride=4,
            source_uri_prefix='gs://source/raw')
        uploads = []
        def shard(task):
            if fail:
                raise RuntimeError('incomplete shard')
            return dict(shard=f'pile.tfrecord.b{task[0]*500}', samples=32000,
                        bytes=123, crc32c='test')
        with mock.patch.object(b, 'parse_args', return_value=args), \
             mock.patch.object(b, 'validate_inputs', return_value=(100000, np.dtype('uint16'))), \
             mock.patch.object(b, 'build_selected_ids', return_value=(Path(directory)/'ids.npy', 'sha')), \
             mock.patch.object(b, 'write_shard', side_effect=shard), \
             mock.patch.object(b.concurrent.futures, 'ProcessPoolExecutor', concurrent.futures.ThreadPoolExecutor), \
             mock.patch.object(b, 'run', side_effect=lambda command: uploads.append(command)):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'incomplete shard'):
                    b.main()
            else:
                b.main()
        return uploads

    def test_failed_shard_cannot_publish_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_fake_build(directory, fail=True), [])
            self.assertFalse((Path(directory)/'manifest.json').exists())

    def test_manifest_contains_verified_coverage_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            uploads = self.run_fake_build(directory)
            self.assertEqual(len(uploads), 1)
            self.assertTrue(uploads[0][-1].endswith('/manifest.json'))
            manifest = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(manifest['record_length'], 8193)
            self.assertEqual(manifest['samples'], 64000)
            self.assertEqual(manifest['source_data_prefix'], 'gs://source/raw/docs')
            self.assertEqual(len(manifest['shards']), 2)

    def test_crc_mismatch_blocks_existing_shard(self):
        with mock.patch.object(b.subprocess, 'check_output', return_value=json.dumps(
                dict(size=120, crc32c_hash='wrong'))):
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                b.verify_object('gs://bucket/shard', dict(bytes=120, crc32c='expected'))


if __name__ == '__main__':
    unittest.main()

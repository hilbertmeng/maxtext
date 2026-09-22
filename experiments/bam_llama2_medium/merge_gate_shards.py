"""Merge immutable sequence outputs, checking runtime and cohort identity."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def merge(parts, out):
    metadata = [json.loads((p/'metadata.json').read_text()) for p in parts]
    for key in ['model', 'checkpoint', 'dtype', 'matmul_precision', 'protocol',
                'runtime', 'cohort_sha256', 'small_gate_cutoff', 'intervention_mode']:
        assert all(m[key] == metadata[0][key] for m in metadata), key
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    for part, meta in zip(parts, metadata):
        for source in sorted(part.glob('seq_*.json')):
            record = json.loads(source.read_text())
            i = record['sequence']
            assert meta['evaluation_shard'][0] <= i < meta['evaluation_shard'][1]
            assert i not in seen, f'duplicate sequence {i}'
            seen[i] = str(part)
            for file in [source, part/f'tokens_{i:03d}.npz']:
                assert file.is_file()
                target = out/file.name
                if target.exists():
                    assert hashlib.sha256(file.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
                else:
                    shutil.copy2(file, target)
    assert sorted(seen) == list(range(32,128)), f'incomplete: {len(seen)}/96'
    meta = dict(metadata[0], evaluation_shard=[32,128], merged_shards=metadata)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    (out/'sequence_sources.json').write_text(json.dumps(seen, indent=2))
    (out/'DONE').write_text('96 paired sequences verified complete [32,128)\n')
    print('MERGED_COMPLETE', len(seen))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('parts', type=Path, nargs='+')
    a = p.parse_args()
    merge(a.parts, a.out)

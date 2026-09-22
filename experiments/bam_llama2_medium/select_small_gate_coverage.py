"""Select read-open/small-write heads using discovery coverage, never loss."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np


def count(task):
    path, valid, thresholds = task
    a = np.load(path, mmap_mode='r')
    mask = (a[..., 26] >= thresholds[:, None, :]) & (a[..., 10] <= .02)
    return (mask & valid[None, :, None]).sum(axis=1)


def select(raw, original, treatments, out, workers=8):
    base = json.loads(original.read_text())
    protocol = json.loads(treatments.read_text())
    stats = base['all_head_stats']
    thresholds = np.full((24, 16), 2.)
    for h in stats:
        if 2 <= h['layer'] <= 22:
            thresholds[h['layer'], h['head']] = h['threshold']
    valid = np.load(raw / 'valid.npy')
    tasks = [(raw / f'sample_{i:03d}.npy', valid[i], thresholds) for i in range(32)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        counts = np.stack(list(pool.map(count, tasks)))
    totals, sequences = counts.sum(0), (counts > 0).sum(0)
    heads = []
    for h in stats:
        l, n = h['layer'], h['head']
        if 2 <= l <= 22 and totals[l, n] >= 512 and sequences[l, n] >= 16:
            heads.append(dict(h, id=f'L{l}H{n}', groups=['small_gate_covered'],
                              small_discovery_tokens=int(totals[l, n]),
                              small_discovery_sequences=int(sequences[l, n])))
    protocol['heads'] = heads
    protocol['predictions'] = {'small_gate_covered':
        'Majority of adequately covered heads have absolute mean delta loss <0.0001 nats/token; no broad harmful zeroing effect expected.'}
    protocol['rule'] = 'read_gate >= max(0.1, discovery head P75) AND original write_gate <=0.02'
    protocol['coverage_selection'] = dict(discovery=[0, 32], minimum_tokens=512,
                                        minimum_sequences=16, layers=[2, 22], loss_used=False)
    out.write_text(json.dumps(protocol, indent=2))
    print('selected', len(heads), [h['id'] for h in heads])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['raw', 'original', 'treatments', 'out']:
        p.add_argument(name, type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    select(a.raw, a.original, a.treatments, a.out, a.workers)

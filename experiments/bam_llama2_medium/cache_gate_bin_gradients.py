"""Reuse captured probability gradients for frozen baseline-gate bins."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np


def one(task):
    path, heads = task
    with np.load(path) as f:
        gates, reads, grads = f['write_gate'], f['read_gate'], f['gradient']
    rows = []
    for h in heads:
        g = gates[h['layer'], :, h['head']]
        r = reads[h['layer'], :, h['head']]
        v = grads[h['layer'], :, h['head']]
        selected = ((r >= h['threshold']) & (g >= h['gate_lower']) & (g < h['gate_upper']))
        step = np.minimum(.01, np.minimum(.1*g, .1*(1-g)))
        rows.append([np.sum(v[selected].astype(np.float64)*step[selected]),
                     selected.sum(), step[selected].sum(dtype=np.float64)])
    sequence = int(path.stem.split('_')[-1])
    primal = json.loads((path.parent/f'check_{sequence:03d}.json').read_text())['primal_delta']
    return sequence, np.asarray(rows), primal


def build(root, protocol_path, out, workers):
    protocol = json.loads(protocol_path.read_text())
    meta = json.loads((root/'metadata.json').read_text())
    assert meta['dtype'] == 'float32' and meta['matmul_precision'] == 'highest'
    heads = protocol['heads']
    values = np.full((128,len(heads),3), np.nan)
    primal = np.full(128, np.nan)
    tasks = [(p, heads) for p in sorted(root.glob('gradient_*.npz'))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i, a, d in pool.map(one, tasks):
            values[i] = a
            primal[i] = d
    assert np.isfinite(values[32:128]).all()
    np.savez_compressed(out, values=values, primal_delta=primal,
                        head_ids=np.array([h['id'] for h in heads]),
                        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                        cohort_sha256=meta['cohort_sha256'], model=meta['model'],
                        checkpoint=meta['checkpoint'], dtype=meta['dtype'],
                        gradient_source_runtime=meta['runtime'])
    print('GRADIENT_CACHE_READY', values.shape, out)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['root','protocol','out']:
        p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    build(a.root,a.protocol,a.out,a.workers)

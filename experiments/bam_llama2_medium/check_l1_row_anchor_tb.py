"""Verify global-L1 anchor continuity from already captured TensorBoard scalars."""
import argparse
import json
from pathlib import Path
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def inspect(logdir, layers=24, block_size=3, recent=3):
    events = EventAccumulator(str(logdir), size_guidance={'scalars': 0})
    events.Reload()
    scalars = {tag: {e.step: e.value for e in events.Scalars(tag)}
               for tag in events.Tags()['scalars']
               if tag.startswith('bam/row_anchor/') or tag == 'learning/raw_grad_norm'}
    local = [l for l in range(layers) if l % block_size != block_size - 1]
    get = lambda l, name, step: scalars[f'bam/row_anchor/layer_{l:03d}/{name}'][step]
    common = sorted(set.intersection(*(set(v) for v in scalars.values())))[-recent:]
    assert common, 'No complete common health steps'
    result = {'logdir': str(logdir), 'checked_steps': common, 'steps': {}}
    for step in common:
        anchor = get(1, 'anchor_rms', step)
        assert math.isfinite(anchor) and anchor > 0, (step, anchor)
        assert anchor == get(1, 'native_rms', step)
        for l in local:
            assert get(l, 'active', step) == (l > 1), (step, l)
            assert get(l, 'anchor_rms', step) == (anchor if l >= 1 else 0), (step, l)
            gate = get(l, 'gate_mean', step)
            assert 0 <= gate <= 1 and math.isfinite(gate), (step, l, gate)
            bins = sum(get(l, n, step) for n in ('gate_0_02','gate_02_04','gate_04_06','gate_06_08','gate_08_1'))
            assert abs(bins - 1) < 1e-5, (step, l, bins)
        result['steps'][step] = {
            'anchor_rms': anchor,
            'raw_grad_norm': scalars['learning/raw_grad_norm'][step],
            'gate_mean_by_local_layer': {l: get(l, 'gate_mean', step) for l in local},
        }
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logdir', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = json.dumps(inspect(args.logdir), indent=2)
    if args.output:
        args.output.write_text(report + '\n')
    print(report)

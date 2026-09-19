"""Fixed-window relay TB summaries; reuses the incremental scalar cache."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--steps', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('health', root /
        '.agents/skills/tpu-training/scripts/report_bam_read_health.py')
    health = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health)
    steps = [int(s) for s in args.steps.split(',')]
    names = ('scale_mean', 'saturated_fraction', 'delta_over_m',
             'anchor_m_cosine', 'mixed_over_m')
    result = {}
    for run in args.runs:
        data = health.Scalars(health.DEFAULT_LOCAL_TB_ROOT / run, steps)
        def window(tag, step):
            return [v.value for v in data.values(tag)
                    if v.step % 10 == 0 and abs(v.step - step) <= 25]
        rows = {}
        for step in steps:
            row = {}
            for label, layers in [('all', range(3, 24)), ('L3-8', range(3, 9)),
                                  ('L9-15', range(9, 16)), ('L16-23', range(16, 24))]:
                row[label] = {}
                for name in names:
                    values = [x for layer in layers for x in window(
                        f'bam/m_relay/layer_{layer:03d}/{name}', step)]
                    row[label][name] = float(np.mean(values)) if values else None
            grad = window('learning/raw_grad_norm', step)
            row['raw_grad_norm'] = float(np.mean(grad)) if grad else None
            rows[step] = row
        result[run] = rows
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

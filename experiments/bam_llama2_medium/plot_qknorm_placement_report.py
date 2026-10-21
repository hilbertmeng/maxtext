#!/usr/bin/env python3
"""Plot registered cumulative gaps and gain ratios; never recalculate metrics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FULL = 'BamMediumPropMShiftReadG050FullQKNormTruePile'
NONE = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
MUDD = 'MuddLlama2MediumPropTruePile'
REFERENCE = 'BamMHAMediumPropC256TruePile'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--experiment-label', default='norm18')
    args = parser.parse_args()
    runs = json.loads(args.report.read_text())['runs']
    assert len(runs) == 1
    report = runs[0]['pending_report']
    gaps = {item['base']: item['series'] for item in report['reports']}
    ratios = {item['comparator']: item['series'] for item in report['gain_ratios']}
    assert all(item['reference'] == REFERENCE for item in report['gain_ratios'])
    available = [(base, label, color) for base, label, color in
                 [(FULL, 'full75', 'tab:blue'), (NONE, 'no QKNorm', 'tab:orange'),
                  (MUDD, 'Mudd', 'tab:green')]
                 if base in gaps and base in ratios]
    assert available, 'No registered comparison has both gap and gain series'
    fig, axes = plt.subplots(len(available), 2, figsize=(12, 3.6*len(available)),
                             squeeze=False, constrained_layout=True)
    for row, (base, label, color) in enumerate(available):
        left, right = axes[row]
        series = gaps[base]
        left.plot([p['step']/1000 for p in series], [p['gap'] for p in series],
                  color=color, marker='o', markersize=3)
        left.set(title=f'Loss gap: {args.experiment_label} minus {label}',
                 ylabel='Loss gap (negative is better)')
        left.axhline(0, color='black', linewidth=.8, linestyle='--')
        series = ratios[base]
        right.plot([p['step']/1000 for p in series], [p['ratio'] for p in series],
                   color=color, marker='o', markersize=3)
        right.set(title=f'Gain ratio vs {label}; common reference: MHA', ylabel='Gain ratio')
        right.axhline(1, color='black', linewidth=.8, linestyle='--')
    for ax in axes.flat:
        ax.set_xlabel('Training step (thousands)')
        ax.grid(alpha=.25)
    fig.suptitle(f'MediumProp {args.experiment_label}: cumulative series through step {report["through_step"]:,}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix('.svg'))
    print(args.output)


if __name__ == '__main__':
    main()

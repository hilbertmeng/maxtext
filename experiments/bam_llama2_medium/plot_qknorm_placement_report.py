#!/usr/bin/env python3
"""Plot cumulative registry report series without recalculating gain ratios."""
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, base, label, color in [(axes[0, 0], FULL, 'full75', 'tab:blue'),
                                    (axes[0, 1], NONE, 'no QKNorm', 'tab:orange')]:
        series = gaps[base]
        ax.plot([p['step']/1000 for p in series], [p['gap'] for p in series],
                color=color, marker='o', markersize=3)
        ax.set(title=f'Loss gap: {args.experiment_label} minus {label}', ylabel='Loss gap (negative is better)')
        ax.axhline(0, color='black', linewidth=.8, linestyle='--')
    for base, label, color in [(FULL, 'vs full75', 'tab:blue'),
                               (NONE, 'vs no QKNorm', 'tab:orange')]:
        series = ratios[base]
        axes[1, 0].plot([p['step']/1000 for p in series], [p['ratio'] for p in series],
                        label=label, color=color, marker='o', markersize=3)
    axes[1, 0].axhline(1, color='black', linewidth=.8, linestyle='--')
    axes[1, 0].set(title='Gain ratio, common reference: MHA', ylabel='Gain ratio')
    axes[1, 0].legend()
    series = ratios[MUDD]
    axes[1, 1].plot([p['step']/1000 for p in series], [p['ratio'] for p in series],
                    color='tab:green', marker='o', markersize=3)
    axes[1, 1].set(title='Gain ratio vs Mudd (without QKNorm)', ylabel='Gain ratio')
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

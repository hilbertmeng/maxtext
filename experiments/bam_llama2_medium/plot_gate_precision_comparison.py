"""Compare within-precision paired interventions; never subtract across precisions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(fp32_path, bf16_path, out):
    summaries = [json.loads(p.read_text()) for p in [fp32_path, bf16_path]]
    assert [h['id'] for h in summaries[0]['heads']] == [h['id'] for h in summaries[1]['heads']]
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True)
    y = np.arange(len(summaries[0]['heads']))
    for ax, summary, precision in zip(axes, summaries, ['FP32 / highest matmul', 'BF16 / training precision']):
        for shift, offset, color, marker in [(-1., -.12, 'tab:blue', 'o'), (1., .12, 'tab:orange', 's')]:
            k = summary['shifts'].index(shift)
            means = np.array([h['mean_delta'][k] for h in summary['heads']]) * 1e6
            ci = np.array([h['ci95'][k] for h in summary['heads']]) * 1e6
            ax.errorbar(means, y + offset, xerr=[means-ci[:, 0], ci[:, 1]-means],
                        fmt=marker, color=color, markersize=4, capsize=2, label=f'alpha={shift:+g}')
        ax.axvline(0, color='black', linewidth=.8)
        ax.grid(axis='x', alpha=.2)
        ax.set_title(f"{precision}; n={summary['n_sequences']}")
        ax.set_xlabel('Paired delta loss (micro-nats/token); negative = improvement')
        ax.legend(loc='best')
    labels = [h['id'] for h in summaries[0]['heads']]
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis()
    fig.suptitle('Conditional write-gate changes; 95% sequence bootstrap intervals\nEach panel uses its own unmodified baseline and its own x-axis scale')
    fig.tight_layout(); fig.savefig(out / 'precision_comparison.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fp32', type=Path); parser.add_argument('bf16', type=Path); parser.add_argument('out', type=Path)
    args = parser.parse_args(); plot(args.fp32, args.bf16, args.out)

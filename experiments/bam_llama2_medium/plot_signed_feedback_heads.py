#!/usr/bin/env python3
"""Plot per-head negative-feedback token fractions at observed training steps."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', nargs='+', type=Path, required=True)
p.add_argument('--steps', default='20,200,1000')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
steps = [int(x) for x in a.steps.split(',')]
snapshots = {}
for path in a.inputs:
    data = json.loads(path.read_text())
    for row in data['snapshots']:
        snapshots[row['step']] = row['signed_write']
fig, axes = plt.subplots(1, len(steps), figsize=(4.1*len(steps), 6.8), layout='constrained', sharey=True, squeeze=False)
for ax, step in zip(axes[0], steps):
    values = snapshots[step]
    matrix = np.array([[values[f'bam/signed_write/layer_{layer:03}/head_{head:02}/negative_fraction'] for head in range(16)] for layer in range(1, 23)])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax.set(title=f'Step {step}', xlabel='Head', xticks=[0, 3, 7, 11, 15], yticks=np.arange(22), yticklabels=np.arange(1, 23))
axes[0, 0].set_ylabel('Layer (L0 and terminal L23 excluded)')
fig.colorbar(im, ax=list(axes[0]), shrink=.8, label='Fraction of tokens with negative feedback gate')
fig.suptitle('Signed LocalO feedback: direction varies across layers and heads')
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=170)
fig.savefig(a.output.with_suffix('.pdf'))

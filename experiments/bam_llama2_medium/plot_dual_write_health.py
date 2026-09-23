#!/usr/bin/env python3
"""Plot one recorded dual-write health snapshot (layer/head detail)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('health', type=Path)
  p.add_argument('--step', type=int)
  p.add_argument('--output', required=True, type=Path)
  a = p.parse_args()
  snapshots = json.loads(a.health.read_text())['snapshots']
  s = snapshots[-1] if a.step is None else next(s for s in snapshots if s['step'] == a.step)
  v = s['values']
  def array(metric):
    return np.array([[v[f'layer_{layer:03d}/head_{head:02d}/{metric}']
                      for head in range(16)] for layer in range(24)])
  read, main_gate, feedback = map(array, ('read_mean', 'main_mean', 'feedback_mean'))
  difference = feedback-main_gate
  corr_difference = array('read_feedback_corr')-array('read_main_corr')
  fig, axes = plt.subplots(1, 4, figsize=(14, 7), layout='constrained')
  layers = np.arange(24)
  for x, label, color in ((read, 'LocalO read', '#666666'),
                           (main_gate, 'Main write', '#1769aa'),
                           (feedback, 'LocalO feedback', '#e46b25')):
    axes[0].plot(x.mean(1), layers, label=label, color=color)
  axes[0].set(title='Mean gate openings', xlabel='Opening', ylabel='Layer', ylim=(23.5, -.5))
  axes[0].legend(fontsize=8)
  axes[0].grid(alpha=.2)
  for ax, data, title, cmap, limits in zip(axes[1:],
      (difference, corr_difference, array('feedback_norm_share')),
      ('Feedback minus main opening', 'Read–feedback corr. minus\nread–main corr.', 'LocalO cumulative write norm share'),
      ('RdBu_r', 'RdBu_r', 'viridis'),
      ((-max(np.abs(difference[1:23]).max(), .001), max(np.abs(difference[1:23]).max(), .001)),
       (-max(np.abs(corr_difference[1:23]).max(), .01), max(np.abs(corr_difference[1:23]).max(), .01)), (0, 1))):
    im = ax.imshow(data, aspect='auto', origin='upper', cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set(title=title, xlabel='Head', xticks=[0, 4, 8, 12, 15])
    fig.colorbar(im, ax=ax, fraction=.055, pad=.02, shrink=.85)
  for ax in axes:
    ax.set_yticks([0, 1, 3, 8, 12, 16, 20, 23])
    for lo, hi in ((-.5, .5), (22.5, 23.5)):
      ax.axhspan(lo, hi, facecolor='none', edgecolor='#777777', hatch='///', linewidth=0)
  fig.suptitle(f'Dual write gates — training step {s["step"]}\nHatched: L0 and final layer (structurally special)', fontsize=14)
  a.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(a.output, dpi=180)
  fig.savefig(a.output.with_suffix('.pdf'))
  plt.close(fig)


if __name__ == '__main__':
  main()

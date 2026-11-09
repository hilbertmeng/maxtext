"""Plot the archived XL 36k spike replay and its validated local checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('root', type=Path)
  p.add_argument('--output-dir', type=Path, required=True)
  args = p.parse_args()
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  from PIL import Image
  root = args.root
  paths = list((root / 'long-replay').glob('*/replay-metrics.jsonl'))
  assert len(paths) == 1, paths
  by_step = {}
  for source in list((root / 'onset').glob('*/replay-metrics.jsonl')) + paths:
    for line in source.read_text().splitlines():
      row = json.loads(line)
      if row['step'] in by_step:
        for key in ('learning/raw_grad_norm', 'learning/loss'):
          assert row[key] == by_step[row['step']][key], (row['step'], key)
      by_step[row['step']] = row
  records = [by_step[step] for step in sorted(by_step)]
  future_path = paths[0].with_name('frozen-batch-comparison.json')
  future = json.loads(future_path.read_text())
  official = json.loads((root / 'official-raw35500-36950.json').read_text())['points']
  fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
  axis = axes[0, 0]
  axis.semilogy([r['step'] for r in records], [r['learning/raw_grad_norm'] for r in records],
                color='#0072b2', lw=.85, label='Original AOT, every update')
  axis.scatter([r['step'] for r in official], [r['value'] for r in official],
               s=10, facecolors='none', edgecolors='#444', lw=.5, label='Formal TB, every 10 updates')
  axis.set_title('Exact replay: onset, peak and recurrences')
  axis.set_ylabel('Raw gradient norm'); axis.set_xlabel('Training step')
  axis.legend(fontsize=7.5)
  axis = axes[0, 1]
  for key, label, color, marker in [('actual', 'Actual evolving parameters', '#d55e00', 'o'),
      ('frozen_parameters', 'Parameters frozen before step 35636', '#0072b2', 's')]:
    axis.semilogy([r['step'] for r in future], [r[key]['learning/raw_grad_norm'] for r in future],
                  marker, color=color, ms=4, label=label)
  axis.set_title('Same future batches: change parameters only')
  axis.set_ylabel('Raw gradient norm'); axis.set_xlabel('Training step')
  axis.legend(fontsize=7.5)
  source_rows, local_checks = {}, {}
  for name, color, step, stem in [('onset', '#0072b2', 35636, 'onset'),
                                ('peak', '#d55e00', 35676, 'peak'),
                                ('tail', '#009e73', 36920, 'tail')]:
    path = root / f'layer-cotangents-{name}.json'
    if path.exists():
      rows = json.loads(path.read_text()); source_rows[name] = rows
      values = np.asarray([r['residual_cotangent_rms'] for r in rows])
      axes[1, 0].semilogy([r['layer'] for r in rows], values / values[-1],
                          '.-', color=color, lw=1, ms=3, label=f'{step}')
    for path in sorted((root / f'mlp-detail-{stem}').glob('layer-*/comparison.json')):
      data = json.loads(path.read_text())
      checks = [r for r in data['arms'] if r['pallas'] and r['block_probes']]
      assert len(checks) == 1
      check = checks[0]; local_checks[str(path)] = check
      gains = check['block_vjp_gains']; layer = data['layer']
      axes[1, 1].semilogy(layer, gains['mlp'], 's', color=color, ms=5)
      axes[1, 1].semilogy(layer, gains['memory_mlp'], 'o', markerfacecolor='none', color=color, ms=4)
  axes[1, 0].set_title('Residual gradient: cumulative backward growth')
  axes[1, 0].set_ylabel('RMS / L27 entry RMS'); axes[1, 0].set_xlabel('Layer index (backward travels left)')
  axes[1, 0].legend(title='Selected-token loss at step', fontsize=7.5, title_fontsize=7.5)
  axes[1, 1].set_title('Frozen layer checks: ordinary vs memory MLP')
  axes[1, 1].set_ylabel('Directional VJP norm gain'); axes[1, 1].set_xlabel('Layer index; points are tested layers only')
  axes[1, 1].plot([], [], 's', color='#555', label='Ordinary SwiGLU')
  axes[1, 1].plot([], [], 'o', markerfacecolor='none', color='#555', label='Dedicated memory SwiGLU')
  axes[1, 1].legend(fontsize=7.5)
  for axis in axes.flat:
    axis.axhline(1, color='.5', lw=.6, ls='--'); axis.grid(alpha=.18)
  for axis in axes[1]:
    axis.set_xlim(-.5, 27.5); axis.set_xticks(range(0, 28, 3))
  fig.suptitle('XL V5 full remat: source of the 35.6k–36.9k gradient wave', fontsize=12)
  fig.text(.04, .015, 'Top: global-batch mean gradients. Bottom: isolated-token cotangents and frozen local VJPs.\n'
           'Different token losses are normalized independently; local gains are not whole-network operator norms.',
           fontsize=7.5, color='#555')
  fig.tight_layout(rect=(0, .055, 1, .96))
  stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
  args.output_dir.mkdir(parents=True, exist_ok=True)
  path = args.output_dir / f'xlprop-v5-36k-spike-diagnosis-{stamp}.png'
  fig.savefig(path, dpi=95)
  plt.close(fig)
  with Image.open(path) as image:
    image.convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                dither=Image.Dither.NONE).save(path, optimize=True)
  path.with_suffix('.json').write_text(json.dumps(dict(root=str(root), replay=str(paths[0]),
      future_batch_checks=future, layer_rows=source_rows, local_vjps=local_checks), indent=2) + '\n')
  print(path)


if __name__ == '__main__':
  main()

#!/usr/bin/env python3
"""Plot recorded gradient norms using the incremental TensorBoard health cache."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from plot_training_report import break_missing_windows
from report_bam_read_health import DEFAULT_LOCAL_TB_ROOT, Scalars


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('runs', nargs='+')
  parser.add_argument('--through-step', type=int, required=True)
  parser.add_argument('--start-step', type=int, default=600)
  parser.add_argument('--event-dir', nargs=2, action='append', default=[],
                      metavar=('RUN', 'DIRECTORY'))
  parser.add_argument('--label', nargs=2, action='append', default=[], metavar=('RUN', 'LABEL'))
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--prefix', default='raw-grad-norm')
  parser.add_argument('--dpi', type=int, default=100)
  parser.add_argument('--title', default='Raw gradient norm')
  parser.add_argument('--include-clipped', action='store_true',
                      help='Also plot recorded learning/grad_norm after gradient clipping')
  args = parser.parse_args()
  if (Path(args.prefix).name != args.prefix or args.start_step < 0
      or args.through_step < args.start_step or args.dpi <= 0):
    parser.error('Invalid filename prefix, step bounds or DPI')

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from PIL import Image

  directories, labels = dict(args.event_dir), dict(args.label)
  fig, axis = plt.subplots(figsize=(12.8, 4.8))
  series, warnings = [], []
  tags = ['learning/raw_grad_norm']
  if args.include_clipped:
    tags.append('learning/grad_norm')
  for run in dict.fromkeys(args.runs):
    directory = directories.get(run, str(DEFAULT_LOCAL_TB_ROOT / run))
    scalars = Scalars(directory, [args.through_step])
    sources = []
    if not str(directory).startswith('gs://'):
      sources = [dict(path=str(p.resolve()), bytes=p.stat().st_size,
                      mtime_ns=p.stat().st_mtime_ns)
                 for p in sorted(Path(directory).glob('events.out.tfevents.*'))]
    for tag in tags:
      points = [p for p in scalars.values(tag)
                if args.start_step <= p.step <= args.through_step]
      if not points:
        parser.error(f'No {tag} samples in the selected interval for {run}')
      invalid = [p.step for p in points if not math.isfinite(p.value) or p.value <= 0]
      if invalid:
        warnings.append(f'{run} {tag}: nonpositive/nonfinite samples cannot appear on a log axis: {invalid}')
      xs, ys = break_missing_windows(
          [p.step for p in points],
          [p.value if math.isfinite(p.value) and p.value > 0 else math.nan for p in points])
      metric = 'raw_grad_norm' if tag.endswith('/raw_grad_norm') else 'clipped_grad_norm'
      axis.plot(xs, ys, linewidth=1.1, linestyle='-' if metric == 'raw_grad_norm' else '--',
                label=f'{labels.get(run, run)}: {metric}')
      series.append(dict(run=run, tag=tag, event_dir=str(directory), sources=sources,
                         through_step=points[-1].step,
                         points=[dict(step=p.step, value=p.value if math.isfinite(p.value) else None)
                                 for p in points]))

  axis.set_yscale('log')
  axis.set_xlabel('Training step')
  axis.set_ylabel('Gradient norm (log scale)')
  axis.set_xlim(args.start_step, args.through_step)
  axis.grid(True, alpha=.22)
  axis.legend(fontsize=9)
  axis.set_title(args.title)
  fig.text(.035, .015, 'Recorded TensorBoard samples; no plotting-time averaging, smoothing or clipping.', fontsize=8)
  fig.tight_layout(rect=(0, .045, 1, 1))
  args.output_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
  path = args.output_dir / f'{args.prefix}-{stamp}.png'
  fig.savefig(path, dpi=args.dpi)
  plt.close(fig)
  with Image.open(path) as source:
    size = source.size
    compact = source.convert('RGB').quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
  compact.save(path, optimize=True, compress_level=9)
  sidecar = path.with_suffix('.json')
  sidecar.write_text(json.dumps(dict(
      figure=str(path), generated_utc=stamp, tags=tags, start_step=args.start_step,
      requested_through_step=args.through_step, log_y=True, series=series, warnings=warnings,
      image=dict(width=size[0], height=size[1], dpi=args.dpi, bytes=path.stat().st_size)),
      indent=2, allow_nan=False) + '\n')
  print(json.dumps(dict(figure=str(path), provenance=str(sidecar),
                        shown=[dict(run=s['run'], tag=s['tag'], through_step=s['through_step']) for s in series],
                        warnings=warnings), indent=2))


if __name__ == '__main__':
  main()

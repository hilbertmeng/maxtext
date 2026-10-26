#!/usr/bin/env python3
"""Overlay selected RUNs in one cumulative gap/ratio figure from registry JSON.

All selected RUNs/baselines share the two panels by default. Values are rendered
from the registry's series, never recalculated here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def events(payload):
  """Accept report-all, pending-report, and loss-report JSON."""
  if isinstance(payload, list):
    return [{'reports': payload, 'gain_ratios': []}]
  if 'runs' in payload:
    return [row['pending_report'] for row in payload['runs'] if row.get('pending_report')]
  if 'reports' in payload:
    return [payload]
  raise ValueError('Expected registry report-all, pending-report, or loss-report JSON')


def collect(payloads, start=600, runs=(), bases=()):
  gaps, ratios = {}, {}
  for payload in payloads:
    for event in events(payload):
      for kind, target in (('reports', gaps), ('gain_ratios', ratios)):
        for report in event.get(kind, []):
          run = report.get('run', event.get('run_name'))
          base = report.get('base', report.get('comparator'))
          if not run or not base:
            raise ValueError('Registry series lacks run/base identity')
          if (runs and run not in runs) or (bases and base not in bases):
            continue
          key = (run, base) if kind == 'reports' else (run, base, report['reference'])
          for point in report['series']:
            if point['step'] >= start:
              target.setdefault(key, {})[point['step']] = point
  return gaps, ratios


def finite(value):
  return isinstance(value, (int, float)) and math.isfinite(value)


def ratio_value(point):
  # Undefined denominators and missing windows remain gaps in the plotted line.
  value = point.get('ratio')
  return value if point.get('status', 'ok') == 'ok' and finite(value) else math.nan


def break_missing_windows(xs, ys):
  """Break lines across absent native windows without imputing any values."""
  if len(xs) < 3:
    return xs, ys
  stride = min(b - a for a, b in zip(xs, xs[1:]))
  plot_x, plot_y = [], []
  for index, (x, y) in enumerate(zip(xs, ys)):
    if index and x - xs[index - 1] > stride * 1.5:
      plot_x.append((x + xs[index - 1]) / 2)
      plot_y.append(math.nan)
    plot_x.append(x)
    plot_y.append(y)
  return plot_x, plot_y


def render(payloads, output_dir, *, prefix='training-progress', start=600,
           runs=(), bases=(), labels=None, title=None, notes=(), sources=(), run_colors=None):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator

  gaps, ratios = collect(payloads, start, runs, bases)
  pairs = sorted(set(gaps) | {key[:2] for key in ratios})
  if not pairs:
    raise ValueError(f'No selected cumulative points at/after step {start}')
  labels = labels or {}
  fig, axes = plt.subplots(2, 1, figsize=(12.8, 8), sharex=True)
  warnings, shown = [], []
  for index, pair in enumerate(pairs):
    run, base = pair
    color = (run_colors or {}).get(run, f'C{index % 10}')
    label = f'{labels.get(run, run)} vs {labels.get(base, base)}'
    gap = gaps.get(pair, {})
    steps = sorted(gap)
    if steps:
      gx, gy = break_missing_windows(steps, [gap[s]['gap'] if finite(gap[s].get('gap')) else math.nan
                                          for s in steps])
      axes[0].plot(gx, gy, color=color, marker='o', markersize=2.5,
                   linewidth=1.5, label=label)
    gain_horizons = {}
    own_ratios = [(key, points) for key, points in ratios.items() if key[:2] == pair]
    if not own_ratios:
      warnings.append(f'No registered gain-ratio series for {run} vs {base}')
    for ri, (key, points) in enumerate(own_ratios):
      reference = key[2]
      xs = sorted(set(steps) | set(points))
      ys = [ratio_value(points.get(s, {})) for s in xs]
      xs, ys = break_missing_windows(xs, ys)
      axes[1].plot(xs, ys, color=color, linestyle=('-', '--', ':')[ri % 3],
                   marker='o', markersize=2.5, linewidth=1.5,
                   label=f'{label}; ref={labels.get(reference, reference)}')
      valid = [s for s in sorted(points) if finite(ratio_value(points[s]))]
      gain_horizons[reference] = max(valid) if valid else None
      if any(not finite(y) for y in ys):
        warnings.append(f'Missing/undefined gain windows remain blank: {run}|{base}|{reference}')
    shown.append(dict(run=run, base=base, gap_through=max(steps) if steps else None,
                      gain_through=gain_horizons))
  for axis, center, ylabel in zip(axes, (0, 1),
      ('Loss gap: experiment - comparator\n(lower is better)',
       'Gain multiplier\n(higher is better)')):
    axis.axhline(center, color='#777', linestyle=':', linewidth=1)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=.2)
    if axis.lines and axis.get_legend_handles_labels()[1]:
      axis.legend(frameon=False, fontsize=8, loc='best')
  end = max(point['step'] for table in (gaps, ratios) for points in table.values()
            for point in points.values())
  ticks = MaxNLocator(nbins=6, integer=True).tick_values(start, max(start + 1, end))
  axes[1].set_xticks(sorted({start, end} | {int(x) for x in ticks if start < x < end}))
  axes[1].set_xlim(max(0, start - 20), end + max(20, (end - start) * .01))
  axes[1].set_xlabel('Training step')
  fig.suptitle(title or 'Cumulative loss gaps and gain multipliers', fontsize=14)
  footer = [f'Registry window means; cumulative from step {start}; no smoothing or extrapolation.',
            *notes]
  if warnings:
    footer.append('Missing/undefined ratios are blank; see provenance JSON for details.')
  bottom = min(.35, .12 + .012 * len(footer))
  fig.subplots_adjust(left=.105, right=.985, bottom=bottom, top=.90, hspace=.15)
  fig.text(.06, .025, '\n'.join(footer), fontsize=8, color='#444')
  output_dir = Path(output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
  path = output_dir / f'{prefix}-{stamp}.png'
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)
  # Keep exact native series and identities beside each unique PNG for reproducibility.
  provenance = dict(figure=str(path), generated_utc=stamp, sources=list(sources),
      plot_start=start, title=title, labels=labels, run_colors=run_colors or {}, notes=list(notes), shown=shown,
      warnings=warnings,
      reports=[dict(run=key[0], base=key[1], series=[points[s] for s in sorted(points)])
               for key, points in gaps.items()],
      gain_ratios=[dict(run=key[0], comparator=key[1], reference=key[2],
                       series=[points[s] for s in sorted(points)])
                   for key, points in ratios.items()])
  sidecar = path.with_suffix('.json')
  sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
  return dict(figure=str(path), provenance=str(sidecar), shown=shown, warnings=warnings)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('reports', nargs='+', type=Path)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--prefix', default='training-progress')
  parser.add_argument('--start-step', type=int, default=600)
  parser.add_argument('--run', action='append', default=[])
  parser.add_argument('--base', action='append', default=[])
  parser.add_argument('--label', nargs=2, action='append', default=[], metavar=('RUN', 'LABEL'))
  parser.add_argument('--color', nargs=2, action='append', default=[], metavar=('RUN', 'COLOR'))
  parser.add_argument('--title')
  parser.add_argument('--note', action='append', default=[])
  args = parser.parse_args()
  if Path(args.prefix).name != args.prefix or args.start_step < 0:
    parser.error('--prefix must be a filename stem and --start-step must be nonnegative')
  try:
    texts = [path.read_bytes() for path in args.reports]
    result = render([json.loads(text) for text in texts], args.output_dir,
        prefix=args.prefix, start=args.start_step, runs=args.run, bases=args.base,
        labels=dict(args.label), run_colors=dict(args.color), title=args.title, notes=args.note,
        sources=[dict(path=str(p.resolve()), sha256=hashlib.sha256(text).hexdigest())
                 for p, text in zip(args.reports, texts)])
  except (ValueError, KeyError, OSError) as error:
    parser.error(str(error))
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()

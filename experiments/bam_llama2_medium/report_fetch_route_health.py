"""Report per-layer-band fetch-route time series from the incremental TB cache."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                      '.claude/skills/tpu-training/scripts'))
from report_bam_read_health import Scalars, DEFAULT_LOCAL_TB_ROOT, _parse_steps, _parse_bands


METRICS = (
    'preclip_negative_fraction', 'zero_fraction', 'cross_mass_per_query',
    'cross_l2_rms_per_query', 'mix_weight_mean', 'mix_weight_rms',
    'mix_weight_negative_fraction',
)


def collect(scalars, steps, bands):
  rows = []
  for band, layers in bands:
    for step in steps:
      row = dict(step=step, band=band)
      for metric in METRICS:
        row[metric] = scalars.band_mean(
            'bam/fetch_route/layer_{layer:03d}/' + metric, step, layers)
      rows.append(row)
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('runs', nargs='+', help='Include comparison RUNs explicitly.')
  parser.add_argument('--steps', required=True)
  parser.add_argument('--bands', default='0-7,8-15,16-23')
  parser.add_argument('--tb-root', type=Path, default=DEFAULT_LOCAL_TB_ROOT)
  parser.add_argument('--json', action='store_true')
  args = parser.parse_args()
  steps, bands = _parse_steps(args.steps), _parse_bands(args.bands)
  result = {run: collect(Scalars(args.tb_root / run, steps), steps, bands)
            for run in args.runs}
  if args.json:
    print(json.dumps(result, indent=2))
    return
  for run, rows in result.items():
    print('RUN=' + run)
    for band, _ in bands:
      selected = [row for row in rows if row['band'] == band]
      print('layers=' + band)
      print('step ' + ' '.join(str(row['step']) for row in selected))
      for metric in METRICS:
        values = [row[metric] for row in selected]
        print(metric + ' ' + ' '.join(
            '--' if value is None else f'{value:.5g}' for value in values))


if __name__ == '__main__':
  main()

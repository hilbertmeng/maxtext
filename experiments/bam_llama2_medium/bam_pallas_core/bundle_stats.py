#!/usr/bin/env python3
"""Summarize libtpu LLO schedule analyses of Pallas kernels from --xla_jf_dump_to dumps.

usage: bundle_stats.py DUMP_DIR [DUMP_DIR ...] [--show N]
Each DUMP_DIR contains llo/; reports, per Pallas kernel program, scheduled bundle counts at the
pre-RA / no-spill / final stages and the head of the final schedule analysis.
"""
import re
import sys
from pathlib import Path

STAGES = ['schedule-analysis_packed-bundles-pre-ra', 'schedule-analysis_packed-bundles-no-spills-fills',
          'schedule-analysis_packed-bundles-post-ra', 'schedule-analysis_final_bundles']


def bundles(path):
  m = re.search(r'total scheduled bundles:\s+(\d+)', path.read_text(errors='replace'))
  return int(m.group(1)) if m else None


def main():
  show = 0
  dirs = []
  args = sys.argv[1:]
  while args:
    a = args.pop(0)
    if a == '--show':
      show = int(args.pop(0))
    else:
      dirs.append(Path(a))
  for d in dirs:
    llo = d / 'llo'
    programs = sorted({re.sub(r'^\d+-', '', p.name).rsplit('-', 1)[0].split('-')[0]
                       for p in llo.iterdir() if 'bam_core' in p.name})
    for prog in programs:
      row = {}
      final = None
      for stage in STAGES:
        hits = [p for p in llo.iterdir() if f'{prog}-' in p.name and p.name.endswith(stage + '.txt')]
        if hits:
          row[stage.split('_', 1)[1]] = bundles(hits[-1])
          if stage == STAGES[-1]:
            final = hits[-1]
      print(f'{d.name} {prog} ' + ' '.join(f'{k}={v}' for k, v in row.items()))
      if final is not None and show:
        print('\n'.join(final.read_text(errors='replace').splitlines()[:show]))


def utilization(path):
  """Per-unit totals from a final_hlo-static-per-bundle-utilization file."""
  lines = Path(path).read_text(errors='replace').splitlines()
  units = lines[1].replace(',', ' ').split()
  cap = [int(x) for x in lines[2].split()]
  rows = [[int(x) for x in l.split()] for l in lines[4:] if l.strip() and l.split()[0].isdigit()]
  tot = [sum(r[i] for r in rows) for i in range(len(units))]
  sat = [sum(1 for r in rows if cap[i] and r[i] >= cap[i]) for i in range(len(units))]
  return units, cap, len(rows), tot, sat


def utilization_report(root):
  for d in sorted(Path(root).iterdir()):
    for f in sorted((d / 'llo').glob('*bam_*final_hlo-static-per-bundle-utilization.txt')):
      units, cap, n, tot, sat = utilization(f)
      prog = f.name.split('-')[1]
      bound = {u: round(t / c) for u, t, c in zip(units, tot, cap) if c}
      print(f'{d.name:38s} {prog:28s} bundles={n:6d} ' + ' '.join(f'{u}={t}' for u, t in zip(units, tot)))
      print(' ' * 40 + 'unit-bound: ' + ' '.join(f'{k}={v}' for k, v in bound.items())
            + f'  VALU-saturated={sat[units.index("VALU")]}')


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == '--utilization':
    utilization_report(sys.argv[2])
  else:
    main()

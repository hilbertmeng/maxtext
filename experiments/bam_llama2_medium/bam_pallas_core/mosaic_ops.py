#!/usr/bin/env python3
"""Count op types (and result vector shapes) in a Mosaic post-apply-vector-layout dump.

usage: mosaic_ops.py DUMP_DIR [stage-suffix]   (default stage: post-apply-vector-layout-simplify)
"""
import collections
import re
import sys
from pathlib import Path


def main():
  d = Path(sys.argv[1]) / 'mosaic'
  stage = sys.argv[2] if len(sys.argv) > 2 else 'post-apply-vector-layout-simplify'
  for f in sorted(d.glob(f'*{stage}.txt')):
    ops = collections.Counter()
    for line in f.read_text(errors='replace').splitlines():
      m = re.search(r'=\s*"?([a-z_]+\.[a-z_.]+)"?', line) or re.match(r'\s*"?([a-z_]+\.[a-z_.]+)"?[ (]', line)
      if m:
        ops[m.group(1)] += 1
    print(f'== {f.name}  total={sum(ops.values())}')
    for op, n in ops.most_common(30):
      print(f'{n:8d} {op}')


if __name__ == '__main__':
  main()

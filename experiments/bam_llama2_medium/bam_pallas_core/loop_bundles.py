#!/usr/bin/env python3
"""List loops in a libtpu final_bundles dump: body bundle ranges from backward branches.

usage: loop_bundles.py FINAL_BUNDLES_FILE
Prints total static bundles and, per backward branch, the loop body length (bundles from the
branch target to the branch). Multiply each body by its known trip count for dynamic cost.
"""
import re
import sys


def main():
  lines = open(sys.argv[1], errors='replace').read().splitlines()
  bundle_re = re.compile(r'^\s*(0x[0-9a-f]+)\s')
  last = 0
  loops = []
  for line in lines:
    m = bundle_re.match(line)
    if not m:
      continue
    b = int(m.group(1), 16)
    last = max(last, b)
    for t in re.finditer(r'sbr\.rel[^}]*?target bundleno = (\d+)', line):
      target = int(t.group(1))
      if target <= b:
        loops.append((target, b))
  print(f'static bundles: {last + 1}')
  for target, b in sorted(loops):
    print(f'loop body bundles {target:6d}..{b:6d} = {b - target + 1}')


if __name__ == '__main__':
  main()

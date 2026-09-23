#!/usr/bin/env python3
"""Reject a launch if effective experiment attributes differ from its Git runtime."""
import argparse
from pathlib import Path
import runpy
import subprocess


def attributes(cls):
  return {name: getattr(cls, name) for name in dir(cls) if not name.startswith('__')}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('commit')
  parser.add_argument('experiments', nargs='+')
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  source = subprocess.check_output(
      ['git', 'show', f'{args.commit}:MaxText/exp.py'], cwd=root, text=True)
  sealed = {}
  exec(compile(source, f'git:{args.commit}:MaxText/exp.py', 'exec'), sealed)
  working = runpy.run_path(str(root / 'MaxText/exp.py'))
  failed = False
  missing = object()
  for name in args.experiments:
    if name not in sealed or name not in working:
      print(f'MISSING_CONFIG {name}'); failed = True; continue
    left, right = attributes(working[name]), attributes(sealed[name])
    for key in sorted(left.keys() | right.keys()):
      if left.get(key, missing) != right.get(key, missing):
        print(f'MISMATCH {name}.{key}: worktree={left.get(key)!r} runtime={right.get(key)!r}')
        failed = True
  if failed:
    raise SystemExit(1)
  print(f'SEALED_CONFIG_OK {args.commit}: {", ".join(args.experiments)}')


if __name__ == '__main__':
  main()

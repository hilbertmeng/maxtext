"""Check selected registered RUNs against the primary exp.py before closeout.

Usage: python audit_exp_ledger.py REGISTRY_SNAPSHOT.json RUN [RUN ...]
The snapshot is a list of registry objects (run_name/registry_name, code_commit,
declared_status). This read-only check never merges architecture changes or
overwrites user-edited configurations.
"""
import argparse
import ast
import json
from pathlib import Path
import re


def audit(source, records, names):
  lines = source.splitlines()
  classes = {n.name:n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)}
  registry = {r.get('registry_name') or r.get('run_name'):r for r in records}
  errors = []
  for name in names:
    if name not in registry:
      errors.append(f'{name}: registry entry missing')
      continue
    record = registry[name]
    config_name = record.get('exp_class') or name
    if config_name not in classes:
      errors.append(f'{name}: primary exp.py configuration missing')
      continue
    node = classes[config_name]
    text = '\n'.join(lines[node.lineno-1:node.end_lineno])
    commit = record.get('code_commit')
    if commit and not re.search(r'\b'+re.escape(commit[:7])+r'\b', text):
      errors.append(f'{name}: registered runtime {commit[:7]} absent from class record')
    if record.get('declared_status') in ('stopped','complete','paused'):
      if not re.search(r'\b(stopped|finished|completed|paused|complete)\b',text,re.I):
        errors.append(f'{name}: terminal/pause annotation missing')
      if not re.search(r'dloss|gap|\bvs\b|reproduc|instab|unstable|failed|matched|exact',text,re.I):
        errors.append(f'{name}: experimental outcome annotation missing')
  return errors


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('registry_json',type=Path)
  p.add_argument('runs',nargs='+')
  p.add_argument('--exp',type=Path,default=Path(__file__).resolve().parents[2]/'MaxText/exp.py')
  a = p.parse_args()
  errors = audit(a.exp.read_text(),json.loads(a.registry_json.read_text()),a.runs)
  for error in errors:
    print(error)
  if errors:
    raise SystemExit(1)
  print(f'PRIMARY_LEDGER_OK {len(a.runs)} RUNs: {a.exp}')


if __name__ == '__main__':
  main()

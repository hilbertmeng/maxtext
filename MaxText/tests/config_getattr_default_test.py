"""Lint: a non-None/False getattr default on the config is dead unless base.yml declares the key.

pyconfig.HyperParameters returns None for every key a class does not declare, so
getattr(cfg, 'key', default) never yields `default`. Declare the key in base.yml (the effective
default of every class) or use a None/False default and handle None explicitly.
"""
import ast
import glob
import os
import unittest

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CONFIG_NAMES = {'cfg', 'config', 'self.config', 'self.cfg', 'self._config'}


def _dead_defaults():
  with open(os.path.join(ROOT, 'configs', 'base.yml')) as f:
    declared = set(yaml.safe_load(f))
  found = []
  for path in glob.glob(os.path.join(ROOT, '**', '*.py'), recursive=True):
    if os.sep + 'tests' + os.sep in path:
      continue
    with open(path) as f:
      tree = ast.parse(f.read(), path)
    for node in ast.walk(tree):
      if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
              and len(node.args) == 3 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
        continue
      if ast.unparse(node.args[0]) not in CONFIG_NAMES:
        continue
      default = node.args[2]
      if isinstance(default, ast.Constant) and default.value in (None, False):
        continue
      if node.args[1].value not in declared:
        found.append(f'{os.path.relpath(path, ROOT)}:{node.lineno} {ast.unparse(node)}')
  return found


class ConfigGetattrDefaultTest(unittest.TestCase):

  def test_no_dead_getattr_defaults(self):
    self.assertEqual(_dead_defaults(), [])


if __name__ == '__main__':
  unittest.main()

"""Shape-matched synthetic CPU statistics benchmark; does not restore a model."""
import argparse
import ast
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--wait-pid', type=int)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  while args.wait_pid and Path(f'/proc/{args.wait_pid}').exists():
    time.sleep(5)
  source = Path(__file__).with_name('xl_qkr4_basis_probe.py')
  tree = ast.parse(source.read_text())
  namespace = {'np': np}
  exec(compile(ast.Module(body=[node for node in tree.body
      if isinstance(node, ast.FunctionDef) and node.name in ('unit', 'stats')],
      type_ignores=[]), str(source), 'exec'), namespace)
  rng = np.random.default_rng(9876)
  raw = {f'L{layer:02d}_{arm}_{side}_raw': rng.normal(size=(1, 2048, 4, dim)).astype(np.float32)
         for layer in range(24) for arm in ('q','k') for side,dim in (('row',64),('col',32))}
  mask = np.ones((1,2048),bool)
  namespace['stats'](raw, mask, layers=(0,), sides=('row',))
  records = []
  reference = None
  for workers in (1, 4, 8, 8, 4, 1):
    start = time.perf_counter()
    result = namespace['stats'](raw, mask, workers=workers)
    elapsed = time.perf_counter()-start
    if reference is None:
      reference = result
    else:
      assert reference.keys() == result.keys()
      for key in reference:
        np.testing.assert_array_equal(reference[key], result[key], err_msg=key)
    records.append({'workers':workers,'seconds':elapsed})
    print(json.dumps(records[-1]),flush=True)
  payload = {'input':'synthetic normal, actual XL basis shapes; CPU stats only',
             'logical_cpus':os.cpu_count(),'records':records,'exact_equal':True}
  args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
  main()

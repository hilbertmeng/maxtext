"""CPU geometry tests without importing the model or a TPU runtime."""
import ast
from pathlib import Path
import unittest
import numpy as np

source = Path(__file__).with_name('xl_qkr4_basis_probe.py')
tree = ast.parse(source.read_text())
namespace = {'np': np}
exec(compile(ast.Module(body=[node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in ('unit', 'stats')],
    type_ignores=[]), str(source), 'exec'), namespace)


class BasisGeometryTest(unittest.TestCase):
  def test_parallel_matches_serial(self):
    rng = np.random.default_rng(3)
    raw = {f'L{layer:02d}_{arm}_{side}_raw': rng.normal(size=(1, 12, 4, 16))
           for layer in range(3) for arm in ('q', 'k') for side in ('row', 'col')}
    mask = np.ones((1, 12), bool)
    serial = namespace['stats'](raw, mask, layers=range(3))
    parallel = namespace['stats'](raw, mask, workers=3, layers=range(3))
    self.assertEqual(serial.keys(), parallel.keys())
    for key in serial:
      np.testing.assert_array_equal(serial[key], parallel[key], err_msg=key)

  def measure(self, q, k):
    raw = {f'L{layer:02d}_{arm}_{side}_raw': q if arm == 'q' else k
           for layer in range(24) for arm in ('q', 'k') for side in ('row', 'col')}
    return namespace['stats'](raw, np.ones(q.shape[:2], bool))

  def test_sign_flip(self):
    q = np.random.default_rng(1).normal(size=(1, 6, 4, 12))
    result = self.measure(q, -q)
    np.testing.assert_allclose(np.diag(result['L01_row_raw_uncentered_cos'][0,:4,4:]), -1)
    np.testing.assert_allclose(result['L01_row_q_in_k_span'], 1)
    np.testing.assert_allclose(result['L01_row_raw_uncentered_rank_energy'][0,3], 1)

  def test_distinct_orthogonal_subspaces(self):
    rng = np.random.default_rng(2)
    q = np.zeros((1, 6, 4, 8)); k = q.copy()
    for t in range(6):
      q[0,t,:,:4] = np.linalg.qr(rng.normal(size=(4,4)))[0]
      k[0,t,:,4:] = np.linalg.qr(rng.normal(size=(4,4)))[0]
    result = self.measure(q,k)
    np.testing.assert_allclose(result['L01_row_q_in_k_span'], 0, atol=1e-12)
    np.testing.assert_allclose(result['L01_row_raw_uncentered_rank_energy'][0,3], .5)


if __name__ == '__main__':
  unittest.main()

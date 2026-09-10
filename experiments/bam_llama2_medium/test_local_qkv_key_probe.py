"""Small synthetic checks of similarity semantics, independent of checkpoints."""
import unittest
import numpy as np
from local_qkv_key_probe import stats


class SimilarityTest(unittest.TestCase):
  def test_opposite_keys_shared_subspace_and_signed_mix(self):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 5, 1, 4)).astype(np.float32)
    raw = {}
    for l in range(24):
      for side in ('row', 'col'):
        for arm in ['q', 'k'] + (['v'] if l % 3 != 2 else []):
          v = x if arm == 'q' else -x if arm == 'k' else np.concatenate((x, -x), -2)
          for stage in ('raw', 'key'):
            raw[f'L{l:02d}_{arm}_{side}_{stage}'] = v
          h = np.ones((1, 5, 2, v.shape[-2]), np.float32)
          if arm == 'k':
            h *= -1
          if arm == 'v':
            h[..., 1] = 0
          raw[f'L{l:02d}_{arm}_{side}_mix'] = h
    result = stats(raw, np.ones((1, 5), bool))
    self.assertAlmostEqual(result['L01_row_key_uncentered_cos'][0,0,1], -1, places=6)
    self.assertAlmostEqual(result['L01_row_key_uncentered_rank_energy'][0,0], 1, places=6)
    self.assertAlmostEqual(result['L01_row_effective_qk_cos'][0], 1, places=6)


if __name__ == '__main__':
  unittest.main()

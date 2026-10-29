"""Invariants that matter for the paired gamma projection diagnostic."""
import unittest

import numpy as np

from dynamic_o_gamma_analysis import canonical_gamma, coarsen, summarize
from probe_dynamic_o_gamma import load_cohort, paired_statistics


class GammaAnalysisTest(unittest.TestCase):
  def test_projection_preserves_mean_or_signed_energy(self):
    x = np.array([[.1, .3], [-.1, -.3], [-1., 1.]])
    mean = coarsen(x, 'head_mean')
    rms = coarsen(x, 'head_signed_rms')
    np.testing.assert_allclose(mean.mean(-1), x.mean(-1))
    np.testing.assert_allclose(np.sum(rms ** 2, -1), np.sum(x ** 2, -1))
    self.assertTrue(np.all(rms[1] < 0))
    np.testing.assert_allclose(mean.std(-1), 0.)
    np.testing.assert_allclose(rms.std(-1), 0.)
    np.testing.assert_array_equal(coarsen(x, 'original'), x)

  def test_scan_axis_and_tail_order(self):
    x = np.broadcast_to((np.arange(32)[:, None, None]
                         + 100 * np.arange(31)[None, :, None]), (32, 31, 64))
    tail = np.full((32, 64), 9999)
    g = canonical_gamma({('params', 'layers', 'gamma'): x,
                         ('params', 'final_local_layer', 'gamma'): tail})
    self.assertEqual(g.shape, (32, 32, 64))
    self.assertEqual(g[7, 11, 0], 711)
    np.testing.assert_array_equal(g[31], tail)

  def test_update_decomposition_and_uniform_initial_state(self):
    g = np.full((32, 32, 64), .2)
    z = summarize(g)
    self.assertAlmostEqual(z['within_head_cv']['p50'], 0.)
    self.assertAlmostEqual(z['within_head_energy'], 0.)
    rng = np.random.default_rng(19)
    g += rng.normal(0, .03, g.shape)
    z = summarize(g)
    self.assertAlmostEqual(z['within_head_energy'] + z['head_mean_shift_energy'], z['update_energy'])
    self.assertGreater(z['within_head_fraction_of_update'], .9)

  def test_cohort_supervises_pythia_eos_and_paired_zero(self):
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
      p = Path(td) / 'cohort.npz'
      tokens = np.ones((4, 2048), np.int32)
      tokens[:, 12] = 0
      np.savez(p, inputs=tokens, targets=tokens, inputs_position=np.tile(np.arange(2048), (4, 1)),
               inputs_segmentation=np.ones_like(tokens), targets_segmentation=(tokens != 0))
      batch, proof = load_cohort(p, 4)
      self.assertTrue(np.all(batch['targets_segmentation'][:, 12] == 1))
      self.assertEqual(proof['count'], 4)
    s = paired_statistics(np.zeros(128))
    self.assertEqual(s['bootstrap95'], [0., 0.])
    self.assertEqual(s['mean'], 0.)


if __name__ == '__main__':
  unittest.main()

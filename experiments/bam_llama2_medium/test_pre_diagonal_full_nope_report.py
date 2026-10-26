"""Check paired incremental-gain orientation, exact windows and undefined cases."""
import unittest
from report_pre_diagonal_full_nope import paired_implementation


class PairedNoPEReportTest(unittest.TestCase):
  def setUp(self):
    self.pair=paired_implementation()

  def test_additive_retention_is_one_and_enhancement_is_above_one(self):
    def constant(x):return {s:x for s in range(0,626,10)}
    # [combo, pre, fullNoPE, base]; pre improves by.4, NoPE by.2.
    rows=self.pair([constant(9.4),constant(9.6),constant(9.8),constant(10.)],600)
    self.assertAlmostEqual(rows[-1]['second_order_ratio'],1.)
    rows=self.pair([constant(9.3),constant(9.6),constant(9.8),constant(10.)],600)
    self.assertAlmostEqual(rows[-1]['second_order_ratio'],1.5)
    rows=self.pair([constant(9.7),constant(9.6),constant(9.8),constant(10.)],600)
    self.assertAlmostEqual(rows[-1]['second_order_ratio'],-.5)

  def test_unequal_horizons_and_missing_raw_samples_never_impute(self):
    caches=[{s:x for s in range(0,626,10)} for x in (9.4,9.6,9.8,10.)]
    caches[0]={s:v for s,v in caches[0].items() if s<=430}
    self.assertEqual([r['step'] for r in self.pair(caches)], [200,400])
    del caches[1][400]
    self.assertEqual([r['step'] for r in self.pair(caches)], [200])

  def test_nonpositive_standalone_gain_is_undefined(self):
    for nope in (10.,10.1):
      caches=[{s:x for s in range(0,626,10)} for x in (9.4,9.6,nope,10.)]
      row=self.pair(caches,600)[-1]
      self.assertIsNone(row['second_order_ratio']);self.assertNotEqual(row['status'],'ok')


if __name__=='__main__':unittest.main()

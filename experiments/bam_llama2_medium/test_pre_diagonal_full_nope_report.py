"""Check paired incremental-gain orientation, exact windows and undefined cases."""
import unittest
from report_pre_diagonal_full_nope import paired_implementation, paired_metrics


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

  def test_symmetric_ratio_is_invariant_when_improvements_are_exchanged(self):
    def result(combo,pre,nope,base):
      caches=[{s:x for s in range(0,626,10)} for x in (combo,pre,nope,base)]
      return paired_metrics(self.pair(caches,600)[-1])
    a=result(9.988,9.990,9.995,10.)
    b=result(9.988,9.995,9.990,10.)
    self.assertAlmostEqual(a['symmetric_additivity_ratio'],.8)
    self.assertAlmostEqual(b['symmetric_additivity_ratio'],.8)
    self.assertAlmostEqual(a['conditional_retention_nope'],.4)
    self.assertAlmostEqual(b['conditional_retention_nope'],.7)
    for combo,expected in ((9.985,1.),(9.982,1.2),(10.003,-.2)):
      self.assertAlmostEqual(result(combo,9.990,9.995,10.)['symmetric_additivity_ratio'],expected)

  def test_symmetric_metric_uses_total_standalone_gain_denominator(self):
    def result(pre,nope):
      caches=[{s:x for s in range(0,626,10)} for x in (9.98,pre,nope,10.)]
      return paired_metrics(self.pair(caches,600)[-1])
    # Individual NoPE gain can be undefined while the symmetric total remains valid.
    self.assertAlmostEqual(result(9.99,10.)['symmetric_additivity_ratio'],2.)
    self.assertIsNone(result(9.99,10.)['conditional_retention_nope'])
    self.assertIsNone(result(9.99,10.01)['symmetric_additivity_ratio'])

  def test_nonpositive_standalone_gain_is_undefined(self):
    for nope in (10.,10.1):
      caches=[{s:x for s in range(0,626,10)} for x in (9.4,9.6,nope,10.)]
      row=self.pair(caches,600)[-1]
      self.assertIsNone(row['second_order_ratio']);self.assertNotEqual(row['status'],'ok')


if __name__=='__main__':unittest.main()

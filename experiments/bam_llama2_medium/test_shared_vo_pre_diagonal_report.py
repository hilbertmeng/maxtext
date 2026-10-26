"""Discriminate real transfer from ordinary-gain denominator differences."""
import unittest
from report_shared_vo_pre_diagonal import add_transfer_metrics
from report_pre_diagonal_full_nope import paired_metrics


def metrics(a,b,c,d):
  raw=dict(step=1000,samples=5,mha_qknorm_gain=a-b,old_bam_gain=a-c,
           new_bam_gain=b-d,interaction_gain=(b-d)-(a-c),
           second_order_ratio=(b-d)/(a-c) if a!=c else None,status='ok')
  return add_transfer_metrics(dict(series=[paired_metrics(raw)]))['series'][0]


class SharedVOPreReportTest(unittest.TestCase):
  def test_equal_absolute_gains_are_additive_despite_different_mha_ratios(self):
    r=metrics(3.,2.9,2.98,2.88)
    self.assertAlmostEqual(r['shared_over_separate_improvement'],1.)
    self.assertAlmostEqual(r['symmetric_additivity_ratio'],1.)
    # Same -.02 gain looks different with the two MHA-relative denominators.
    self.assertNotEqual((3.2-2.98)/(3.2-3.),(3.2-2.88)/(3.2-2.9))

  def test_opposite_orientations_of_same_interaction(self):
    r=metrics(3.,2.9,2.97,2.88)
    self.assertAlmostEqual(r['shared_over_separate_improvement'],1.5)
    self.assertLess(r['symmetric_additivity_ratio'],1.)
    r=metrics(3.,2.9,2.98,2.87)
    self.assertLess(r['shared_over_separate_improvement'],1.)
    self.assertGreater(r['symmetric_additivity_ratio'],1.)

  def test_undefined_denominator_and_reversal_stay_visible(self):
    self.assertIsNone(metrics(3.,2.9,2.98,2.9)['shared_over_separate_improvement'])
    r=metrics(3.,2.9,3.01,2.88)
    self.assertLess(r['shared_over_separate_improvement'],0.)
    self.assertLess(r['shared_minus_separate_improvement'],0.)


if __name__=='__main__':unittest.main()

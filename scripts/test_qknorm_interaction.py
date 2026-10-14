import unittest
from report_qknorm_interaction import interaction_series
class InteractionTest(unittest.TestCase):
 def test_identical_improvement_and_synergy(self):
  points=range(0,240,10)
  for bq,mq,expected in [(2.8,2.9,1.),(2.75,2.9,1.5)]:
   c=[dict.fromkeys(points,v) for v in [bq,mq,2.9,3.0]];r=interaction_series(c)[0];self.assertAlmostEqual(r['second_order_ratio'],expected);self.assertAlmostEqual(r['interaction_gain'],r['bam_qknorm_gain']-r['mha_qknorm_gain'])
 def test_complete_windows_and_nonpositive_denominator(self):
  c=[dict.fromkeys(range(0,240,10),v) for v in [2.8,2.9,3.,3.]];self.assertIsNone(interaction_series(c)[0]['second_order_ratio']);del c[0][200];self.assertEqual(interaction_series(c),[])
if __name__=='__main__':unittest.main()

import unittest
from report_mshift_cross_scale import aligned_series,multiplier
class CrossScaleTest(unittest.TestCase):
 def caches(self,center,gain,base):
  points=range(center-20,center+21,10)
  return [{s:2-gain for s in points},{s:2-base for s in points},{s:2 for s in points}]
 def test_same_fraction_different_steps_and_absolute_loss_scales(self):
  med=self.caches(2700,.24,.20);xl=self.caches(10000,.12,.10)
  rows=aligned_series(med,xl,through_xl_step=10000)
  v=next(v for v in rows if v['percent']==20)
  self.assertEqual((v['medium_step'],v['xl_step']),(2700,10000))
  self.assertAlmostEqual(v['medium']['ratio'],1.2);self.assertAlmostEqual(v['xl']['ratio'],1.2)
 def test_incomplete_raw_window_is_not_interpolated(self):
  c=self.caches(2700,.24,.20);del c[0][2700]
  self.assertEqual(multiplier(c,2700)['status'],'incomplete_window')
 def test_nonpositive_parent_gain_is_undefined(self):
  self.assertIsNone(multiplier(self.caches(2700,.1,0),2700)['ratio'])
if __name__=='__main__':unittest.main()

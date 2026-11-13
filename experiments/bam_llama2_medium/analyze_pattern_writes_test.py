import unittest
import numpy as np
from analyze_pattern_writes import route_difference,weak_write,component_projection


class AnalysisTest(unittest.TestCase):
  def test_empty_routes_excluded_and_disjoint_routes(self):
    a=np.array([[0.,0.],[1.,0.]]);b=np.array([[0.,0.],[0.,1.]])
    d=route_difference(a,b)
    np.testing.assert_array_equal(d['valid'],[False,True])
    self.assertAlmostEqual(d['tv'][1],1.)
    self.assertAlmostEqual(d['js'][1],np.log(2))

  def test_weak_space_includes_null_and_excludes_strong(self):
    m=np.array([[10.,0.,0.],[0.,.01,0.]])
    weak=np.array([[0.,0.,1.],[0.,0.,0.]])
    strong=np.array([[1.,0.,0.],[0.,0.,0.]])
    self.assertEqual(weak_write(m,weak)['rank99'],1)
    self.assertAlmostEqual(weak_write(m,weak)['weak_content_fraction'],1.)
    self.assertAlmostEqual(weak_write(m,strong)['weak_content_fraction'],0.)

  def test_canceling_components_can_exceed_one(self):
    total=np.array([1.,0.]);a=np.array([2.,1.]);b=total-a
    self.assertEqual(component_projection(a,total),2.)
    self.assertEqual(component_projection(b,total),-1.)

  def test_zero_write_has_no_energy_fraction(self):
    self.assertIsNone(weak_write(np.eye(2),np.zeros((2,2)))['weak_content_fraction'])
    self.assertIsNone(component_projection(np.ones(2),np.zeros(2)))


if __name__=='__main__':unittest.main()

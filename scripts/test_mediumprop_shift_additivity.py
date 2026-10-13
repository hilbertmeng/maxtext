"""Joint-arm effects on identical windows, metadata and delivery persistence."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report_mediumprop_shift_additivity as registry


class AdditivityTest(unittest.TestCase):
  def caches(self, joint_gain, a_gain=.002, b_gain=.020):
    points = range(0, 436, 10)
    return tuple({s: 5.-g for s in points} for g in (joint_gain, a_gain, b_gain, 0.))

  def test_no_half_full_and_super_addition_and_arm_symmetry(self):
    for gain, expected in [(.020,0.),(.021,.5),(.022,1.),(.023,1.5)]:
      caches = self.caches(gain)
      row = registry.additivity_series(*caches,200,25,10,200)[0]
      self.assertAlmostEqual(row['retention'],expected)
      self.assertAlmostEqual(row['total_fraction'],gain/.022)
      swapped = registry.additivity_series(caches[0],caches[2],caches[1],caches[3],200,25,10,200)[0]
      self.assertEqual(row['retention'],swapped['retention'])

  def test_actual_2000_example(self):
    row = registry.additivity_series(*self.caches(.020214,.007090,.013252),200,25,10,200)[0]
    self.assertAlmostEqual(row['retention'],.006962/.007090)
    self.assertAlmostEqual(row['total_fraction'],.020214/.020342)

  def test_common_windows_and_means_before_division(self):
    points = range(0,436,10)
    base = {s: 5.+s/1000 for s in points}
    a = {s: base[s]-(.2+s/1000) for s in points}
    b = {s: base[s]-.4 for s in points}
    joint = {s: base[s]-(base[s]-a[s])-(base[s]-b[s])+.05 for s in points}
    rows = registry.additivity_series(joint,a,b,base,200,25,10,400)
    self.assertEqual([p['step'] for p in rows],[200,400])
    self.assertTrue(all(p['samples']==5 for p in rows))
    self.assertAlmostEqual(rows[0]['retention'],.875)
    self.assertAlmostEqual(rows[1]['total_fraction'],.95)
    del b[190]
    self.assertEqual([p['step'] for p in registry.additivity_series(joint,a,b,base,200,25,10,400)],[400])
    self.assertEqual(registry.additivity_series(joint,a,{},base,200,25,10),[])

  def test_zero_negative_and_numerical_parity_are_undefined(self):
    for gain,status in [(0.,'near_zero_individual_gain'),(1e-12,'near_zero_individual_gain'),
                        (-.001,'nonpositive_individual_gain')]:
      row = registry.additivity_series(*self.caches(.020,gain,.020),200,25,10,200)[0]
      self.assertIsNone(row['retention'])
      self.assertAlmostEqual(row['total_fraction'],.020/(gain+.020))
      self.assertEqual(row['status'],status)
    for gain in (-.020,-.030):
      row = registry.additivity_series(*self.caches(.020,gain,.020),200,25,10,200)[0]
      self.assertIsNone(row['total_fraction'])
    row = registry.additivity_series(*self.caches(-.002),200,25,10,200)[0]
    self.assertLess(row['retention'],0)



if __name__ == '__main__':
  unittest.main()

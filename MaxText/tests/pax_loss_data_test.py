"""Contracts that distinguish historical PaxML eval from generic LM scoring."""
import unittest
import numpy as np
from pax_loss_data import Example,make_batch,aggregate

def record(x,y=None):
 e=Example();e.features.feature['input_ids'].int64_list.value.extend(x)
 if y is not None:e.features.feature['labels'].int64_list.value.extend(y)
 return e

class PaxLossDataTest(unittest.TestCase):
 def test_shift_eos_padding(self):
  b=make_batch([record([4,0,6])],'pile',4)
  np.testing.assert_array_equal(b['inputs'],[[4,0,6,0]])
  np.testing.assert_array_equal(b['targets'],[[0,6,0,0]])
  np.testing.assert_array_equal(b['mask'],[[1,1,1,1]])
 def test_answer_mask_and_truncation(self):
  b=make_batch([record([4,5,7,0,8,9],[-100,-100,7,0,8,9])],'flan',4)
  np.testing.assert_array_equal(b['targets'],[[5,7,0,8]])
  np.testing.assert_array_equal(b['mask'],[[0,1,0,1]])
  np.testing.assert_array_equal(b['segments'],[[1,1,1,1]])
 def test_empty_answer_and_weighted_mean(self):
  s=aggregate([6,10,0],[2,10,0])
  self.assertAlmostEqual(s['token_mean'],16/12)
  self.assertAlmostEqual(s['example_mean'],4/3)
  self.assertEqual(s['zero_answer_examples'],1)
 def test_reject_already_shifted_labels(self):
  with self.assertRaises(ValueError):make_batch([record([4,5,7],[-100,7,0])],'flan',4)

if __name__=='__main__':unittest.main()

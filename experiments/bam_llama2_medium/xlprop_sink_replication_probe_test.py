import unittest
from xlprop_sink_replication_probe import select_batches
class ReplicationSelectionTest(unittest.TestCase):
 def test_replicate_inputs_without_adopting_future_parameters(self):
  original=object();future1=object();future2=object();state=object()
  c=dict(data=original,rng='r0',state=state,frozen_batches=[dict(step=35677,data=future1,rng='r1',state='future_weights1'),dict(step=35678,data=future2,rng='r2',state='future_weights2')])
  b=select_batches(c)
  self.assertEqual([x['step']for x in b],[35676,35677,35678]);self.assertIs(b[0]['data'],original);self.assertIs(b[1]['data'],future1);self.assertIs(b[2]['data'],future2)
  self.assertTrue(all('state'not in x for x in b));self.assertIs(c['state'],state)
 def test_missing_batch_blocks_replication(self):
  with self.assertRaises(AssertionError):select_batches(dict(data=object(),rng=0,frozen_batches=[dict(step=35677,data=object(),rng=1)]))
 def test_wrong_order_blocks_mislabeled_replication(self):
  with self.assertRaises(AssertionError):select_batches(dict(data=object(),rng=0,frozen_batches=[dict(step=35678,data=object(),rng=1),dict(step=35677,data=object(),rng=2)]))
if __name__=='__main__':unittest.main()

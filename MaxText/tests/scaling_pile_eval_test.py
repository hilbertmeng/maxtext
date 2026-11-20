"""Data and aggregate contracts for scaling evaluation, including short last batch."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from scaling_pile_eval import make_batch,aggregate,prepare_data
from pile_validation_data import write_record,file_integrity

class Contracts(unittest.TestCase):
  def test_verified_records_roundtrip_and_corruption(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);source=root/'data.tfrecord'
      with source.open('wb') as stream:
        write_record(stream,[8,0,9,10,11]);write_record(stream,[11,12,0])
      digest=file_integrity(source)['sha256']
      meta=prepare_data(source,root/'prepared',4,digest,2)
      self.assertEqual(meta['target_tokens_before_eos_mask'],6)
      tokens=np.load(root/'prepared/tokens.npy');lengths=np.load(root/'prepared/lengths.npy')
      np.testing.assert_array_equal(lengths,[5,3]);np.testing.assert_array_equal(tokens[1],[11,12,0,0,0])
      broken=bytearray(source.read_bytes());broken[-1]^=1;source.write_bytes(broken)
      with self.assertRaises(ValueError):prepare_data(source,root/'bad',4,file_integrity(source)['sha256'],2)
  def test_real_eos_and_padding_differ(self):
    tokens=np.array([[1,0,3,4,5],[1,2,0,0,0]],np.uint16);lengths=np.array([5,3])
    b,n=make_batch(tokens,lengths,0,4,4)
    x=np.full((4,4),2.,np.float32);x[2:]=np.nan;x[1,2:]=np.nan
    a=aggregate(x,b)
    self.assertEqual(n,2);self.assertEqual(a['count_including_eos'].sum(),6)
    self.assertEqual(a['count_excluding_eos'].sum(),4)
    self.assertEqual(a['sum_including_eos'].sum(),12)
  def test_tail_is_retained_no_duplicated_rows(self):
    b,n=make_batch(np.array([[2,3,4,0,0]],np.uint16),np.array([3]),0,4,4)
    self.assertEqual(n,1);self.assertEqual(b['targets_segmentation'].sum(),2)
    self.assertEqual(b['inputs_segmentation'][1:].sum(),0)
  def test_nonfinite_on_scored_token_rejected(self):
    b,_=make_batch(np.array([[1,2,3]],np.uint16),np.array([3]),0,1,2)
    with self.assertRaises(FloatingPointError):aggregate(np.array([[1.,np.nan]]),b)
  def test_exact_token_weighted_sum(self):
    b,_=make_batch(np.array([[1,2,3,0],[1,2,0,0]],np.uint16),np.array([4,2]),0,2,3)
    a=aggregate(np.array([[1.,2.,3.],[10.,999.,999.]],np.float32),b)
    self.assertEqual(a['sum_including_eos'].sum()/a['count_including_eos'].sum(),4.)

if __name__=='__main__':unittest.main()

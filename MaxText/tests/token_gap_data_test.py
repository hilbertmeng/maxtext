"""Critical data contracts: shifted answer masks, EOS, fixed targets and CRC failures."""
import io
import struct
import tempfile
from pathlib import Path
import unittest
import numpy as np
from token_gap_data import Example,masked_crc,record_index,read_record,feature_ints,context_window,padded_batch,answer_windows


class DataContractTest(unittest.TestCase):
  def test_shifted_answer_mask_and_eos(self):
    x=np.array([[11,12,13,14,0,0],[31,32,0,0,0,0]],np.int32)
    y=np.array([[-100,-100,-100,14,0,-100],[-100,32,-100,-100,-100,-100]],np.int32)
    b,n=padded_batch(x,y,np.array([5,3]),[0,1],4,5)
    np.testing.assert_array_equal(b['mask'][:2],[[0,0,1,1,0],[1,0,0,0,0]])
    self.assertEqual(b['targets'][0,2],14);self.assertEqual(b['targets'][0,3],0)
    self.assertEqual(b['segments'][0,3],1);self.assertEqual(b['segments'][0,4],0)
    self.assertEqual(n,2)

  def test_context_retains_identical_target_and_absolute_positions(self):
    x=np.arange(30,dtype=np.int32)
    for c in [4,8,16]:
      inputs,targets,n,score,pos=context_window(x,20,c,pad_to=16)
      self.assertEqual(targets[score],20);self.assertEqual(inputs[score],19)
      np.testing.assert_array_equal(inputs[:n],np.arange(20-c,20))
      np.testing.assert_array_equal(pos[:n],np.arange(20-c,20))
      self.assertEqual(n,c)
    with self.assertRaises(ValueError):context_window(x,0,8)
    with self.assertRaises(ValueError):context_window(x,20,16,pad_to=8)

  def test_tfrecord_wire_format_and_corruption(self):
    e=Example();e.features.feature['input_ids'].int64_list.value.extend([1,20,0])
    e.features.feature['labels'].int64_list.value.extend([-100,20,-100])
    body=e.SerializeToString();header=struct.pack('<Q',len(body))
    frame=header+struct.pack('<I',masked_crc(header))+body+struct.pack('<I',masked_crc(body))
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'data';p.write_bytes(frame*2);offsets,_=record_index(p)
      self.assertEqual(offsets.tolist(),[0,len(frame)])
      with p.open('rb') as f:
        z=read_record(f,offsets[1]);np.testing.assert_array_equal(feature_ints(z,'labels'),[-100,20,-100])
      damaged=bytearray(frame);damaged[-5]^=1
      with self.assertRaises(ValueError):read_record(io.BytesIO(damaged),0)
      p.write_bytes(frame[:-1])
      with self.assertRaises(ValueError):record_index(p)

  def test_long_answer_windows_cover_each_target_exactly_once(self):
    x=np.arange(95,dtype=np.int32);y=np.full_like(x,-100);y[22:94]=x[22:94]
    seen=[]
    for xx,yy,start in answer_windows(x,y,length=32,stride=8):
      self.assertLessEqual(len(xx),33)
      np.testing.assert_array_equal(xx,x[start:start+len(xx)])
      seen.extend((start+np.flatnonzero(yy[1:]>=0)+1).tolist())
    self.assertEqual(seen,list(range(22,94)))
    short=answer_windows(x[:30],y[:30],length=32)
    self.assertEqual(len(short),1);np.testing.assert_array_equal(short[0][0],x[:30])

  def test_fp32_ce_matches_reference_on_bf16_logits(self):
    import jax.numpy as jnp
    from token_gap_ce import cross_entropy
    logits=jnp.array([[9.,-.2,-4.],[1.,1.001,1.003]],dtype=jnp.bfloat16)
    targets=jnp.array([[1.,0.,0.],[0.,0.,1.]])
    got,_=cross_entropy(logits,targets);z=np.asarray(logits,dtype=np.float64)
    logz=np.log(np.exp(z-z.max(-1,keepdims=True)).sum(-1))+z.max(-1)
    expected=logz-z[[0,1],[0,2]]
    np.testing.assert_allclose(got,expected,atol=2e-7,rtol=2e-5)


if __name__=='__main__':unittest.main()

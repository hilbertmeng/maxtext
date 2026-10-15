import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pile_validation_data import write_record
from pile_bucket_eval import prepare_data, make_batch, summarize_xent

class BucketEvalTest(unittest.TestCase):
  def test_bucket_edges_eos_tail_and_dummy_nan(self):
    ids = np.arange(1026,dtype=np.uint16).reshape(2,513)
    ids[0,1] = 0  # real EOS in the first prediction slot
    batch,n = make_batch(ids,np.array([513,20]),0,3,512)
    self.assertEqual(n,2)
    self.assertEqual(batch['targets_segmentation'].sum(),531)
    xent = jnp.broadcast_to(jnp.arange(1,513,dtype=jnp.float32),(3,512))
    xent = jnp.where(batch['targets_segmentation']!=0,xent,jnp.nan)
    result = jax.jit(summarize_xent)(xent,batch['targets'],batch['targets_segmentation'])
    np.testing.assert_array_equal(result['count_including_eos'],[[256,256],[19,0],[0,0]])
    np.testing.assert_array_equal(result['count_excluding_eos'],[[255,256],[19,0],[0,0]])
    np.testing.assert_array_equal(result['sum_including_eos'],[[32896,98432],[190,0],[0,0]])
    self.assertEqual(result['sum_excluding_eos'][0,0],32895)
    self.assertEqual(int(result['nonfinite_tokens']),0)
    self.assertEqual(float(result['total_including_eos']),32896+98432+190)

  def test_invalid_entropy_is_not_hidden(self):
    result = summarize_xent(jnp.full((1,256),jnp.nan),jnp.ones((1,256)),jnp.ones((1,256)))
    self.assertEqual(int(result['nonfinite_tokens']),256)

  def test_actual_shell_configuration_contract_for_all_models(self):
    import os,subprocess,sys
    root=Path(__file__).resolve().parents[2]
    names=['BamMHAMediumPropC256T2048Matched','MuddLlama2MediumPropC256T2048',
           'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048']
    env=dict(os.environ,PILE_BUCKET_CONFIG_ONLY='1',PILE_BUCKET_PYTHON=sys.executable,
             JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
    for i,name in enumerate(names):
      with self.subTest(name=name):
        r=subprocess.run(['bash','scripts/run_pile_bucket_eval.sh',name,
             'gs://unused/checkpoints/13500/items','config_test_'+str(i),'gs://unused/output'],
             cwd=root,env=env,text=True,capture_output=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stdout[-3000:]+r.stderr[-3000:])
        self.assertIn('PILE_BUCKET_CONFIG_OK '+name,r.stdout)

  def test_original_record_boundaries_and_cache_integrity(self):
    with tempfile.TemporaryDirectory() as name:
      source=Path(name)/'source.tf';cache=Path(name)/'cache'
      with source.open('wb') as f:
        write_record(f,[11,0,22,33,44])
        write_record(f,[55,0,66])
      digest=hashlib.sha256(source.read_bytes()).hexdigest()
      meta=prepare_data(source,cache,4,digest,2)
      self.assertEqual(meta['target_tokens_before_eos_mask'],6)
      ids=np.load(cache/'tokens.npy');lengths=np.load(cache/'lengths.npy')
      np.testing.assert_array_equal(ids,[[11,0,22,33,44],[55,0,66,0,0]])
      np.testing.assert_array_equal(lengths,[5,3])
      ids[0,0]=999;np.save(cache/'tokens.npy',ids)
      with self.assertRaisesRegex(ValueError,'checksum'):
        prepare_data(source,cache,4,digest,2)

if __name__=='__main__':unittest.main()

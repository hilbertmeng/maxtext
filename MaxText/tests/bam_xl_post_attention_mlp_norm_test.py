"""XL transfer keeps its true single-shift lineage and parameter tree."""
import unittest
import jax
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest

PARENT='BamXLPropSharedR400GeneralReadMShiftReadG050TruePile'
EXP='BamXLPropMShiftReadG050PostAttentionWriteMLPNormTruePile'

class XLPostAttentionMlpNormTest(MLPWriteTest,unittest.TestCase):
  def test_xl_parent_contract_and_full_parameter_budget(self):
    a,b=[self.config(n) for n in (PARENT,EXP)]
    allowed={'exp_class','model_name','compare_runs','record_training_health_metrics','record_internal_nn_metrics',
      'bam_write_controls_post_residual','bam_attention_write_controls_post_residual',
      'bam_mlp_write_controls_post_residual','bam_attention_write_controls_norm'}
    changed={k for k in a.get_keys() if str(a.get_keys()[k])!=str(b.get_keys()[k])}
    self.assertTrue(all(k in allowed or k.startswith('bam_record_') for k in changed),changed)
    self.assertTrue(b.bam_attention_write_controls_post_residual)
    self.assertFalse(b.bam_mlp_write_controls_post_residual)
    self.assertFalse(b.bam_write_controls_post_residual)
    self.assertEqual(b.bam_attention_write_controls_norm,'mlp')
    self.assertEqual(b.bam_attention_mlp_write_shared_norm,'none')
    self.assertFalse(b.bam_share_read_attention_write_down)
    self.assertEqual(b.mlp_dim_by_block,[6127,5938,6127])
    self.assertEqual(b.bam_final_local_mlp_dim,6126)
    self.assertEqual(b.bam_m_shift_mode,'read');self.assertEqual(b.bam_m_shift_init,.5)
    self.assertEqual(b.tensorboard_gcs_rotate_period,500)
    self.assertFalse(b.record_training_health_metrics)
    self.assertTrue(all(not v for k,v in b.get_keys().items() if k.startswith('bam_record_')))
    shapes=[]
    for c in (a,b):
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),1432383640)
      self.assertTrue(any('mlp_address_down' in p for p in flat))
      shapes.append({p:v.shape for p,v in flat.items()})
    self.assertEqual(*shapes)
    print('XL_POST_ATTN_MLP_NORM_PARAMS',1432383640,flush=True)

if __name__=='__main__':unittest.main()

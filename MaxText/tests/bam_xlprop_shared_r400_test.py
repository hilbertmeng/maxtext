"""Formal R400 sharing budget and private MLP shape regression."""
import unittest
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_xlprop_projection_sharing_test import XLProjectionSharingTest, BASE
NEW='BamXLPropMShiftReadG050ReadAttentionSharedR400TruePile'
class R400Test(XLProjectionSharingTest):
  def test_formal_parameter_budget_and_terminal_gate(self):
    trees={}
    for name,expected in ((BASE,1432383640),(NEW,1432399000)):
      c=self.config(name)
      args,*_=train_compile.get_shaped_inputs(self.mesh(c),c)
      flat=flatten_dict(nn.unbox(args[0].params));trees[name]=flat
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),expected)
      if name==NEW:
        self.assertEqual((c.bam_local_full_m_read_bottleneck_dim,c.bam_write_v_bottleneck_dim,c.bam_mlp_write_address_rank),(400,400,384))
        self.assertFalse(any('P_loc_down' in p for p in flat))
        self.assertTrue(all(v.shape[-1]==400 for p,v in flat.items() if 'full_m_read_down' in p and p[-1]=='kernel'))
        self.assertEqual([v.shape for p,v in flat.items() if 'mlp_address_down' in p and p[-1]=='kernel'],[(1920,9,384)])
        self.assertFalse(c.record_training_health_metrics)
        self.assertFalse(c.bam_record_m_shift_health)
        self.assertEqual(c.tensorboard_gcs_rotate_period,500)
        self.assertEqual(c.checkpoint_period,250)
        self.assertEqual(c.learning_rate_schedule_steps,50000)
      print('R400_BUDGET_OK',name,expected,flush=True)
    for p,v in trees[BASE].items():
      if 'mlp_address_down' in p or 'mlp_address_up' in p:self.assertEqual(v.shape,trees[NEW][p].shape)
if __name__=='__main__':unittest.main()

"""Verify the exact21-layer allocation and every scanned write/health location."""
import unittest,functools,numpy as np,jax
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train,train_compile
from bam_mlp_write_test import MLPWriteTest
EXP='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdL21TruePile'
class L21BudgetTest(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_budget_original_qk_and_seven_writers(self):
  c=self.config(EXP);self.assertEqual(c.num_decoder_layers,21);self.assertEqual(c.mlp_dim_by_block,[3177]*3)
  self.assertEqual((c.bam_local_qk_col_output_dim,c.bam_partial_rope_nope_dim,c.bam_standard_qk_dim),(57,57,18))
  self.assertFalse(c.bam_local_qk_add_before_rope);self.assertFalse(c.qk_norm)
  self.assertEqual((c.bam_k,c.bam_v,c.bam_abs_v_compression_dim),(75,32,8));self.assertEqual(c.bam_mlp_write_address_rank,256)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
  count=sum(int(np.prod(v.shape)) for v in flat.values());self.assertEqual(count,432088816)
  down=[v for p,v in flat.items() if 'mlp_address_down' in p];self.assertEqual(len(down),1);self.assertEqual(sorted(down[0].shape),sorted((1200,256,7)))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
  for layer in range(21):
   self.assertIn(f'bam/concat/qk_scores/layer_{layer:03d}/bam_over_standard',metrics)
   self.assertEqual(f'bam/concat/mlp_write_gate/layer_{layer:03d}/mean' in metrics,layer%3==1)
  print('L21_EXACT_BUDGET_ORIGINAL_QK_FULL_TRAINSTEP_SEVEN_WRITERS_OK',count,flush=True)
if __name__=='__main__':unittest.main()

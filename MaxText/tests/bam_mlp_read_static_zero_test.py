"""Zero static addresses keep dynamic reads alive and static parameters learnable."""
import math
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
import bam_mlp_read_test as base_tests
from bam_mlp_read_test import MlpBamReadTest
from layers import bam_mlp

EXP='BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesStaticZeroTruePile'
NORMAL='BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile'

class MlpReadStaticZeroTest(MlpBamReadTest):
  def test_full_budget_and_health(self):
    c=self.config(EXP);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,_,_,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432090944)
    self.assertEqual(c.mlp_dim_by_block,[3898,3772,3898])
    self.assertEqual(c.bam_mlp_read_static_key_init,'zero')
    self.assertFalse(c.record_training_health_metrics)
    self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
    self.assertFalse(any('gate' in '/'.join(p) for p in flat if 'mlp_bam_read' in p))
    print('ZERO_STATIC_FULL_BUDGET_OK',432090944,flush=True)

  def test_only_static_changes_and_dynamic_read_is_nonzero(self):
    c=self.config(EXP,dtype='float32',weight_dtype='float32');oldc=self.config(NORMAL,dtype='float32',weight_dtype='float32')
    reader=bam_mlp.BamMlpRead(c);old=bam_mlp.BamMlpRead(oldc)
    x=jax.random.normal(jax.random.key(60),(1,8,1200));matrix=jax.random.normal(jax.random.key(61),(1,8,75,32))
    for seed in (62,63,64):
      with nn.partitioning.axis_rules([]):
        p=nn.unbox(reader.init(jax.random.key(seed),x,matrix))
        q=nn.unbox(old.init(jax.random.key(seed),x,matrix))
        y=reader.apply(p,x,matrix)
      self.assertEqual(p['params'].keys(),q['params'].keys())
      for name in p['params']:
        if name=='static_key':
          np.testing.assert_array_equal(p['params'][name],0)
          self.assertGreater(float(jnp.linalg.norm(q['params'][name])),0)
        else:
          for a,b in zip(jax.tree.leaves(p['params'][name]),jax.tree.leaves(q['params'][name])):
            np.testing.assert_array_equal(a,b,err_msg=name)
      self.assertGreater(float(jnp.linalg.norm(y)),.01)
      static_grad=jax.grad(lambda a:jnp.mean(reader.apply({'params':a},x,matrix)**2))(p['params'])['static_key']
      self.assertGreater(float(jnp.linalg.norm(static_grad)),0)
    print('ZERO_STATIC_DYNAMIC_ALIVE_SAME_OTHER_PARAMS_OK',flush=True)

  def test_scanned_static_learning_and_causality(self):
    with mock.patch.object(base_tests,'ARMS',[EXP]):
      super().test_scan_gradients_and_causality()

if __name__=='__main__':unittest.main()

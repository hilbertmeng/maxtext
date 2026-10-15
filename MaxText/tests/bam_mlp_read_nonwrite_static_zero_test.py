"""Configuration-only nonwrite12 successor: local budgets and selected-path learning."""
import math
import unittest
from unittest import mock
import jax
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
import bam_mlp_read_test as base_tests
from bam_mlp_read_test import MlpBamReadTest
from layers import bam_mlp

EXP='BamMediumPropMShiftReadG050MLPReadNonWriteLayersNoReadGatesStaticZeroTruePile'

class NonWriteStaticZeroTest(MlpBamReadTest):
  def test_full_local_budget_location_and_health(self):
    def tree(c):
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      return flatten_dict(nn.unbox(args[0].params))
    c=self.config(EXP);flat=tree(c);parent=tree(self.config(base_tests.PARENT))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432097856)
    reader={p:a for p,a in flat.items() if 'mlp_bam_read' in p}
    self.assertEqual(sum(math.prod(a.shape) for a in reader.values()),373248*12)
    self.assertFalse(any('gate' in '/'.join(p) for p in reader))
    self.assertEqual(c.mlp_dim_by_block,[3898,3676,3898])
    self.assertEqual(c.bam_mlp_read_static_key_init,'zero')
    self.assertFalse(c.bam_mlp_read_gates)
    self.assertEqual([i for i in range(18) if bam_mlp.read_enabled(c,i)],
                     [i for i in range(18) if i%3!=1])
    self.assertFalse(c.record_training_health_metrics)
    self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
    selected={p[:p.index('mlp_bam_read')] for p in reader}
    prefixes={p[:p.index('mlp')] for p in flat if 'mlp' in p}
    for prefix in prefixes:
      count=lambda t:sum(math.prod(a.shape) for p,a in t.items() if p[:len(prefix)]==prefix)
      self.assertEqual(count(flat)-count(parent),-6912 if prefix in selected else 0)
    print('NONWRITE12_ZERO_STATIC_LOCAL_BUDGET_OK',432097856,flush=True)

  def test_selected_post_attention_ordering(self):
    with mock.patch.object(base_tests,'ARMS',[EXP]):
      super().test_post_attention_read_and_single_decay()

  def test_selected_scanned_gradients_and_causality(self):
    with mock.patch.object(base_tests,'ARMS',[EXP]):
      super().test_scan_gradients_and_causality()

if __name__=='__main__':unittest.main()

"""Ungated MLP memory reads: pruned gates, exact budget and causal learning."""
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

EXP='BamMediumPropMShiftReadG050MLPReadAllLayersNoReadGatesTruePile'

class MlpReadNoGatesTest(MlpBamReadTest):
  def test_full_budget_without_gate_parameters(self):
    c=self.config(EXP);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,_,_,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(a.shape) for a in flat.values()),432090944)
    reader={p:a for p,a in flat.items() if 'mlp_bam_read' in p}
    self.assertEqual(sum(math.prod(a.shape) for a in reader.values()),373248*18)
    self.assertFalse(any('gate' in '/'.join(p) for p in reader))
    self.assertEqual(c.mlp_dim_by_block,[3898,3772,3898])
    self.assertFalse(c.record_training_health_metrics)
    self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
    self.assertEqual(sum(bam_mlp.read_enabled(c,i) for i in range(18)),18)
    for p,a in flat.items():
      if 'mlp' in p and p[-2]=='wi_1':
        gate=flat[p[:-2]+('wi_0','kernel')]
        self.assertEqual(gate.shape[-1]-a.shape[-1],600)
    print('UNGATED_MLP_READ_FULL_BUDGET_OK',432090944,flush=True)

  def test_raw_dynamic_plus_static_two_read_reference(self):
    for dtype in (jnp.float32,jnp.bfloat16):
      c=self.tiny(EXP);c.get_keys()['dtype']=dtype
      reader=bam_mlp.BamMlpRead(c)
      x=jax.random.normal(jax.random.key(50),(1,8,150)).astype(dtype)
      matrix=jax.random.normal(jax.random.key(51),(1,8,75,32)).astype(dtype)
      with nn.partitioning.axis_rules([]):
        v=reader.init(jax.random.key(52),x,matrix);actual=reader.apply(v,x,matrix)
      p=nn.unbox(v['params']);self.assertFalse(any('gate' in k for k in p))
      hidden=nn.gelu(jnp.einsum('btd,dr->btr',x,p['read_down']['kernel'].astype(dtype)))
      key=jnp.einsum('btr,rnv->btnv',hidden,p['read_up']['kernel'].astype(dtype))+p['read_pre_bias'].astype(dtype)
      direction=key*jax.lax.rsqrt(jnp.mean(key.astype(jnp.float32)**2,-1,keepdims=True)+c.bam_read_key_epsilon).astype(dtype)
      dynamic=jnp.einsum('btkv,btnv->btnk',matrix,direction*c.bam_read_key_scale)
      static=jnp.einsum('btkv,vn->btnk',matrix,p['static_key'].astype(dtype))
      expected=(dynamic+static).reshape(actual.shape)
      error=float(jnp.linalg.norm(actual.astype(jnp.float32)-expected.astype(jnp.float32))/jnp.linalg.norm(expected.astype(jnp.float32)))
      self.assertLess(error,1e-6 if dtype==jnp.float32 else .015)
      # Compare the parent with identical shared leaves, isolating the removed gates.
      pc=self.tiny(base_tests.ARMS[-1]);pc.get_keys()['dtype']=dtype;parent=bam_mlp.BamMlpRead(pc)
      with nn.partitioning.axis_rules([]):
        pv=nn.unbox(parent.init(jax.random.key(52),x,matrix))
        pv['params'].update(p)
        parent_output=parent.apply(pv,x,matrix)
      parent_expected=(dynamic*pc.bam_read_gate_init+static).reshape(actual.shape)
      error=float(jnp.linalg.norm(parent_output.astype(jnp.float32)-parent_expected.astype(jnp.float32))/jnp.linalg.norm(parent_expected.astype(jnp.float32)))
      self.assertLess(error,1e-6 if dtype==jnp.float32 else .015)
      print('UNGATED_TWO_READ_REFERENCE_OK',str(dtype),error,flush=True)

  def test_new_arm_post_attention_ordering(self):
    with mock.patch.object(base_tests,'ARMS',[EXP]):
      super().test_post_attention_read_and_single_decay()

  def test_new_arm_scanned_gradients_and_causality(self):
    with mock.patch.object(base_tests,'ARMS',[EXP]):
      super().test_scan_gradients_and_causality()

if __name__=='__main__':unittest.main()

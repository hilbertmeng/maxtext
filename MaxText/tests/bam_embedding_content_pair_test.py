"""Embedding amplitude controls on the uncompensated private-MLP parent."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train_compile
import bam_common_read_raw_write_test as base_tests
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers.models import Transformer

PREFIX=base_tests.PARENT.removesuffix('TruePile')+'CommonReadScaleUnscaledMLPContent'
PARENT=PREFIX+'TruePile'
FIXED=PREFIX+'EmbeddingFixedTruePile'
UNSCALED=PREFIX+'EmbeddingUnscaledTruePile'
TAU=.0016052833


class EmbeddingContentPairTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,*args,**kwargs):
    self.experiment=getattr(self,'experiment',FIXED)
    return base_tests.CommonReadRawWriteTest.tiny(self,*args,**kwargs)

  def test_formal_budget_and_only_embedding_content_changes(self):
    parent=self.config(PARENT)
    for run,tau in [(FIXED,TAU),(UNSCALED,1.)]:
      c=self.config(run)
      changed={k for k,v in c.get_keys().items() if v!=parent.get_keys()[k]}
      self.assertEqual(changed,{'model_name','exp_class','compare_runs','bam_embedding_write_content_divisor'})
      self.assertEqual(c.bam_embedding_write_content_divisor,tau)
      self.assertEqual(c.bam_mlp_write_content_divisor,1.)
      self.assertFalse(c.bam_mlp_write_content_global_rmsnorm)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertTrue(c.bam_common_vo_read_scale and c.bam_pallas_core and c.bam_pallas_write)
      self.assertEqual(c.compare_runs,[PARENT])
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertTrue(c.scan_layers and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112210)
      self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if p[-1]=='common_vo_read_scale'),18)

  def test_initial_memory_gain_and_unchanged_random_parameters(self):
    self.experiment=FIXED;c=self.tiny();self.experiment=UNSCALED;rawc=self.tiny()
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    memories=[]
    def capture(next_fun,args,kwargs,context):
      out=next_fun(*args,**kwargs)
      if context.method_name=='initial_bam_matrix':memories.append(out)
      return out
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),nn.intercept_methods(capture):
      y,p=Transformer(c,mesh,quant=None).init_with_output(jax.random.key(7),tok,pos,seg,tok)
      raw_y,q=Transformer(rawc,mesh,quant=None).init_with_output(jax.random.key(7),tok,pos,seg,tok)
    self.assertEqual(len(memories),2)
    np.testing.assert_allclose(memories[1],TAU*memories[0],rtol=3e-5,atol=1e-8)
    for a,b in zip(jax.tree.leaves(nn.unbox(p)),jax.tree.leaves(nn.unbox(q))):np.testing.assert_array_equal(a,b)
    for values in (y,raw_y):
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(values)))

  def test_fixed_full_model_native_pallas_all_gradients(self):
    self.experiment=FIXED
    StandardValueTest.check_pallas(self,True)

  def test_unscaled_full_model_native_pallas_all_gradients(self):
    self.experiment=UNSCALED
    StandardValueTest.check_pallas(self,True)


if __name__=='__main__':unittest.main()

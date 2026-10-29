"""Whole-vector vs unscaled private-MLP writes, including the fused write path."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
import bam_common_read_raw_write_test as parent_tests
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers.bam_write_content import prepare_mlp_content, fixed_content

PARENT=parent_tests.EMBEDDING_NORM
PREFIX=parent_tests.PARENT.removesuffix('TruePile')
GLOBAL=PREFIX+'CommonReadScaleGlobalMLPContentNormTruePile'
UNSCALED=PREFIX+'CommonReadScaleUnscaledMLPContentTruePile'


class MLPGlobalContentNormTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,*args,**kwargs):
    self.experiment=getattr(self,'experiment',GLOBAL)
    return parent_tests.CommonReadRawWriteTest.tiny(self,*args,**kwargs)

  def test_formal_budgets_and_only_intended_configuration_changes(self):
    parent=self.config(PARENT)
    for run,global_norm in [(GLOBAL,True),(UNSCALED,False)]:
      c=self.config(run)
      changed={k for k,v in c.get_keys().items() if v!=parent.get_keys()[k]}
      expected={'model_name','exp_class','compare_runs','bam_mlp_write_content_divisor'}
      if global_norm:expected.add('bam_mlp_write_content_global_rmsnorm')
      self.assertEqual(changed,expected)
      self.assertEqual(c.bam_mlp_write_content_global_rmsnorm,global_norm)
      self.assertIsNone(c.bam_embedding_write_content_divisor)
      self.assertIsNone(c.bam_attention_write_content_divisor)
      self.assertEqual(c.bam_mlp_write_content_divisor,1.)
      self.assertEqual(c.compare_runs,[PARENT])
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
      self.assertTrue(c.bam_common_vo_read_scale and c.bam_pallas_core and c.bam_pallas_write)
      self.assertTrue(c.scan_layers and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112210)
      self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if p[-1]=='common_vo_read_scale'),18)

  @staticmethod
  def options(**overrides):
    return SimpleNamespace(**dict(dict(bam_mlp_write_content_global_rmsnorm=True,
        bam_mlp_write_content_divisor=1.,bam_write_rms_statistics_dtype='float32',
        bam_write_rms_epsilon=None,normalization_layer_epsilon=1e-6),**overrides))

  def test_global_reference_gradients_and_head_energy_distribution(self):
    c=self.options();x=jnp.array([1.,1.,3.,3.])
    ref=lambda a:a*jax.lax.rsqrt(jnp.mean(a*a)+1e-6)
    call=lambda a:prepare_mlp_content(a,c)
    np.testing.assert_allclose(call(x),ref(x),rtol=1e-6)
    np.testing.assert_allclose(jax.jacrev(call)(x),jax.jacrev(ref)(x),rtol=1e-6,atol=1e-7)
    self.assertNotEqual(float(jax.jacrev(call)(x)[0,2]),0.)
    y=call(x).reshape(2,2);energy=jnp.mean(y*y,axis=-1)
    np.testing.assert_allclose(energy[1]/energy[0],9.,rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(y*y),4.,rtol=1e-6)
    zero=jnp.zeros_like(x)
    self.assertTrue(bool(jnp.all(jnp.isfinite(jax.jacrev(call)(zero)))))
    np.testing.assert_array_equal(call(zero),zero)

  def test_bf16_statistics_and_reject_double_normalization(self):
    c=self.options();x=jnp.linspace(-.01,.01,1200).astype(jnp.bfloat16)
    xf=x.astype(jnp.float32)
    expected=(xf*jax.lax.rsqrt(jnp.mean(xf*xf)+1e-6)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(prepare_mlp_content(x,c),expected)
    for invalid in [None,.0045444682]:
      with self.assertRaises(ValueError):
        prepare_mlp_content(x,self.options(bam_mlp_write_content_divisor=invalid))

  def test_disabled_path_is_identity_and_unscaled_write_has_expected_gain(self):
    x=jnp.linspace(-.02,.02,1200)
    c=self.options(bam_mlp_write_content_global_rmsnorm=False)
    self.assertIs(prepare_mlp_content(x,c),x)
    tau=.0045444682
    np.testing.assert_array_equal(fixed_content(x,1.),x)
    np.testing.assert_allclose(fixed_content(x,1.),tau*fixed_content(x,tau),rtol=2e-7,atol=1e-9)

  def test_global_full_model_native_pallas_all_gradients(self):
    self.experiment=GLOBAL
    StandardValueTest.check_pallas(self,True)

  def test_unscaled_full_model_native_pallas_all_gradients(self):
    self.experiment=UNSCALED
    StandardValueTest.check_pallas(self,True)


if __name__=='__main__':unittest.main()

"""Frozen-address normalization transport and scan/tail capture checks."""
import sys
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import max_utils
import bam_pythia_common_read_raw_write_test as common
from layers.models import Transformer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium'))
import h_read_address_ablation as probe


class ReadAddressAblationTest(unittest.TestCase):
  def test_context_zero_targets_only_selected_layer(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=4,dtype='float32')
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tokens=jnp.ones((1,4),jnp.int32)
      positions=jnp.arange(4)[None];segments=jnp.ones_like(tokens)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=model.init(jax.random.key(7),tokens,positions,segments,tokens)
        with probe.address_intervention(zero_mha=jnp.array([0,0,2]),record_context=True):
          result,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
        values=np.asarray(probe.extract_addresses(aux['intermediates'],2,'mha_context_probe'))
        self.assertEqual(values.shape,(3,2));self.assertGreater(values[2,0],0.)
        self.assertEqual(values[2,1],0.)
        np.testing.assert_array_equal(values[:2,0],values[:2,1])
        self.assertTrue(np.isfinite(result[0]).all())
        with probe.address_intervention(zero_mha=jnp.array([0,0,0]),record_context=True):
          _,base=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
        with probe.address_intervention(zero_mha=jnp.array([0,1,0]),record_context=True):
          _,residual=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
        context=np.asarray(probe.extract_addresses(residual['intermediates'],2,'mha_context_probe'))
        np.testing.assert_array_equal(context[:,0],context[:,1])
        output=np.asarray(probe.extract_addresses(residual['intermediates'],2,'mha_output_probe'))
        self.assertEqual(output[1,2],0.)
        self.assertEqual(output[1,1],output[1,3])
        self.assertGreater(output[1,0],0.)
        base_write=np.asarray(probe.extract_addresses(base['intermediates'],2,'mha_write_content_probe'))
        residual_write=np.asarray(probe.extract_addresses(residual['intermediates'],2,'mha_write_content_probe'))
        np.testing.assert_array_equal(base_write,residual_write)
        print('MHA_CONTEXT_ZERO_SCAN_TAIL_TARGET_BOUNDARY_OK',flush=True)
    finally:helper.tearDown()

  def test_mean_address_is_not_renormalized(self):
    key=jax.random.normal(jax.random.key(1),(1,5,2,64))
    logit=jax.random.normal(jax.random.key(2),(1,5,2))
    mean=jax.random.normal(jax.random.key(3),(2,64))*.12
    proxy,gate=probe.frozen_inputs(key,logit,mean,1e-6)
    expected=jax.nn.sigmoid(logit)[...,None]*mean
    actual=jax.nn.sigmoid(gate)[...,None]*probe.normalized_address(proxy,1e-6)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-7)
    self.assertLess(float(jnp.sqrt(jnp.mean(mean**2))),.2)
    proxy_b,gate_b=probe.frozen_inputs(key.astype(jnp.bfloat16),logit.astype(jnp.bfloat16),mean,1e-6)
    self.assertEqual(proxy_b.dtype,jnp.bfloat16)
    self.assertEqual(gate_b.dtype,jnp.bfloat16)
    actual_b=jax.nn.sigmoid(gate_b.astype(jnp.float32))[...,None]*probe.normalized_address(proxy_b,1e-6)
    expected_b=jax.nn.sigmoid(logit.astype(jnp.bfloat16).astype(jnp.float32))[...,None]*mean.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_allclose(actual_b,expected_b,rtol=.02,atol=.001)

  def test_scanned_addresses_and_noop_preserve_outputs(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=128,dtype='float32')
      cfg.get_keys().update(bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tokens=jnp.ones((1,128),jnp.int32)
      positions=jnp.arange(128)[None];segments=jnp.ones_like(tokens)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=model.init(jax.random.key(7),tokens,positions,segments,tokens)
        original=model.apply(params,tokens,positions,segments,tokens)[0]
        with probe.address_intervention(record=True):
          result,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
        np.testing.assert_array_equal(original,result[0])
        means=probe.extract_addresses(aux['intermediates'],2)
        self.assertEqual(means.shape,(3,2,2,64))
        with probe.address_intervention(jnp.zeros((3,2),bool),means):
          noop=model.apply(params,tokens,positions,segments,tokens)[0]
        np.testing.assert_array_equal(original,noop)
        with probe.address_intervention(jnp.array([[False,False],[False,False],[True,True]]),means):
          frozen=model.apply(params,tokens,positions,segments,tokens)[0]
        self.assertTrue(np.isfinite(frozen).all())
        print('FROZEN_ADDRESS_MEAN_NORM_SCAN_TAIL_NOOP_OK',flush=True)
    finally:helper.tearDown()


if __name__=='__main__':unittest.main()

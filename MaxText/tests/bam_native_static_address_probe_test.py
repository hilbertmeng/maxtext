"""Native mean-address calibration and gate compensation preserve original no-op."""
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
import h_native_static_address_probe as probe
from h_mha_native_probe import extract
from h_read_address_ablation import freeze_address_parameters


class NativeStaticAddressTest(unittest.TestCase):
  def test_capture_noop_and_scanned_tail_freeze(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=4,dtype='float32')
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
      tokens=jnp.ones((1,4),jnp.int32);positions=jnp.arange(4)[None];segments=jnp.ones_like(tokens)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=nn.unbox(model.init(jax.random.key(7),tokens,positions,segments,tokens))
        original=model.apply(params,tokens,positions,segments,tokens)[0]
        with probe.address_probe(record=True):
          captured,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
        np.testing.assert_array_equal(original,captured[0])
        means=jnp.stack([extract(aux['intermediates'],'native_address_'+arm,2) for arm in ['q','k']],axis=1)
        self.assertEqual(means.shape,(3,2,2,64))
        def forward(params,control,means):
          with probe.address_probe(control,means):result=model.apply(params,tokens,positions,segments,tokens)
          return result[0]
        compiled=jax.jit(forward);control=np.zeros((3,2),bool)
        noop=compiled(params,jnp.asarray(control),means)
        np.testing.assert_allclose(original,noop,rtol=1e-6,atol=1e-6)
        control[2]=True;frozen=freeze_address_parameters(params,np.asarray(means),control,cfg.param_scan_axis)
        result=compiled(frozen,jnp.asarray(control),means)
        self.assertTrue(np.isfinite(result).all())
        print('NATIVE_STATIC_ADDRESS_CALIBRATION_NOOP_SCAN_TAIL_OK',flush=True)
    finally:helper.tearDown()


if __name__=='__main__':unittest.main()

"""Verify native FP32 MHA intervention routes and no-op behavior."""
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
import h_mha_native_probe as probe


class NativeMhaProbeTest(unittest.TestCase):
  def test_scan_tail_noop_and_both_output_paths(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=4,dtype='float32')
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
      tokens=jnp.ones((1,4),jnp.int32);positions=jnp.arange(4)[None];segments=jnp.ones_like(tokens)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=model.init(jax.random.key(7),tokens,positions,segments,tokens)
        original=model.apply(params,tokens,positions,segments,tokens)[0]
        def forward(params,modes):
          with probe.intervene_mha(modes,record=True):
            result,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
          return result[0],probe.extract(aux['intermediates'],'native_mha_head_probe',2),probe.extract(aux['intermediates'],'native_mha_write_probe',2)
        compiled=jax.jit(forward)
        noop,heads,writes=compiled(params,jnp.array([0,0,0]))
        np.testing.assert_allclose(original,noop,rtol=1e-6,atol=1e-6)
        _,residual,rwrites=compiled(params,jnp.array([0,1,0]))
        np.testing.assert_array_equal(writes,rwrites)
        self.assertEqual(float(residual[1,1]),float(residual[1,2]))
        _,both,bwrites=compiled(params,jnp.array([0,2,0]))
        self.assertEqual(float(both[1,0]),float(both[1,2]))
        # Native deferred factors already include content norm and write gate.
        self.assertNotEqual(float(bwrites[1]),float(writes[1]))
        _,tail,_=compiled(params,jnp.array([0,0,1]))
        self.assertEqual(float(tail[2,1]),float(tail[2,2]))
        print('NATIVE_MHA_SCAN_TAIL_NOOP_RESIDUAL_WRITE_PATHS_OK',flush=True)
    finally:helper.tearDown()


if __name__=='__main__':unittest.main()

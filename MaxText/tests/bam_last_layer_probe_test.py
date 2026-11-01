"""Verify scanned/tail branch interventions and preservation of M writers."""
import sys
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import bam_pythia_common_read_raw_write_test as common
from layers.models import Transformer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium'))
import h_last_layer_probe as probe
from h_mha_native_probe import extract

class LastLayerProbeTest(unittest.TestCase):
  def test_noop_context_local_and_writer_preservation(self):
    h=common.PythiaCommonReadRawWriteTest();h.setUp()
    try:
      cfg=h.tiny(common.H,length=4,dtype='float32')
      mesh=jax.sharding.Mesh(__import__('max_utils').create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tokens=jnp.ones((1,4),jnp.int32);positions=jnp.arange(4)[None];segments=jnp.ones_like(tokens)
      names,grid=probe.intervention_grid(3)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=model.init(jax.random.key(7),tokens,positions,segments,tokens)
        original=model.apply(params,tokens,positions,segments,tokens)[0]
        def forward(params,scales):
          with probe.intervene(scales,record=True):
            result,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
          return result[0],extract(aux['intermediates'],'last_path_head_probe',2),extract(aux['intermediates'],'last_path_write_probe',2)
        compiled=jax.jit(forward);noop,heads,writes=compiled(params,jnp.asarray(grid[0]))
        np.testing.assert_allclose(original,noop,rtol=1e-6,atol=1e-6);self.assertLess(float(heads[:,4].max()),1e-12)
        _,hc,wc=compiled(params,jnp.asarray(grid[1]));np.testing.assert_allclose(hc[2,3],heads[2,1],rtol=1e-6)
        _,ho,wo=compiled(params,jnp.asarray(grid[2]));np.testing.assert_allclose(ho[2,3],heads[2,0],rtol=1e-6)
        _,hb,wb=compiled(params,jnp.asarray(grid[3]));self.assertEqual(float(hb[2,3]),0)
        np.testing.assert_array_equal(writes,wc);np.testing.assert_array_equal(writes,wo);np.testing.assert_array_equal(writes,wb)
        _,hm,wm=compiled(params,jnp.asarray(grid[4]));np.testing.assert_array_equal(writes,wm)
        _,hh,_=compiled(params,jnp.asarray(grid[7]));np.testing.assert_allclose(hh[2,3],heads[2,1]*.25,rtol=1e-6)
        print('LAST_LAYER_NOOP_CONTEXT_LOCAL_BOTH_HALF_AND_WRITER_PRESERVATION_OK',flush=True)
    finally:h.tearDown()
if __name__=='__main__':unittest.main()

"""Read gate probes observe real fused inputs and preserve the model forward."""
import importlib.util
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import max_utils
import bam_pythia_common_read_raw_write_test as common
from layers.models import Transformer

path=Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium/h_write_balance_probe.py'
spec=importlib.util.spec_from_file_location('write_balance_probe',path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class ReadGateProbeTest(unittest.TestCase):
  def test_statistics_distinguish_gate_and_static_multiplier_and_qk_prefix(self):
    logits=tuple(jnp.zeros((1,2,1)) for _ in range(4))
    multipliers=tuple(jnp.full((1,2,1),2.) for _ in range(4))
    values=tuple(jnp.array([[[[1.,3.,100.]],[[1.,3.,100.]]]]) for _ in range(4))
    v=np.asarray(probe.read_gate_statistics(logits,multipliers,values,2))[0]
    np.testing.assert_allclose(v[:,:4],np.tile([.5,.25,2.,4.],(4,1)))
    np.testing.assert_allclose(v[:2,4],5.)
    np.testing.assert_allclose(v[2:,4],10010/3,rtol=1e-6)

  def test_scan_and_tail_capture_leaves_logits_unchanged(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=128,dtype='float32')
      cfg.get_keys().update(bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tok=jnp.ones((1,128),jnp.int32)
      pos=jnp.arange(128)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=model.init(jax.random.key(7),tok,pos,seg,tok)
        before=model.apply(params,tok,pos,seg,tok)[0]
        with probe.capture_writes(),probe.capture_reads():
          after,aux=model.apply(params,tok,pos,seg,tok,mutable=['intermediates'])
        np.testing.assert_array_equal(before,after[0])
        reads=np.asarray(probe.extract_read_stats(aux['intermediates'],2))
        self.assertEqual(reads.shape,(3,1,4,5));self.assertTrue(np.isfinite(reads).all())
        np.testing.assert_allclose(reads[...,0],.05,atol=1e-7)
        np.testing.assert_allclose(reads[...,1],.05**2,atol=1e-8)
        np.testing.assert_allclose(reads[...,2:4],1.,atol=1e-6)
        writes=np.asarray(probe.extract_stats(aux['intermediates']))
        self.assertEqual(writes.shape,(2,1,2,9))
        print('READ_GATE_PROBE_EXACT_FORWARD_EIGHT_GATES_SCAN_TAIL_OK',flush=True)
    finally:helper.tearDown()

if __name__=='__main__':unittest.main()

"""Verify causal gate interventions and scan/tail layer selection."""
import sys
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import bam_pythia_common_read_raw_write_test as common
from layers.models import Transformer

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/bam_llama2_medium'))
import h_read_gate_ablation as probe
from h_write_balance_probe import extract_read_stats


class ReadGateAblationTest(unittest.TestCase):
  def test_exact_upper_quarter_only(self):
    names, values = probe.intervention_matrix(32,24)
    self.assertEqual(len(names),7)
    for name, value in zip(names,values):
      if name == 'learned':
        self.assertFalse(value.any())
        continue
      region,arm,operation=name.split('_');expected=np.zeros((32,2),np.int32)
      self.assertEqual(region,'upper')
      indices=range(24,32)
      cols={'Q':[0],'K':[1],'QK':[0,1]}[arm]
      expected[np.ix_(list(indices),cols)]=1 if operation=='initial' else 2
      np.testing.assert_array_equal(value,expected)
    x=jnp.array([1.,2.],jnp.bfloat16)
    np.testing.assert_array_equal(probe.replace_logit(x,0,.05),x)
    self.assertTrue(np.all(np.asarray(jax.nn.sigmoid(probe.replace_logit(x,2,.05)))==0))

  def test_scan_tail_control_and_baseline_exact(self):
    helper=common.PythiaCommonReadRawWriteTest();helper.setUp()
    try:
      cfg=helper.tiny(common.H,length=128,dtype='float32')
      cfg.get_keys().update(bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tokens=jnp.ones((1,128),jnp.int32)
      positions=jnp.arange(128)[None];segments=jnp.ones_like(tokens)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=unfreeze(nn.unbox(model.init(jax.random.key(7),tokens,positions,segments,tokens)))
        flat=flatten_dict(params)
        for path,value in list(flat.items()):
          if path[-1] in ('W_lq_gate_b0','W_lk_gate_b0'):flat[path]=jnp.full_like(value,4.)
        params=unflatten_dict(flat)
        original=model.apply(params,tokens,positions,segments,tokens)[0]
        def forward(control):
          with probe.intervene_read_gates(control,.05,record=True):
            result,aux=model.apply(params,tokens,positions,segments,tokens,mutable=['intermediates'])
          return result[0],extract_read_stats(aux['intermediates'],2)
        raw,_=forward(jnp.zeros((3,2),jnp.int32))
        np.testing.assert_array_equal(original,raw)
        compiled=jax.jit(forward)
        unchanged,stats=compiled(jnp.zeros((3,2),jnp.int32))
        # Whole-forward JIT vs primitive eager lowering can differ by FP32 ULPs.
        np.testing.assert_allclose(original,unchanged,rtol=1e-6,atol=1e-6)
        _,stats=compiled(jnp.array([[2,0],[0,1],[2,2]],jnp.int32))
        expected=np.array([[0,float(jax.nn.sigmoid(4.))],
                           [float(jax.nn.sigmoid(4.)),.05],[0,0]])
        np.testing.assert_allclose(np.asarray(stats)[:,0,:2,0],expected,atol=1e-6)
        print('QK_GATE_ABLATION_BASELINE_EXACT_SCAN_TAIL_LAYER_SELECTION_OK',flush=True)
    finally:helper.tearDown()


if __name__=='__main__':unittest.main()

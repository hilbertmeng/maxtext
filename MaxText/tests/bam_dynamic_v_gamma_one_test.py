"""Dynamic V gamma1 plus original O gamma.5: unchanged static reads and fusion."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest

PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile'
RUN=PARENT.replace('050TruePile','050LocalVDynamicNormScale100TruePile')

class DynamicVGammaOneTest(MLPWriteTest,unittest.TestCase):
  tiny=CommonAllReadGammaTest.tiny

  def test_formal_budget_and_initial_neutral_v(self):
    c=self.config(RUN)
    self.assertEqual(c.compare_runs,[PARENT])
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write)
    self.assertFalse(c.bam_common_vo_read_scale or c.bam_complete_local_o_read_scale)
    self.assertIsNone(c.bam_local_vo_dynamic_norm_scale_depth_slope)
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);args,*_=train_compile.get_shaped_inputs(mesh,c)
    f=flatten_dict(nn.unbox(args[0].params));self.assertEqual(sum(math.prod(v.shape) for v in f.values()),432130624)
    for arm in ['v','o']:self.assertEqual(sum(math.prod(v.shape) for k,v in f.items() if k[-1]=='local_'+arm+'_dynamic_norm_scale'),9216)
    cfg,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN,initial=.5)
    np.testing.assert_array_equal(p['params']['local_v_dynamic_norm_scale'],np.ones((2,32)))
    np.testing.assert_array_equal(p['params']['local_o_dynamic_norm_scale'],np.full((2,32),.5))
    # The constant gamma consumes no RNG; all existing weights stay identical.
    pcfg,pmesh,parent,pp,pa,pm=LocalODynamicNormScaleTest.fixture(self,run=PARENT,initial=.5)
    source=flatten_dict(pp);target=flatten_dict(p)
    for k in source:np.testing.assert_array_equal(target[k],source[k])
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      a=module.apply(p,*args,M_in=matrix,deterministic=True)
      b=parent.apply(pp,*args,M_in=matrix,deterministic=True)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
    print('DYNAMIC_V1_O05_BUDGET_IDENTITY_OK',flush=True)

  def test_model_f32_native_pallas_gradients(self):CommonAllReadGammaTest.check_model(self,RUN)
  def test_model_bf16_native_pallas_gradients(self):CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,bf16_fp32_reference=True)

if __name__=='__main__':unittest.main()

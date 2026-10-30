"""Constant initialization of H read scalars, preserving parent defaults and Pallas gradients."""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from layers.models import Transformer
from bam_config import validate_bam_config
from bam_common_all_read_gamma_test import CommonAllReadGammaTest, H
from bam_common_read_raw_write_test import CommonReadRawWriteTest

RUN = H.removesuffix('TruePile') + 'ReadScaleInitOneTruePile'

class ReadScaleInitTest(CommonAllReadGammaTest):
  def test_isolated_configuration_and_scan_initialization(self):
    parent=self.config(H);cfg=self.config(RUN)
    changed={k for k,v in cfg.get_keys().items() if v!=parent.get_keys()[k]}
    self.assertEqual(changed,{'exp_class','model_name','compare_runs','bam_common_vo_read_scale_init'})
    self.assertEqual(cfg.compare_runs,[H]);self.assertFalse(cfg.qk_norm)
    self.assertEqual((cfg.mlp_dim,cfg.bam_memory_mlp_dim,cfg.bam_last_mlp_dim),(2510,1128,3904))
    selected=[]
    for run in [H,RUN]:
      cfg=self.tiny(run);mesh=jax.sharding.Mesh(__import__('max_utils').create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):p=nn.unbox(model.init(jax.random.key(31),tok,pos,seg,tok))
      selected.append(flatten_dict(p))
      for path,value in selected[-1].items():
        if path[-1]=='common_vo_read_scale':
          if run==RUN:np.testing.assert_array_equal(value,np.ones(value.shape))
          elif 'layers' in path:np.testing.assert_allclose(jnp.moveaxis(value,cfg.param_scan_axis,0).reshape(-1),1/np.sqrt(1+2*np.arange(2)),rtol=1e-6)
          else:np.testing.assert_allclose(value,[1/math.sqrt(5)],rtol=1e-6)
    self.assertEqual(selected[0].keys(),selected[1].keys())
    for path,value in selected[0].items():
      if path[-1]!='common_vo_read_scale':np.testing.assert_array_equal(value,selected[1][path])
    for value in [float('nan'),float('inf'),True]:
      with self.assertRaisesRegex(ValueError,'finite scalar'):validate_bam_config(dict(parent.get_keys(),bam_common_vo_read_scale_init=value))

  def test_constant_init_native_pallas_all_gradients(self):
    self.check_model(RUN)

  def test_unit_init_gamma_configuration_and_all_gradients(self):
    run=RUN.removesuffix('TruePile')+'LocalODynamicNormScale050TruePile'
    base=self.config(RUN);cfg=self.config(run)
    changed={k for k,v in cfg.get_keys().items() if v!=base.get_keys()[k]}
    self.assertEqual(changed,{'exp_class','model_name','compare_runs','bam_local_o_dynamic_norm_scale','bam_local_o_dynamic_norm_scale_init'})
    self.assertEqual(cfg.bam_common_vo_read_scale_init,1.)
    self.assertEqual(cfg.compare_runs,[RUN])
    self.assertEqual(cfg.bam_local_o_dynamic_norm_scale_init,.5)
    self.check_model(run)

if __name__=='__main__':unittest.main()

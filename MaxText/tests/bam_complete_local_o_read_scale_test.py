"""Whole localO head gains: independent static/dynamic scope and fused gradients."""
import contextlib
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils
import pyconfig
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_common_all_read_gamma_test import CommonAllReadGammaTest
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
from layers.attentions import BamAttention

P='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75'
PARENT=P+'LocalODynamicNormScale050TruePile'
RUN=P+'LocalOReadScalePerHead050TruePile'

class CompleteLocalOReadScaleTest(MLPWriteTest,unittest.TestCase):
  tiny=CommonAllReadGammaTest.tiny

  def test_formal_budget_initial_values_and_health(self):
    c=self.config(RUN)
    self.assertEqual(c.compare_runs,[PARENT,PARENT.replace('050TruePile','050PerHeadTruePile')]);
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write); self.assertFalse(c.bam_common_vo_read_scale or c.bam_local_o_dynamic_norm_scale)
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);args,*_=train_compile.get_shaped_inputs(mesh,c)
    f=flatten_dict(nn.unbox(args[0].params));self.assertEqual(sum(math.prod(v.shape) for v in f.values()),432112480)
    self.assertEqual(sum(math.prod(v.shape) for k,v in f.items() if k[-1]=='o_read_scale'),288)
    self.assertFalse(any(k[-1] in ('local_o_dynamic_norm_scale','v_read_scale','common_vo_read_scale') for k in f))
    cfg,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
    np.testing.assert_array_equal(p['params']['o_read_scale'],np.full(2,.5,np.float32))
    health=train.compute_params_norm(p,cfg);self.assertEqual(len([k for k in health if k.startswith('parameter_values/') and k.endswith('o_read_scale_head_mean')]),1)
    wd=flatten_dict(train.get_wd_tree(cfg,p));self.assertTrue(all(v==0. for k,v in wd.items() if k[-1]=='o_read_scale'))

  def test_explicit_static_dynamic_scope_and_all_gradients(self):
    cfg,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
    rc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_complete_local_o_read_scale=False)))
    ref=BamAttention(config=rc,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,
        read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=0,max_target_length=4,
        max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=cfg.dtype)
    f=flatten_dict(p);sp=next(k for k in f if k[-1]=='static_o_key');op=next(k for k in f if k[-1]=='o_read_scale')
    f[sp]=.1*jax.random.normal(jax.random.key(20),f[sp].shape);f[op]=jnp.array([-.4,1.2]);p=unflatten_dict(f)
    target=jax.random.normal(jax.random.key(21),(4,4,150));mtarget=jax.random.normal(jax.random.key(22),matrix.shape)
    def loss(reference,p,m,x):
      scale=flatten_dict(p)[op].reshape(1,1,2,1)
      def intercept(next_fun,aa,kw,ctx):
        result=next_fun(*aa,**kw)
        if ctx.method_name=='_independent_local_vo':return result[0],result[1]*scale
        if ctx.method_name=='_static_column' and aa[1]=='o':return result*scale
        return result
      with nn.intercept_methods(intercept) if reference else contextlib.nullcontext():
        y,mm=(ref if reference else module).apply(p,x,x,args[2],args[3],M_in=m,deterministic=True)
      return jnp.mean(y*target)+jnp.mean(mm*mtarget),(y,mm)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      (_,a),ga=jax.value_and_grad(lambda p,m,x:loss(False,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
      (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(True,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
    for x,y in zip(jax.tree.leaves((a,ga)),jax.tree.leaves((b,gb))):np.testing.assert_allclose(x,y,rtol=1e-3,atol=3e-6)
    self.assertTrue(bool(jnp.all(flatten_dict(ga[0])[op]!=0)))

  def test_native_pallas_f32_model_gradients(self):CommonAllReadGammaTest.check_model(self,RUN)
  def test_native_pallas_bf16_model_gradients(self):CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7,bf16_fp32_reference=True)

if __name__=='__main__':unittest.main()

"""Separate residual bases and context-only M write, including fused execution."""
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
from bam_common_all_read_gamma_test import CommonAllReadGammaTest,H
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest

RUN=H.removesuffix('TruePile')+'SeparateContextLocalOutputTruePile'

class SeparateContextLocalOutputTest(MLPWriteTest,unittest.TestCase):
  tiny=CommonAllReadGammaTest.tiny

  def test_formal_budget_and_parent_switches(self):
    c=self.config(RUN);p=self.config(H)
    changed={k for k,v in c.get_keys().items() if v!=p.get_keys()[k]}
    self.assertEqual(changed,{'model_name','exp_class','compare_runs','bam_separate_context_local_output','bam_attention_negative_diagonal','mlp_dim','base_mlp_dim','bam_last_mlp_dim'})
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2110,1128,3504))
    self.assertFalse(c.bam_attention_negative_diagonal)
    self.assertTrue(c.bam_common_vo_read_scale and c.bam_pallas_core and c.bam_pallas_write)
    self.assertEqual(c.compare_runs,[H])
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112210)
    self.assertEqual(sum(math.prod(v.shape) for k,v in flat.items() if 'out_local' in k),18*1200**2)
    for path in flat:
      if 'final_local_layer' in path:
        self.assertFalse(any(n in path for n in ('W_gw','P_loc_down','P_loc_up','memory_mlp')))
    print('SEPARATE_OUTPUT_BUDGET_OK',flush=True)

  def test_balanced_split_formal_local_mlp_budget(self):
    c=self.config(RUN.replace('OutputTruePile','OutputBalancedMLPTruePile'))
    parent=self.config(RUN)
    changed={k for k,v in c.get_keys().items() if v!=parent.get_keys()[k]}
    self.assertEqual(changed,{'model_name','exp_class','base_mlp_dim','mlp_dim','bam_memory_mlp_dim'})
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2243,995,3504))
    self.assertTrue(c.bam_separate_context_local_output)
    self.assertFalse(c.bam_attention_negative_diagonal)
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c);f=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in f.values()),432112210)
    # Both regular branches use3*D parameters per hidden unit; fixed address/gate
    # costs remain. Tail pays400 units entirely from the ordinary MLP.
    self.assertEqual((2510-2243)+(1128-995),400)
    ordinary=sum(math.prod(v.shape)//17 for k,v in f.items() if 'layers' in k and 'mlp' in k)
    small=sum(math.prod(v.shape)//17 for k,v in f.items() if 'layers' in k and any(
        n in k for n in ('memory_mlp','mlp_address_down','mlp_address_up','mlp_write_gate','mlp_write_gate_bias')))
    self.assertEqual((ordinary,small),(8074800,4040000))
    print('BALANCED_COMPLETE_BRANCH_RATIO',ordinary/small,flush=True)

    for path,value in f.items():
      if 'layers' in path and any('mlp' in n.lower() for n in path):
        print('BALANCED_MLP_PARAM', '/'.join(path),value.shape,math.prod(value.shape),flush=True)
    print('BALANCED_LOCAL_BUDGET_OK',flush=True)

  def test_output_reference_context_only_write_and_self_attention(self):
    c,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
    parts={};written=[]
    def intercept(next_fun,aa,kw,ctx):
      if ctx.method_name=='_project_context_and_local':parts['context'],parts['local']=aa[2],aa[3]
      if ctx.method_name=='_write':written.append(aa[0])
      return next_fun(*aa,**kw)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),nn.intercept_methods(intercept):
      y,mout=module.apply(p,*args,M_in=matrix,deterministic=True)
    pp=p['params']
    self.assertFalse(np.array_equal(pp['out']['kernel'],pp['out_local']['kernel']))
    expected=jnp.einsum('btnh,nhd->btd',parts['context'],pp['out']['kernel'])+jnp.einsum('btnh,nhd->btd',parts['local'],pp['out_local']['kernel'])
    np.testing.assert_allclose(y,expected,rtol=1e-5,atol=1e-6)
    np.testing.assert_array_equal(written[0],parts['context'])
    self.assertGreater(float(jnp.linalg.norm(parts['context'][:,0])),0.)
    f=flatten_dict(p);g=dict(f)
    # Independent O keys affect the residual, while attention M write ignores O.
    for path in f:
      if path[-1]=='static_o_key' or 'W_lo_c8_up' in path:g[path]=f[path]+.2
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      yy,mm=module.apply(unflatten_dict(g),*args,M_in=matrix,deterministic=True)
      def mloss(params):return jnp.sum(module.apply(params,*args,M_in=matrix,deterministic=True)[1])
      grads=flatten_dict(jax.grad(mloss)(p))
    self.assertGreater(float(jnp.max(jnp.abs(yy-y))),1e-5)
    np.testing.assert_array_equal(mm,mout)
    for path,value in grads.items():
      if 'out_local' in path or path[-1]=='static_o_key':np.testing.assert_array_equal(value,jnp.zeros_like(value))
    print('SEPARATE_OUTPUT_CONTEXT_WRITE_SELF_OK',flush=True)

  def test_native_pallas_f32_full_model_gradients(self):
    CommonAllReadGammaTest.check_model(self,RUN)

  def test_native_pallas_bf16_full_model_gradients(self):
    CommonAllReadGammaTest.check_model(self,RUN,dtype='bfloat16',bf16_atol=1e-7)

if __name__=='__main__':unittest.main()

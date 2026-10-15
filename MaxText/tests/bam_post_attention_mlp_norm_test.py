"""Reuse the MLP norm once for attention write controls without sharing projections."""
import unittest
import jax
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_shared_write_activation_test import SharedWriteActivationTest
from layers import fusion

EXP='BamMediumPropMShiftReadG050PostAttentionWriteMLPNormTruePile'
PARENT='BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile'

class PostAttentionMlpNormTest(SharedWriteActivationTest):
  def test_full_budget_and_unchanged_parameter_tree(self):
    trees=[]
    for arm in (PARENT,EXP):
      c=self.config(arm);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432111680)
      self.assertEqual(c.mlp_dim_by_block,[3802,3676,3802])
      self.assertEqual(c.bam_attention_mlp_write_shared_norm,'none')
      self.assertTrue(any('mlp_address_down' in p for p in flat))
      self.assertFalse(c.record_training_health_metrics)
      self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
      trees.append({p:v.shape for p,v in flat.items()})
    self.assertEqual(*trees)
    print('POST_ATTN_MLP_NORM_FULL_BUDGET_OK',432111680,flush=True)

  def test_once_per_norm_actual_controls_and_identical_initialization(self):
    for index in (0,1,2):
      c,mesh,module,args,m,variables=self.layer_fixture(EXP,index)
      parent=fusion.SubDecoderLayer(self.tiny(arm=PARENT),mesh,layer_inx=index,sliding_window_size=4)
      tape={}
      names=('pre_self_attention_layer_norm','post_self_attention_layer_norm','W_gw',
             'P_loc_down','mlp_address_down','mlp_write_gate','mlp')
      def intercept(next_fun,a,k,ctx):
        y=next_fun(*a,**k)
        if ctx.method_name=='__call__' and ctx.module.name in names:
          tape.setdefault(ctx.module.name,[]).append((a[0],y))
        return y
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        with nn.intercept_methods(intercept):module.apply(variables,*args,M_in=m,is_global=True)
        self.assertEqual(len(tape['pre_self_attention_layer_norm']),1)
        self.assertEqual(len(tape['post_self_attention_layer_norm']),1)
        h=tape['post_self_attention_layer_norm'][0][1]
        np.testing.assert_array_equal(tape['W_gw'][0][0],h)
        np.testing.assert_array_equal(tape['P_loc_down'][0][0],h)
        np.testing.assert_array_equal(tape['mlp'][0][0],h)
        if index==1:
          np.testing.assert_array_equal(tape['mlp_address_down'][0][0],h)
          np.testing.assert_array_equal(tape['mlp_write_gate'][0][0],h)
          self.assertGreater(float(np.linalg.norm(
              tape['P_loc_down'][0][1]-tape['mlp_address_down'][0][1])),1e-6)
        a=module.init(jax.random.key(32),*args,M_in=m,is_global=True)
        b=parent.init(jax.random.key(32),*args,M_in=m,is_global=True)
        af=flatten_dict(nn.unbox(a['params']));bf=flatten_dict(nn.unbox(b['params']))
        self.assertEqual(af.keys(),bf.keys())
        for p in af:np.testing.assert_array_equal(af[p],bf[p],err_msg=str(p))
        for x,y in zip(module.apply(a,*args,M_in=m,is_global=True),parent.apply(b,*args,M_in=m,is_global=True)):
          np.testing.assert_array_equal(x,y)
    print('POST_ATTN_MLP_NORM_ONCE_CORRECT_INPUT_SAME_INIT_OK',flush=True)

  def test_scanned_gradients_and_causality(self):
    self.check_gradients(EXP)

if __name__=='__main__':unittest.main()

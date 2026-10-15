"""Pythia SwiGLU: exact budgets, preserved non-MLP trees and causal gradients."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from pythia_test import PythiaTest
from pythia_family_test import ARMS

SWI=[ARMS[0]+'SwiGLU',ARMS[1]+'SwiGLU',ARMS[2]+'SwiGLU']
COUNTS=[432388800,434356331,432348080]
MLP=[207496800,207467984,249337632]

class PythiaSwiGLUTest(PythiaTest):
  def test_full_size_budgets_and_non_mlp_shapes(self):
    for old,new,total,mlp in zip(ARMS,SWI,COUNTS,MLP):
      flats=[]
      for arm in (old,new):
        c=self.config(exp_class=arm,base_emb_dim=1200,base_num_decoder_layers=18,
            base_num_query_heads=16,base_num_kv_heads=16,head_dim=75,vocab_size=50432)
        m,mesh=self.model(c)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):
          p=jax.eval_shape(lambda k:m.init(k,*self.inputs(),enable_dropout=False)['params'],jax.random.key(0))
        flats.append(flatten_dict(nn.unbox(p)))
        self.assertTrue(c.pythia_parallel_residual)
        self.assertFalse(c.record_training_health_metrics)
        self.assertFalse(c.qk_norm)
        self.assertEqual(c.pythia_rotary_dim,18)
      self.assertEqual(sum(math.prod(v.shape) for v in flats[1].values()),total)
      self.assertEqual(sum(math.prod(v.shape) for k,v in flats[1].items() if 'mlp' in k and k[-2] in ('wi','wi_0','wi_1','wo')),mlp)
      shape=lambda f:{k:v.shape for k,v in f.items() if 'mlp' not in k}
      self.assertEqual(shape(flats[0]),shape(flats[1]))
      print('SWIGLU_BUDGET',new,total,mlp,flush=True)

  def test_all_arm_gradients_and_causality(self):
    for arm in SWI:
      opts=dict(exp_class=arm,base_num_decoder_layers=3,base_mlp_dim=128,
          pythia_rotary_dim=4,remat_policy='full')
      if arm.startswith('Mudd'):opts.update(mlp_dim_by_layer=[64,128,192])
      if arm.startswith('Bam'):
        opts.update(base_emb_dim=300,base_num_query_heads=4,base_num_kv_heads=4,
            head_dim=75,pythia_rotary_dim=18,base_num_decoder_layers=6)
      c=self.config(**opts)
      if c.bam_enabled:
        c.get_keys().update(emb_bam_num_head=4,mlp_dim_by_block=[128,96,128],
            bam_local_full_m_read_bottleneck_dim=64,bam_write_v_bottleneck_dim=64,
            bam_mlp_write_address_rank=64,bam_splash_attention=False,
            bam_layer_modes=c.bam_layer_modes[:6])
      m,mesh,p=self.init(c)
      args=self.inputs()
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        def run(p,t):return m.apply({'params':p},t,*args[1:],enable_dropout=False)[0]
        val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(run(p,args[0])**2)))(p)
        self.assertTrue(np.isfinite(float(val)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)))
        flat=flatten_dict(nn.unbox(grad))
        ins=[v for k,v in flat.items() if 'mlp' in k and k[-2] in ('wi','wi_0','wi_1') and k[-1]=='kernel']
        self.assertTrue(ins)
        self.assertTrue(all(float(jnp.linalg.norm(g))>0 for g in ins))
        y=run(p,args[0]);z=run(p,args[0].at[:,-1].set(12))
        np.testing.assert_allclose(y[:,:-1],z[:,:-1],rtol=3e-5,atol=3e-5)
      print('SWIGLU_GRAD_CAUSAL',arm,float(val),flush=True)

  def test_reject_unrequested_variant_and_invalid_widths(self):
    with self.assertRaises(ValueError):self.config(mlp_activations=['silu','linear'])
    with self.assertRaises(ValueError):self.config(exp_class=SWI[0],mlp_activations=['gelu_exact'])
    with self.assertRaises(ValueError):self.config(exp_class=SWI[1],mlp_dim_by_layer=[64])

if __name__=='__main__':unittest.main()

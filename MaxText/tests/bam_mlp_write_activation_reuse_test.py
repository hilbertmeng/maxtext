"""Actual pre-attention activation reuse, not another projection of post-attention x."""
import unittest
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
import bam_projection_sharing_test as sharing
from layers import linears
from layers.models import Transformer
EXPS=[('BamMediumPropMShiftReadG050SharedR256ReuseActivationMLPWriteTruePile',256,432104480),
      ('BamMediumPropMShiftReadG050SharedR384ReuseActivationMLPWriteTruePile',384,432118688)]
class ActivationReuseTest(MLPWriteTest,unittest.TestCase):
  mesh=sharing.ProjectionSharingTest.mesh
  tiny=sharing.ProjectionSharingTest.tiny
  attention=sharing.ProjectionSharingTest.attention
  def test_formal_budgets_and_no_private_down(self):
    for exp,rank,total in EXPS:
      c=self.config(exp);args,*_=train_compile.get_shaped_inputs(self.mesh(c),c)
      f=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(int(np.prod(v.shape)) for v in f.values()),total)
      self.assertFalse(any('mlp_address_down' in p or 'P_loc_down' in p for p in f))
      self.assertEqual([v.shape for p,v in f.items() if 'mlp_address_up' in p and p[-1]=='kernel'],[(rank,6,16,32)])
      self.assertFalse(c.record_training_health_metrics);self.assertFalse(c.bam_record_m_shift_health)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      print('ACTIVATION_REUSE_BUDGET_OK',exp,total,flush=True)
  def test_reuses_exact_gelu_activation_once(self):
    for exp,_,_ in EXPS:
      c=self.tiny(exp);c.get_keys().update(scan_layers=False,remat_policy='none');mesh=self.mesh(c)
      model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
      captured=[];original=linears.DenseGeneral.__call__
      def capture(module,inputs,*args,**kw):
        out=original(module,inputs,*args,**kw)
        if module.name in ('full_m_read_down','mlp_address_up','mlp_write_gate'):
          value=out if module.name=='full_m_read_down' else inputs
          jax.debug.callback(lambda v,n=module.name:captured.append((n,np.asarray(v).copy())),value,ordered=True)
        return out
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(linears.DenseGeneral,'__call__',capture):
        model.init(jax.random.key(31),tok,pos,tok,mask,mask,enable_dropout=False)
      jax.effects_barrier();downs=[v for n,v in captured if n=='full_m_read_down'];ups=[v for n,v in captured if n=='mlp_address_up'];gates=[v for n,v in captured if n=='mlp_write_gate']
      self.assertEqual(len(downs),3);self.assertEqual(len(ups),1);self.assertEqual(len(gates),1)
      np.testing.assert_allclose(ups[0],np.asarray(nn.gelu(jnp.asarray(downs[1]))),rtol=1e-6,atol=1e-6)
      self.assertEqual(gates[0].shape[-1],150)  # Gate still consumes residual, not reused rank8 activation.
  def test_scan_gradient_causality(self):
    for exp,_,_ in EXPS:
      c=self.tiny(exp);mesh=self.mesh(c);model=Transformer(c,mesh,quant=None)
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init(jax.random.key(30),tok,pos,tok,mask,mask,enable_dropout=False)['params']
        def out(params,t):return model.apply({'params':params},t,pos,t,mask,mask,enable_dropout=False)[0]
        loss,g=jax.jit(jax.value_and_grad(lambda params:jnp.mean(out(params,tok)**2)))(p)
        self.assertTrue(np.isfinite(float(loss)));flat=flatten_dict(nn.unbox(g))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
        for name in ('full_m_read_down','mlp_address_up','mlp_write_gate'):
          self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if name in path),0,name)
        np.testing.assert_allclose(out(p,tok)[:,:3],out(p,tok.at[:,3].set(5))[:,:3],rtol=1e-6,atol=1e-6)
  def test_optional_attention_return_preserves_existing_contract(self):
    c=self.tiny(EXPS[0][0]);a=self.attention(c);x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32));pos=jnp.arange(4)[None];seg=jnp.ones((1,4),jnp.int32)
    p=a.init(jax.random.key(3),x,x,pos,seg,M_in=m,defer_write=True)
    old=a.apply(p,x,x,pos,seg,M_in=m,defer_write=True)
    new=a.apply(p,x,x,pos,seg,M_in=m,defer_write=True,return_shared_features=True)
    self.assertEqual(len(old),3);self.assertEqual(len(new),4)
    for u,v in zip(jax.tree.leaves(old),jax.tree.leaves(new[:3])):np.testing.assert_array_equal(u,v)
    np.testing.assert_allclose(new[3],a.apply(p,x,method=a.shared_read_write_features),rtol=1e-6,atol=1e-6)
if __name__=='__main__':unittest.main()

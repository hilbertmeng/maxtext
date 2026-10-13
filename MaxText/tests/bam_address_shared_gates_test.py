"""Shared address features: gate routing, scan gradients, budget and initialization."""
import functools
import math
import unittest
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions, initializers, linears
from layers.models import Transformer

PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
NEW='BamMediumPropSharedR256GeneralReadAddressSharedGatesTruePile'
READ_GATES=('W_lq_gate','W_lk_gate','W_lv_gate','W_R_gate',
            'general_q_static_gate','general_k_static_gate','general_v_static_gate','general_o_static_gate')

class SharedGatesTest(MLPWriteTest, unittest.TestCase):
  def tiny(self):
    c=self.config(NEW,bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,mlp_dim_by_block=[32,24,32],
        bam_layer_modes=c.bam_layer_modes[:3])
    return c

  def mesh(self,c):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)

  def attention(self,c):
    return attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,
        head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,
        mesh=self.mesh(c),attention_kernel='dot_product_chunk',dtype=c.dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type,
        kernel_init=initializers.get_init_method(c.init_method))

  def test_full_config_budget_and_gate_shapes(self):
    for exp,total,rank in ((PARENT,432111104,1200),(NEW,432100832,256)):
      c=self.config(exp);mesh=self.mesh(c)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),total)
      names=set()
      for path,value in flat.items():
        for name in READ_GATES+('W_gw','mlp_write_gate'):
          if name in path and path[-1]=='kernel':
            self.assertEqual(value.shape[0],rank,(path,value.shape));names.add(name)
      self.assertEqual(names,set(READ_GATES+('W_gw','mlp_write_gate')))
    print('SHARED_GATES_PARAMS_OK total=432100832 savings=2537472 MLP=[3852,3725,3852] residue=-10272',flush=True)

  def test_single_cached_read_hidden_and_gate_inputs(self):
    c=self.tiny();a=self.attention(c)
    x=jax.random.normal(jax.random.key(10),(1,4,150));pos=jnp.arange(4)[None]
    m=jax.random.normal(jax.random.key(11),(1,4,75,32));seg=jnp.ones((1,4),jnp.int32)
    original=linears.DenseGeneral.__call__;calls=[]
    def capture(module,inputs,*args,**kw):
      calls.append((module.name,inputs.shape[-1]));return original(module,inputs,*args,**kw)
    with mock.patch.object(linears.DenseGeneral,'__call__',capture):
      a.init(jax.random.key(12),x,x,pos,seg,M_in=m,defer_write=True)
    self.assertEqual(sum(name=='full_m_read_down' for name,_ in calls),1)
    self.assertEqual(sum(name=='P_loc_down' for name,_ in calls),1)
    for name in READ_GATES+('W_gw',):
      widths=[width for called,width in calls if called==name]
      self.assertEqual(widths,[8],(name,widths))

  def test_direct_and_deferred_write_use_same_hidden_gate(self):
    c=self.tiny();a=self.attention(c)
    x=jax.random.normal(jax.random.key(20),(2,4,150));o=jax.random.normal(jax.random.key(21),(2,4,2,75))
    m=jax.random.normal(jax.random.key(22),(2,4,75,32))
    v=a.init(jax.random.key(23),o,x,m,method=a._write)
    direct,gate=a.apply(v,o,x,m,method=a._write)
    content,address,deferred_gate=a.apply(v,o,x,method=a._deferred_write_factors)
    h=nn.gelu(a.apply(v,x,method=lambda mod,z:mod.P_loc_down(z)))
    p=nn.unbox(v['params']);expected=jax.nn.sigmoid(h@p['W_gw']['kernel']+p['gw_b0'])
    np.testing.assert_allclose(gate,expected,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(gate,deferred_gate)
    np.testing.assert_allclose(direct,c.bam_lambda_decay*m+jnp.einsum('btnk,btnv->btkv',content,address),rtol=1e-5,atol=1e-5)

  def test_scan_all_gate_gradients_and_future_causality(self):
    c=self.tiny();mesh=self.mesh(c);model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
    call=(tok,pos,tok,mask,mask)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      params=model.init({'params':jax.random.key(30),'dropout':jax.random.key(31),'aqt':jax.random.key(32)},*call,enable_dropout=False)['params']
      flat=flatten_dict(nn.unbox(params))
      # Static Q/K/O keys start at zero: activate them to test their gate consumers.
      for path,value in list(flat.items()):
        if path[-1] in ('static_q_key','static_k_key','static_o_key'):
          flat[path]=jnp.full_like(value,.03)
      params=unflatten_dict(flat)
      def output(p,tokens):
        return model.apply({'params':p},tokens,pos,tokens,mask,mask,enable_dropout=False,rngs={'aqt':jax.random.key(32)})[0]
      val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(output(p,tok)**2)))(params)
      self.assertTrue(np.isfinite(float(val)))
      self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
      f=flatten_dict(grad)
      for name in READ_GATES+('W_gw','mlp_write_gate','full_m_read_down','P_loc_down','mlp_address_down'):
        self.assertGreater(sum(float(jnp.sum(g*g)) for path,g in f.items() if name in path),0,name)
      base=output(params,tok);changed=output(params,tok.at[:,3].set(5))
      np.testing.assert_allclose(base[:,:3],changed[:,:3],rtol=1e-6,atol=1e-6)

  def test_write_init_analytical_moment_matches_parent(self):
    # Numerical Gaussian quadrature independently checks the exact GELU moment.
    nodes,weights=np.polynomial.hermite.hermgauss(96)
    for d,r in ((150,8),(1200,256)):
      q=d*.006**2;x=np.sqrt(2*q)*nodes
      gelu=x*np.array([.5*(1+math.erf(z/math.sqrt(2))) for z in x])
      moment=float(np.dot(weights,gelu*gelu)/np.sqrt(np.pi))
      base=lambda key,shape,dtype,in_axis=0,out_axis=1:jnp.ones(shape,dtype)
      scale=float(initializers.bam_shared_address_gate_init(base,d,r)(jax.random.key(0),(1,),jnp.float32)[0])
      self.assertAlmostEqual(r*.006**2*scale**2*moment,d*.006**2,places=7)

if __name__=='__main__':unittest.main()

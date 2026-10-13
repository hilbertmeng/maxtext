"""Tied read/write banks: exact budget, reuse, post-attention input and scan gradients."""
import unittest
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions, initializers, linears
from layers.models import Transformer

BASE='BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
EXPS=[('BamMediumPropMShiftReadG050ReadAttentionSharedR256TruePile',256,False,432111680),
      ('BamMediumPropMShiftReadG050ReadAttentionSharedR384TruePile',384,False,432121472),
      ('BamMediumPropMShiftReadG050ReadAttentionMLPSharedR384TruePile',384,True,432118688)]
class ProjectionSharingTest(MLPWriteTest,unittest.TestCase):
  def mesh(self,c):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
  def tiny(self,exp):
    c=self.config(exp,bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,mlp_dim_by_block=[32,24,32],
        bam_layer_modes=c.bam_layer_modes[:3])
    return c
  def attention(self,c):
    return attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,
        head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,
        mesh=self.mesh(c),attention_kernel='dot_product_chunk',dtype=c.dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type,
        kernel_init=initializers.get_init_method(c.init_method))
  def test_formal_budget_and_layer_specific_ties(self):
    for exp,rank,mlp,total in EXPS:
      c=self.config(exp);args,_,_,_=train_compile.get_shaped_inputs(self.mesh(c),c)
      flat=flatten_dict(nn.unbox(args[0].params))
      actual=sum(int(np.prod(v.shape)) for v in flat.values())
      self.assertEqual(actual,total,(exp,actual,total))
      self.assertFalse(any('P_loc_down' in p for p in flat))
      downs=[(p,v) for p,v in flat.items() if 'full_m_read_down' in p and p[-1]=='kernel']
      self.assertEqual(len(downs),3)
      for p,v in downs:self.assertEqual(v.shape,(1200,6,rank),(p,v.shape))
      self.assertEqual(any('mlp_address_down' in p for p in flat),not mlp)
      self.assertEqual(sum(6 for p in flat if 'mlp_address_up' in p and p[-1]=='kernel'),6)
      self.assertEqual(sum(6 for p in flat if 'M_shift_gate' in p and p[-1]=='kernel'),18)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      print('PROJECTION_BUDGET_OK',exp,actual,c.mlp_dim_by_block,flush=True)
  def test_attention_computes_down_once_and_preserves_raw_carry(self):
    c=self.tiny(EXPS[1][0]);a=self.attention(c)
    x=jax.random.normal(jax.random.key(10),(1,4,150));pos=jnp.arange(4)[None]
    m=jax.random.normal(jax.random.key(11),(1,4,75,32));seg=jnp.ones((1,4),jnp.int32)
    original=linears.DenseGeneral.__call__;calls=[]
    def capture(module,inputs,*args,**kw):
      calls.append(module.name);return original(module,inputs,*args,**kw)
    with mock.patch.object(linears.DenseGeneral,'__call__',capture):
      v=a.init(jax.random.key(12),x,x,pos,seg,M_in=m,defer_write=True)
    self.assertEqual(calls.count('full_m_read_down'),1)
    self.assertNotIn('P_loc_down',calls)
    _,carry,_=a.apply(v,x,x,pos,seg,M_in=m,defer_write=True)
    np.testing.assert_array_equal(carry,m)
  def test_direct_and_deferred_write_and_shared_kernel_gradient(self):
    c=self.tiny(EXPS[0][0]);a=self.attention(c)
    x=jax.random.normal(jax.random.key(20),(1,4,150));o=jax.random.normal(jax.random.key(21),(1,4,2,75))
    m=jax.random.normal(jax.random.key(22),(1,4,75,32))
    v=a.init(jax.random.key(23),o,x,m,method=a._write)
    direct,gate=a.apply(v,o,x,m,method=a._write)
    content,address,deferred_gate=a.apply(v,o,x,method=a._deferred_write_factors)
    np.testing.assert_array_equal(gate,deferred_gate)
    np.testing.assert_allclose(direct,c.bam_lambda_decay*m+jnp.einsum('btnk,btnv->btkv',content,address),rtol=1e-5,atol=1e-5)
    self.assertNotIn('P_loc_down',v['params'])
    grad=jax.grad(lambda p:jnp.sum(a.apply({'params':p},o,x,m,method=a._write)[0]**2))(v['params'])
    self.assertGreater(float(jnp.sum(nn.unbox(grad)['full_m_read_down']['kernel']**2)),0)
  def test_mlp_reprojects_post_attention_input_and_scan_is_causal(self):
    c=self.tiny(EXPS[2][0]);mesh=self.mesh(c);model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
    c.get_keys().update(scan_layers=False,remat_policy='none')
    original=linears.DenseGeneral.__call__;captured=[]
    def capture(module,inputs,*args,**kw):
      if module.name=='full_m_read_down':captured.append(inputs)
      return original(module,inputs,*args,**kw)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      with mock.patch.object(linears.DenseGeneral,'__call__',capture):
        params=model.init({'params':jax.random.key(30),'dropout':jax.random.key(31),'aqt':jax.random.key(32)},tok,pos,tok,mask,mask,enable_dropout=False)['params']
      # Two blocks project once, the MLP-write block projects the same bank twice.
      self.assertEqual(len(captured),4)
      self.assertFalse(np.allclose(np.asarray(captured[1]),np.asarray(captured[2])))
      c.get_keys().update(scan_layers=True,remat_policy='full')
      model=Transformer(c,mesh,quant=None)
      params=model.init({'params':jax.random.key(30),'dropout':jax.random.key(31),'aqt':jax.random.key(32)},tok,pos,tok,mask,mask,enable_dropout=False)['params']
      self.assertFalse(any('mlp_address_down' in p or 'P_loc_down' in p for p in flatten_dict(params)))
      def output(p,t):return model.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False,rngs={'aqt':jax.random.key(32)})[0]
      val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(output(p,tok)**2)))(params)
      self.assertTrue(np.isfinite(float(val)))
      self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
      for name in ('full_m_read_down','P_loc_up','mlp_address_up','M_shift_gate'):
        self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in flatten_dict(nn.unbox(grad)).items() if name in p),0,name)
      np.testing.assert_allclose(output(params,tok)[:,:3],output(params,tok.at[:,3].set(5))[:,:3],rtol=1e-6,atol=1e-6)
  def test_shift_boundary_and_initial_opening(self):
    m=jnp.arange(1*4*2*3,dtype=jnp.float32).reshape(1,4,2,3)
    gate=jnp.full((1,4,3),.5);seg=jnp.array([[1,1,2,2]]);pos=jnp.array([[0,1,0,1]])
    mixed=attentions.mix_bam_previous_token(m,gate,seg,pos)
    np.testing.assert_array_equal(mixed[:,[0,2]],m[:,[0,2]])
    np.testing.assert_array_equal(mixed[:,1],.5*(m[:,0]+m[:,1]))
if __name__=='__main__':unittest.main()

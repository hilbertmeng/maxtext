"""Shared write-address activations: actual inputs, one projection and joint gradients."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_post_residual_write_test import PostResidualWriteTest
from layers import fusion
from layers.models import Transformer

ATTENTION = 'BamMediumPropPostAttentionWriteSharedR256AttentionNormTruePile'
MLP = 'BamMediumPropPostAttentionWriteSharedR256MLPNormTruePile'
PARENT = 'BamMediumPropMShiftReadG050PostAttentionWriteControlsTruePile'

class SharedWriteActivationTest(PostResidualWriteTest):
  def mesh_for(self,c):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)

  def test_full_size_shared_budget(self):
    trees=[]
    for arm in (ATTENTION,MLP):
      c=self.config(arm);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432104480)
      self.assertFalse(any('mlp_address_down' in p for p in flat))
      self.assertTrue(any('P_loc_down' in p for p in flat))
      self.assertEqual(c.mlp_dim_by_block,[3802,3761,3802])
      self.assertEqual(c.bam_local_full_m_read_bottleneck_dim,256)
      self.assertFalse(c.record_training_health_metrics)
      self.assertFalse(c.record_internal_nn_metrics)
      self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
      trees.append({p:v.shape for p,v in flat.items()})
    self.assertEqual(*trees)
    print('SHARED_WRITE_FULL_PARAMS_OK',432104480,flush=True)

  def layer_fixture(self,arm,layer=1):
    c=self.tiny(arm=arm);mesh=self.mesh_for(c)
    module=fusion.SubDecoderLayer(c,mesh,layer_inx=layer,sliding_window_size=4)
    x=jax.random.normal(jax.random.key(30),(1,4,150))
    m=jax.random.normal(jax.random.key(31),(1,4,75,32))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    args=(x,seg,pos,tok,None,True,'train',None)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      variables=nn.unbox(module.init(jax.random.key(32),*args,M_in=m,is_global=True))
    # Unequal learned scales distinguish norm sources beyond their identical init.
    variables['params']['pre_self_attention_layer_norm']['scale']=jnp.linspace(-.35,.2,150)
    variables['params']['post_self_attention_layer_norm']['scale']=jnp.linspace(.3,-.25,150)
    return c,mesh,module,args,m,variables

  def test_actual_norm_and_single_projection(self):
    for arm in (ATTENTION,MLP):
      for layer in (0,1):
        with self.subTest(arm=arm,layer=layer):
          c,mesh,module,args,m,variables=self.layer_fixture(arm,layer)
          tape={}
          names=('pre_self_attention_layer_norm','post_self_attention_layer_norm',
                 'P_loc_down','P_loc_up','mlp_address_down','mlp_address_up','W_gw','mlp_write_gate')
          def intercept(next_fun,a,k,ctx):
            y=next_fun(*a,**k)
            if ctx.method_name=='__call__' and ctx.module.name in names:
              tape.setdefault(ctx.module.name,[]).append((a[0],y))
            return y
          with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),nn.intercept_methods(intercept):
            output=module.apply(variables,*args,M_in=m,is_global=True)
          self.assertEqual(len(tape['P_loc_down']),1)
          self.assertNotIn('mlp_address_down',tape)
          a=tape['pre_self_attention_layer_norm'][1][1]
          b=tape['post_self_attention_layer_norm'][0][1]
          self.assertGreater(float(jnp.linalg.norm(a-b)),.01)
          np.testing.assert_array_equal(tape['W_gw'][0][0],a)
          expected=a if layer==0 or arm==ATTENTION else b
          np.testing.assert_array_equal(tape['P_loc_down'][0][0],expected)
          h=jax.nn.gelu(tape['P_loc_down'][0][1],approximate=True)
          np.testing.assert_array_equal(tape['P_loc_up'][0][0],h)
          if layer==1:
            np.testing.assert_array_equal(tape['mlp_address_up'][0][0],h)
            np.testing.assert_array_equal(tape['mlp_write_gate'][0][0],b)
          else:
            old=fusion.SubDecoderLayer(self.tiny(arm=PARENT),mesh,layer_inx=0,sliding_window_size=4)
            with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
              prior=old.apply(variables,*args,M_in=m,is_global=True)
            for p,q in zip(output,prior):np.testing.assert_array_equal(p,q)
    print('SHARED_WRITE_ONE_PROJECTION_CORRECT_NORM_OK',flush=True)

  def test_both_consumers_contribute_gradient(self):
    for arm in (ATTENTION,MLP):
      c,mesh,module,args,m,variables=self.layer_fixture(arm)
      def loss(params,ca,cm):
        _,captured=module.apply({'params':params},*args,M_in=m,is_global=True,
            capture_intermediates=lambda mdl,method: method=='__call__' and mdl.name in ('P_loc_up','mlp_address_up'),
            mutable=['intermediates'])
        a=captured['intermediates']['self_attention']['P_loc_up']['__call__'][0]
        b=captured['intermediates']['mlp_address_up']['__call__'][0]
        return ca*jnp.mean(a*a)+cm*jnp.mean(b*b)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        g=[nn.unbox(jax.grad(loss)(variables['params'],*weights))['self_attention']['P_loc_down']['kernel']
           for weights in ((1.,0.),(0.,1.),(1.,1.))]
      self.assertGreater(float(jnp.linalg.norm(g[0])),0)
      self.assertGreater(float(jnp.linalg.norm(g[1])),0)
      np.testing.assert_allclose(g[2],g[0]+g[1],rtol=2e-5,atol=1e-9)
    print('SHARED_WRITE_JOINT_GRADIENT_OK',flush=True)

  def test_new_arms_scan_gradients_and_causality(self):
    for arm in (ATTENTION,MLP):
      c=self.tiny(arm=arm);mesh=self.mesh_for(c);model=Transformer(c,mesh,quant=None)
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},tok,pos,tok,seg,seg,enable_dropout=False)['params']
        val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(call(p,tok)**2)))(params)
        self.assertTrue(np.isfinite(float(val)))
        self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)))
        flat=flatten_dict(nn.unbox(grad))
        self.assertFalse(any('mlp_address_down' in p for p in flat))
        for name in ('P_loc_down','P_loc_up','mlp_address_up','W_gw','mlp_write_gate'):
          self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flat.items() if name in p),0,name)
        y=call(params,tok);changed=call(params,tok.at[:,3].set(8))
        np.testing.assert_allclose(y[:,:3],changed[:,:3],rtol=3e-5,atol=3e-5)
    print('SHARED_WRITE_SCAN_FINITE_GRAD_CAUSAL_OK',flush=True)

if __name__=='__main__':unittest.main()

"""Independent HC Appendix J reference, gradients, and full Llama integration."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import residual_connections_test as ref
from layers.residual_connections import DynamicHyperConnection, mhc_update
from layers.models import Transformer
import max_utils

RUN='BamMHAMediumPropC256TruePileDHC4Tanh'

class DHCTest(unittest.TestCase):
  setUp=ref.ResidualModelTest.setUp
  tearDown=ref.ResidualModelTest.tearDown
  config=ref.MLPWriteTest.config
  tiny=ref.ResidualModelTest.tiny

  def test_paper_reference_and_gradients(self):
    cfg=self.config(RUN)
    state=jax.random.normal(jax.random.key(3),(2,3,4,8))
    module=DynamicHyperConnection(cfg,5)
    p=nn.unbox(module.init(jax.random.key(4),state)['params'])
    read,post,mix=module.apply({'params':p},state)
    np.testing.assert_array_equal(read,state[...,1,:])
    np.testing.assert_array_equal(post,jnp.ones((2,3,4)))
    np.testing.assert_array_equal(mix,jnp.broadcast_to(jnp.eye(4),(2,3,4,4)))
    # Force learned asymmetric signed matrices; detect source/destination transpose,
    # stream-shared projections, affine normalization, and post-tanh scaling order.
    p['dynamic_alpha_fn']=jax.random.normal(jax.random.key(11),(8,5))*.2
    p['dynamic_beta_fn']=jax.random.normal(jax.random.key(12),(8,))*.2
    p['dynamic_alpha_scale']=jnp.array([.7]);p['dynamic_beta_scale']=jnp.array([.8])
    p['static_alpha']=jax.random.normal(jax.random.key(13),(4,5))*.3
    p['static_beta']=jnp.array([-.2,.3,1.,-.5])
    p['router_norm']['scale']=jnp.arange(8)/8+.5
    p['router_norm']['bias']=jnp.arange(8)/20
    x=np.array(state);q=jax.tree.map(np.array,p)
    norm=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
    norm=norm*q['router_norm']['scale']+q['router_norm']['bias']
    alpha=q['static_alpha']+q['dynamic_alpha_scale']*np.tanh(norm@q['dynamic_alpha_fn'])
    beta=q['static_beta']+q['dynamic_beta_scale']*np.tanh(norm@q['dynamic_beta_fn'])
    read,post,mix=module.apply({'params':p},state)
    np.testing.assert_allclose(read,np.einsum('...n,...nd->...d',alpha[...,0],x),atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(post,beta,atol=1e-6)
    np.testing.assert_allclose(mix,alpha[...,1:],atol=1e-6)
    branch=jax.random.normal(jax.random.key(14),(2,3,8))
    expected=np.einsum('...ij,...id->...jd',alpha[...,1:],x)+beta[...,None]*np.array(branch)[...,None,:]
    np.testing.assert_allclose(mhc_update(state,branch,post,mix),expected,atol=1e-6,rtol=1e-5)
    def objective(params):
      r,b,m=module.apply({'params':params},state)
      return jnp.square(mhc_update(state,r,b,m)).sum()
    grads=jax.grad(objective)(p)
    for leaf in jax.tree.leaves(grads):
      self.assertTrue(bool(jnp.isfinite(leaf).all()))
      self.assertGreater(float(jnp.linalg.norm(leaf)),0)
    bf=state.astype(jnp.bfloat16)
    r,b,m=module.apply({'params':p},bf)
    got=mhc_update(bf,branch.astype(bf.dtype),b,m)
    self.assertEqual(got.dtype,jnp.bfloat16)
    self.assertTrue(bool(jnp.isfinite(got).all()))

  def test_full_recipe_shapes_budget_and_validation(self):
    cfg=self.config(RUN)
    self.assertEqual(cfg.mlp_dim,3195)
    self.assertFalse(cfg.mhc_learned_head)
    self.assertEqual(cfg.residual_connection,'dhc')
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=Transformer(cfg,mesh,quant=None)
    t=jax.ShapeDtypeStruct((1,4),jnp.int32)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      p=jax.eval_shape(lambda t:nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,t,t,t,decoder_segment_ids=t,enable_dropout=False)['params']),t)
    count=sum(v.size for v in jax.tree.leaves(p))
    self.assertEqual(count,432143736)
    print('DHC_FULL_PARAMETER_COUNT',count,'MLP_DIM',cfg.mlp_dim,flush=True)
    from residual_config import validate_residual_config
    for key,value in [('mhc_learned_head',True),('dhc_norm_epsilon',0),('dhc_norm_epsilon',float('nan'))]:
      with self.assertRaises(ValueError):validate_residual_config(dict(cfg.get_keys(),**{key:value}))

  def test_full_model_gradient_causality_remat_bf16(self):
    cfg=self.tiny(RUN)
    cfg.get_keys()['bam_splash_attention']=False
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None,:];mask=jnp.ones_like(tokens)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      model=Transformer(cfg,mesh,quant=None)
      p=nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},tokens,pos,tokens,mask,decoder_segment_ids=mask,enable_dropout=False)['params'])
      def forward(params,t=tokens):return model.apply({'params':params},t,pos,tokens,mask,decoder_segment_ids=mask,enable_dropout=False)[0]
      from layers.fusion import SubDecoderLayer
      from layers.residual_connections import ResidualDecoderStack
      _,captured=model.apply({'params':p},tokens,pos,tokens,mask,decoder_segment_ids=mask,
          enable_dropout=False,mutable=['intermediates'],
          capture_intermediates=lambda module,_:isinstance(module,(SubDecoderLayer,ResidualDecoderStack)))
      captured_stack=captured['intermediates']['decoder']['residual_stack']
      final_state=captured_stack['layers_1']['__call__'][0][0]
      stack_out=captured_stack['__call__'][0]
      np.testing.assert_allclose(stack_out,jnp.sum(final_state,axis=-2),atol=1e-6)
      out=forward(p);grad=jax.grad(lambda p:jnp.square(forward(p)).mean())(p)
      self.assertTrue(all(bool(jnp.isfinite(g).all()) for g in jax.tree.leaves(grad)))
      self.assertGreater(float(jnp.linalg.norm(grad['decoder']['residual_stack']['layers_0']['attn_connection']['dynamic_alpha_fn'])),0)
      changed=forward(p,tokens.at[0,-1].set(5))
      np.testing.assert_allclose(out[:,:-1],changed[:,:-1],atol=1e-6)
      cfg.get_keys()['remat_policy']='full'
      remat=Transformer(cfg,mesh,quant=None)
      actual=remat.apply({'params':p},tokens,pos,tokens,mask,decoder_segment_ids=mask,enable_dropout=False)[0]
      np.testing.assert_allclose(actual,out,atol=1e-5,rtol=1e-5)
      cfg.get_keys()['dtype']=jnp.bfloat16
      bf=Transformer(cfg,mesh,quant=None).apply({'params':p},tokens,pos,tokens,mask,decoder_segment_ids=mask,enable_dropout=False)[0]
      self.assertTrue(bool(jnp.isfinite(bf).all()))

if __name__=='__main__':unittest.main()

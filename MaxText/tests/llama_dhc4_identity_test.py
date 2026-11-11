"""DHC transport-only identity ablation: parity, intervention and integration."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import residual_connections_test as ref
from layers.residual_connections import DynamicHyperConnection, mhc_update
from layers.models import Transformer
import max_utils

PARENT='BamMHAMediumPropC256TruePileDHC4Tanh'
RUN=PARENT+'Identity'

class IdentityTest(unittest.TestCase):
  setUp=ref.ResidualModelTest.setUp
  tearDown=ref.ResidualModelTest.tearDown
  config=ref.MLPWriteTest.config
  tiny=ref.ResidualModelTest.tiny

  def test_only_transport_changes_when_learned(self):
    base=self.config(PARENT);cfg=self.config(RUN)
    x=jax.random.normal(jax.random.key(1),(2,3,4,8))
    a=DynamicHyperConnection(base,3);b=DynamicHyperConnection(cfg,3)
    p=nn.unbox(a.init(jax.random.key(2),x)['params'])
    q=nn.unbox(b.init(jax.random.key(2),x)['params'])
    for u,v in zip(jax.tree.leaves(p),jax.tree.leaves(q)):
      np.testing.assert_array_equal(u,v)
    for u,v in zip(a.apply({'params':p},x),b.apply({'params':q},x)):
      np.testing.assert_array_equal(u,v)
    # Learned signed, asymmetric transport; retained read and post must be exact.
    p['dynamic_alpha_fn']=jax.random.normal(jax.random.key(3),(8,5))*.3
    p['dynamic_beta_fn']=jax.random.normal(jax.random.key(4),(8,))*.2
    p['dynamic_alpha_scale']=jnp.array([.7])
    p['static_alpha']=jax.random.normal(jax.random.key(5),(4,5))
    p['static_beta']=jnp.array([-.2,.3,.7,1.1])
    ra,pa,ma=a.apply({'params':p},x)
    rb,pb,mb=b.apply({'params':p},x)
    np.testing.assert_array_equal(ra,rb);np.testing.assert_array_equal(pa,pb)
    np.testing.assert_array_equal(mb,np.broadcast_to(np.eye(4),(2,3,4,4)))
    self.assertGreater(float(jnp.linalg.norm(ma-mb)),.1)
    branch=jax.random.normal(jax.random.key(6),(2,3,8))
    np.testing.assert_allclose(mhc_update(x,branch,pb,mb),np.array(x)+np.array(pb)[...,None]*np.array(branch)[...,None,:],rtol=1e-6,atol=1e-6)
    def objective(q):
      r,p,m=b.apply({'params':q},x)
      return jnp.square(mhc_update(x,r,p,m)).sum()
    g=jax.grad(objective)(p)
    for key in ('static_alpha','dynamic_alpha_fn'):
      np.testing.assert_array_equal(g[key][...,1:],jnp.zeros_like(g[key][...,1:]))
      self.assertGreater(float(jnp.linalg.norm(g[key][...,0])),0)
    self.assertGreater(float(jnp.linalg.norm(g['dynamic_beta_fn'])),0)
    self.assertTrue(all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(g)))
    bf=x.astype(jnp.bfloat16);r,p,m=b.apply({'params':p},bf)
    self.assertEqual(mhc_update(bf,r,p,m).dtype,jnp.bfloat16)

  def test_full_shape_budget_and_config_parity(self):
    a=self.config(PARENT);b=self.config(RUN)
    allowed={'model_name','run_name','exp_class','compare_runs','mhc_identity_mixing'}
    # check_exp_runtime_config independently verifies every effective sealed attr.
    for key in ('mlp_dim','emb_dim','num_decoder_layers','mhc_num_streams','mhc_learned_head','dhc_norm_epsilon','scan_layers','learning_rate','steps','init_weights','record_training_health_metrics','record_internal_nn_metrics'):
      if key in a.get_keys(): self.assertEqual(a.get_keys()[key],b.get_keys()[key],key)
    self.assertTrue(b.mhc_identity_mixing);self.assertEqual(b.mlp_dim,3195)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(b),b.mesh_axes)
    model=Transformer(b,mesh,quant=None);t=jax.ShapeDtypeStruct((1,4),jnp.int32)
    with mesh,nn.partitioning.axis_rules(b.logical_axis_rules):
      p=jax.eval_shape(lambda t:nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,t,t,t,decoder_segment_ids=t,enable_dropout=False)['params']),t)
    self.assertEqual(sum(v.size for v in jax.tree.leaves(p)),432143736)

  def test_initial_full_model_parity_gradient_and_remat(self):
    a=self.tiny(PARENT);b=self.tiny(RUN)
    a.get_keys()['bam_splash_attention']=False;b.get_keys()['bam_splash_attention']=False
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(b),b.mesh_axes)
    t=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None,:];mask=jnp.ones_like(t)
    with mesh,nn.partitioning.axis_rules(b.logical_axis_rules):
      ma=Transformer(a,mesh,quant=None);mb=Transformer(b,mesh,quant=None)
      def init(m):return nn.unbox(m.init({'params':jax.random.key(8),'dropout':jax.random.key(8)},t,pos,t,mask,decoder_segment_ids=mask,enable_dropout=False)['params'])
      pa=init(ma);pb=init(mb)
      self.assertEqual(jax.tree.structure(pa),jax.tree.structure(pb))
      for u,v in zip(jax.tree.leaves(pa),jax.tree.leaves(pb)):np.testing.assert_array_equal(u,v)
      def forward(m,p,t=t):return m.apply({'params':p},t,pos,t,mask,decoder_segment_ids=mask,enable_dropout=False)[0]
      out=forward(mb,pb);np.testing.assert_array_equal(out,forward(ma,pa))
      grad=jax.grad(lambda p:jnp.square(forward(mb,p)).mean())(pb)
      self.assertTrue(all(bool(jnp.isfinite(v).all()) for v in jax.tree.leaves(grad)))
      g=grad['decoder']['residual_stack']['layers_0']['attn_connection']['dynamic_alpha_fn']
      np.testing.assert_array_equal(g[...,1:],jnp.zeros_like(g[...,1:]));self.assertGreater(float(jnp.linalg.norm(g[...,0])),0)
      np.testing.assert_allclose(out[:,:-1],forward(mb,pb,t.at[0,-1].set(5))[:,:-1],atol=1e-6)
      b.get_keys()['remat_policy']='full'
      np.testing.assert_allclose(out,forward(Transformer(b,mesh,quant=None),pb),atol=1e-5,rtol=1e-5)
      b.get_keys()['dtype']=jnp.bfloat16
      self.assertTrue(bool(jnp.isfinite(forward(Transformer(b,mesh,quant=None),pb)).all()))

if __name__=='__main__':unittest.main()

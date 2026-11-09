"""Initialization-only Llama mHC ablation: parameter roles and gradients."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import residual_connections_test as ref
PARENT = 'BamMHAMediumPropC256TruePilemHC'
VARIANT = PARENT + 'PythiaInit'
class LlamaMHCPythiaInitTest(unittest.TestCase):
  setUp = ref.ResidualModelTest.setUp
  tearDown = ref.ResidualModelTest.tearDown
  config = ref.MLPWriteTest.config
  tiny = ref.ResidualModelTest.tiny
  def test_formal_config_changes_only_initialization(self):
    a, b = self.config(PARENT), self.config(VARIANT)
    delta = {k for k in a.get_keys() if str(a.get_keys()[k]) != str(b.get_keys()[k])}
    self.assertFalse(delta - {'model_name','exp_class','compare_runs','init_method'}, delta)
    self.assertEqual((a.init_method,b.init_method), ('006normal','pythia'))
    self.assertEqual((b.mlp_dim,b.emb_dim,b.learning_rate), (3133,1200,.0003))
    self.assertFalse(b.scan_layers)
    self.assertGreater(b.tensorboard_gcs_rotate_period, 0)
  def init_model(self, name):
    c = self.tiny(name)
    c.get_keys().update(emb_dim=128,mlp_dim=256,head_dim=16,num_query_heads=8,num_kv_heads=8,vocab_size=256)
    mesh = jax.sharding.Mesh(ref.max_utils.create_device_mesh(c),c.mesh_axes)
    m = ref.Transformer(c,mesh,quant=None)
    t = jnp.array([[1,2,3,4]],jnp.int32)
    args = (t,jnp.arange(4)[None],t,jnp.ones_like(t),jnp.ones_like(t))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p = m.init(jax.random.key(19),*args,enable_dropout=False)['params']
    return c,m,mesh,p,args
  def test_actual_parameter_roles_and_unchanged_router(self):
    _,_,_,pa,_ = self.init_model(PARENT)
    b,_,_,pb,_ = self.init_model(VARIANT)
    fa,fb = flatten_dict(nn.unbox(pa)),flatten_dict(nn.unbox(pb))
    self.assertEqual(fa.keys(),fb.keys())
    roles=set()
    for path,value in fb.items():
      core = path[-1] in ('embedding','logits_dense') or (path[-1]=='kernel' and path[-2] in ('query','key','value','out','wi_0','wi_1','wo'))
      if core:
        role=path[-2] if path[-1]=='kernel' else path[-1]
        std=2/b.num_decoder_layers/np.sqrt(b.emb_dim) if role in ('out','wo') else np.sqrt(2/(5*b.emb_dim))
        np.testing.assert_allclose(value,fa[path]*(std/.006),rtol=2e-6,atol=1e-7,err_msg=str(path))
        roles.add(role)
      else: np.testing.assert_array_equal(value,fa[path],err_msg=str(path))
    self.assertEqual(roles,{'embedding','logits_dense','query','key','value','out','wi_0','wi_1','wo'})
  def test_forward_and_all_parameter_gradients_finite(self):
    c,m,mesh,p,args=self.init_model(VARIANT)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      def loss(params):
        logits=m.apply({'params':params},*args,enable_dropout=False)[0]
        return -jnp.take_along_axis(jax.nn.log_softmax(logits),args[0][...,None],-1).mean()
      value,grad=jax.value_and_grad(loss)(p)
    self.assertTrue(np.isfinite(value))
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(nn.unbox(grad))))
    self.assertGreater(sum(float(jnp.sum(x*x)) for path,x in flatten_dict(nn.unbox(grad)).items() if path[-1]=='router_kernel'),0)
if __name__=='__main__': unittest.main()

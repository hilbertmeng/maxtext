"""Independent NoPE Q/K scalars: exact product semantics and native/Pallas gradients."""
import contextlib, math, unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,train,train_compile
from bam_config import validate_bam_config
from bam_common_all_read_gamma_test import CommonAllReadGammaTest,H
from bam_local_o_dynamic_norm_scale_test import LocalODynamicNormScaleTest
RUN=H.removesuffix('TruePile')+'IndependentQKReadScaleTruePile'
class IndependentQKScaleTest(CommonAllReadGammaTest):
  def test_formal_budget_validation_health_and_weight_decay(self):
    parent=self.config(H);cfg=self.config(RUN)
    self.assertEqual({k for k,v in cfg.get_keys().items() if v!=parent.get_keys()[k]}, {'exp_class','model_name','compare_runs','bam_independent_qk_read_scale'})
    self.assertEqual(cfg.compare_runs,[H]);self.assertFalse(cfg.qk_norm);self.assertFalse(cfg.bam_common_read_scale_qk)
    cfg.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);args,*_=train_compile.get_shaped_inputs(mesh,cfg);flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112246)
    for name in ['local_q_read_scale','local_k_read_scale']:
      self.assertEqual(sum(math.prod(v.shape) for k,v in flat.items() if k[-1]==name),18)
    for flags in [{'bam_concat_qk':False},{'bam_common_read_scale_qk':True}]:
      with self.assertRaisesRegex(ValueError,'Independent Q/K'):validate_bam_config(dict(cfg.get_keys(),**flags))
    tree={'params':{'local_q_read_scale':jnp.ones(1),'local_k_read_scale':jnp.ones(1)}}
    self.assertTrue(all(v==0 for v in jax.tree.leaves(train.get_wd_tree(cfg,tree))))
    metrics=train.compute_params_norm(tree,cfg)
    self.assertEqual(float(metrics['parameter_values/local_q_read_scale']),1.)
    self.assertEqual(float(metrics['parameter_values/local_k_read_scale']),1.)

  def test_exact_q_k_scopes_product_and_nonzero_scalar_gradients(self):
    cfg,mesh,module,p,args,matrix=LocalODynamicNormScaleTest.fixture(self,run=RUN)
    flat=flatten_dict(p);qp=next(k for k in flat if k[-1]=='local_q_read_scale');kp=next(k for k in flat if k[-1]=='local_k_read_scale');vp=next(k for k in flat if k[-1]=='common_vo_read_scale')
    np.testing.assert_array_equal(flat[qp],[1.]);np.testing.assert_array_equal(flat[kp],[1.])
    captured=[]
    def intercept(next_fun,args,kwargs,context):
      if context.method_name=='_query_chunk_op':captured.append((args[0],args[1]))
      return next_fun(*args,**kwargs)
    def call(q,k,vo=.37):
      f=dict(flat);f[qp]=jnp.reshape(q,(1,));f[kp]=jnp.reshape(k,(1,));f[vp]=jnp.reshape(jnp.float32(vo),(1,))
      return module.apply(unflatten_dict(f),*args,M_in=matrix,deterministic=True)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      with nn.intercept_methods(intercept):base=call(jnp.float32(1),jnp.float32(1))
      q0,k0=captured[-1]
      for q,k in [(.7,.8),(2.,.5),(-.7,.8),(0.,1.)]:
        with nn.intercept_methods(intercept):actual=call(jnp.float32(q),jnp.float32(k))
        qa,ka=captured[-1]
        np.testing.assert_allclose(qa[...,:75],q*q0[...,:75],rtol=2e-5,atol=2e-6);np.testing.assert_allclose(ka[...,:75],k*k0[...,:75],rtol=2e-5,atol=2e-6)
        np.testing.assert_array_equal(qa[...,75:],q0[...,75:]);np.testing.assert_array_equal(ka[...,75:],k0[...,75:])
        if q*k==1.:
          for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(base)):np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
      target=jax.random.normal(jax.random.key(33),base[0].shape)
      dq,dk=jax.grad(lambda q,k:jnp.sum(call(q,k)[0]*target),argnums=(0,1))(jnp.float32(1),jnp.float32(1))
      self.assertGreater(abs(float(dq)),1e-8);self.assertGreater(abs(float(dk)),1e-8)
      np.testing.assert_allclose(dq,dk,rtol=2e-4,atol=2e-6)

  def test_independent_native_pallas_all_gradients(self):self.check_model(RUN)
if __name__=='__main__':unittest.main()

"""The compensated seed preserves its parent while leaving the vector embedding larger."""
import contextlib
import io
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig
from layers import models, initializers
from pythia_initialization_test import PythiaInitializationTest

PARENT='BamPythiaMediumPropT2048SerialMLPBamInitFix'
EXP='BamPythiaMediumPropT2048SerialMLPPythiaEmbeddingScaledWrite'

class EmbeddingWriteScaleTest(PythiaInitializationTest):
  def test_only_embedding_parameters_change(self):
    a,b=[self.bam_config(n) for n in (PARENT,EXP)]
    allowed={'exp_class','model_name','run_name','compare_runs','token_embedding_init_method','bam_embedding_write_input_scale'}
    differing={k for k in a.get_keys() if str(a.get_keys()[k])!=str(b.get_keys()[k])}
    self.assertLessEqual(differing,allowed)
    ta,tb=[flatten_dict(nn.unbox(self.init(c)[2])) for c in (a,b)]
    self.assertEqual(ta.keys(),tb.keys())
    changed=[]
    for path in ta:
      if path[-1]=='embedding':
        np.testing.assert_allclose(tb[path],ta[path]*(math.sqrt(2/(5*b.emb_dim))/.006),rtol=2e-6,atol=1e-8)
        changed.append(path)
      else:np.testing.assert_array_equal(ta[path],tb[path],err_msg=str(path))
    self.assertEqual(len(changed),1)
    self.assertEqual(b.bam_write_rms_epsilon,1e-6)
    self.assertFalse(b.pythia_parallel_residual)

  def test_full_budget_and_flags(self):
    c=self.bam_config(EXP,full=True);m,mesh=self.model(c)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):
      p=jax.eval_shape(lambda k:m.init(k,*self.inputs(),enable_dropout=False)['params'],jax.random.key(19))
    count=sum(math.prod(v.shape) for v in flatten_dict(nn.unbox(p)).values())
    self.assertEqual(count,432356690)
    self.assertEqual(c.mlp_dim_by_block,[5832,5643,5832])
    self.assertEqual(c.bam_embedding_write_input_scale,.006/math.sqrt(2/(5*c.emb_dim)))
    self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertFalse(c.qk_norm)
    self.assertFalse(c.record_training_health_metrics)
    self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
    print('COMPENSATED_EMBEDDING_PARAMS',count,flush=True)

  def test_seed_compensation_and_default_identity(self):
    c=self.bam_config(EXP,full=True)
    def config(scale):return pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_embedding_write_input_scale=scale)))
    for dtype in (jnp.float32,jnp.bfloat16):
      def module(scale):return models.EmbeddingBamWrite(config(scale),16,dtype,jnp.float32,None,initializers.get_bam_init(c))
      x=jax.random.normal(jax.random.key(32),(1,32,1200))
      old=(x*.006).astype(dtype);large=(x*math.sqrt(2/(5*1200))).astype(dtype)
      base,new=module(1.),module(c.bam_embedding_write_input_scale)
      with nn.partitioning.axis_rules([]):
        p=base.init(jax.random.key(33),old)
        q=new.init(jax.random.key(33),large)
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(q)):np.testing.assert_array_equal(a,b)
        y,z=base.apply(p,old).astype(jnp.float32),new.apply(p,large).astype(jnp.float32)
        rel=float(jnp.linalg.norm(y-z)/jnp.linalg.norm(y))
        self.assertLess(rel,2e-6 if dtype==jnp.float32 else .015)
        # The scale is applied to all seed projections, once, and stays differentiable.
        manual=base.apply(p,large*jnp.asarray(c.bam_embedding_write_input_scale,dtype))
        np.testing.assert_array_equal(new.apply(p,large),manual)
        grad=jax.grad(lambda e:jnp.sum(new.apply(p,e).astype(jnp.float32)**2))(large)
        self.assertTrue(np.isfinite(grad).all());self.assertGreater(float(jnp.linalg.norm(grad)),0)
        print('COMPENSATED_SEED_REL_L2',str(dtype),rel,flush=True)
    for scale in (0.,-1.,float('nan'),float('inf')):
      with self.assertRaises(ValueError):module(scale).init(jax.random.key(33),old)

  def test_new_forward_gradients_causality(self):
    c=self.bam_config(EXP);m,mesh,p=self.init(c);args=self.inputs()
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      f=jax.jit(lambda p,t:m.apply({'params':p},t,*args[1:],enable_dropout=False)[0])
      value,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,args[0])**2)))(p)
      self.assertTrue(np.isfinite(value));self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(grad)))
      for fragment in ('embedding','W_emb_u','W_emb_v_down','M_shift_gate','wi'):
        self.assertGreater(sum(float(jnp.linalg.norm(v)) for k,v in flatten_dict(nn.unbox(grad)).items() if fragment in k),0,fragment)
      np.testing.assert_allclose(f(p,args[0])[:,:6],f(p,args[0].at[:,6:].set(17))[:,:6],atol=2e-6)

if __name__=='__main__':unittest.main()

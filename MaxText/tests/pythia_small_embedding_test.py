"""Embedding-only controls and compact health have no effect on model math."""
import contextlib
import copy
import io
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig
from layers import embedding_stem_health
from pythia_test import PythiaTest


PAIRS = [('PythiaMediumPropT2048SerialMLP','PythiaMediumPropT2048SerialMLPSmallEmbedding'),
         ('MuddPythiaMediumPropT2048SerialMLP','MuddPythiaMediumPropT2048SerialMLPSmallEmbedding')]


class SmallEmbeddingTest(unittest.TestCase):
  def setUp(self):
    self.h=PythiaTest(); self.h.setUp()
  def tearDown(self):
    self.h.tearDown()
  def config(self, name, full=False):
    opts = dict(base_mlp_dim=256,pythia_rotary_dim=4)
    if full:
      opts=dict(base_emb_dim=1200,base_num_decoder_layers=18,base_num_query_heads=16,
                base_num_kv_heads=16,head_dim=75,pythia_rotary_dim=18,vocab_size=50432,
                max_target_length=2048,max_prefill_predict_length=2048)
    with contextlib.redirect_stdout(io.StringIO()):
      return self.h.config(exp_class=name,**opts)

  def test_only_embedding_changes_and_full_budgets(self):
    for parent,child in PAIRS:
      a,b=[self.config(n) for n in (parent,child)]
      allowed={'model_name','exp_class','run_name','compare_runs','token_embedding_init_method',
               'record_embedding_stem_health_metrics'}
      differing={k for k in a.get_keys() if str(a.get_keys()[k])!=str(b.get_keys()[k])}
      self.assertLessEqual(differing,allowed)
      with contextlib.redirect_stdout(io.StringIO()):
        pp,qq=[flatten_dict(nn.unbox(self.h.init(c)[2])) for c in (a,b)]
      self.assertEqual(pp.keys(),qq.keys())
      for k in pp:
        if k[-1]=='embedding':
          np.testing.assert_allclose(qq[k],pp[k]*(.006/math.sqrt(2/(5*b.emb_dim))),rtol=2e-6,atol=1e-8)
        else:np.testing.assert_array_equal(pp[k],qq[k],err_msg=str(k))
      counts=[]
      for name in (parent,child):
        c=self.config(name,True);m,mesh=self.h.model(c)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):
          p=jax.eval_shape(lambda key:m.init(key,*self.h.inputs(),enable_dropout=False)['params'],jax.random.key(19))
        counts.append(sum(math.prod(v.shape) for v in flatten_dict(nn.unbox(p)).values()))
        self.assertEqual(c.tensorboard_gcs_rotate_period,500)
        self.assertFalse(c.qk_norm);self.assertFalse(c.record_training_health_metrics)
        self.assertFalse(c.pythia_parallel_residual)
      self.assertEqual(*counts)
      print('SMALL_EMBEDDING_PARAMS',child,counts[1],flush=True)

  def test_health_forward_gradient_parity_and_actual_updates(self):
    for _,name in PAIRS:
      c=self.config(name)
      off=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),record_embedding_stem_health_metrics=False)))
      with contextlib.redirect_stdout(io.StringIO()):m,mesh,p=self.h.init(c)
      baseline,_=self.h.model(off);batch=self.h.inputs()
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        def run(model,params):return model.apply({'params':params},*batch,enable_dropout=False,mutable=['intermediates'])
        out,inter=run(m,p);expected,_=run(baseline,p)
        np.testing.assert_allclose(out[0],expected[0],atol=1e-6)
        g1=jax.jit(jax.grad(lambda z:jnp.mean(run(m,z)[0][0])))(p)
        g2=jax.jit(jax.grad(lambda z:jnp.mean(run(baseline,z)[0][0])))(p)
        for x,y in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)):
          np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
        plain=nn.unbox(p);new=copy.deepcopy(plain)
        new['token_embedder']['embedding']=plain['token_embedder']['embedding']+1e-4
        fun=jax.jit(lambda step:embedding_stem_health.metrics(inter,{'params':plain},{'params':new},step,100))
        sampled=fun(jnp.array(100));unsampled=fun(jnp.array(101))
        self.assertEqual(len(sampled),9)
        self.assertTrue(all(np.isfinite(v) for v in sampled.values()))
        self.assertAlmostEqual(float(sampled['embedding_stem/embedding_update_rms']),1e-4,places=7)
        self.assertEqual(float(unsampled['embedding_stem/embedding_update_rms']),0.)
        self.assertGreater(float(sampled['embedding_stem/mlp_over_embedding']),0.)

  def test_writer_uses_buffered_step_for_health_cadence(self):
    import train
    c=SimpleNamespace(record_embedding_stem_health_metrics=True,embedding_stem_health_period=100,
                      enable_tensorboard=True,metrics_file='',gcs_metrics=False)
    metrics={'scalar':{'learning/loss':1.,'embedding_stem/mlp_rms':2.},'scalars':{}}
    for step,wanted in [(0,True),(1,False),(100,True)]:
      train.clear_buffered_metrics()
      with mock.patch.object(train,'write_metrics_to_tensorboard') as writer:
        train.write_metrics(None,None,None,metrics,step,c)
        train.write_metrics(None,None,None,metrics,step+1,c)
        self.assertEqual('embedding_stem/mlp_rms' in writer.call_args.args[1]['scalar'],wanted)
        self.assertEqual(writer.call_args.args[2],step)
    train.clear_buffered_metrics()


if __name__=='__main__':unittest.main()

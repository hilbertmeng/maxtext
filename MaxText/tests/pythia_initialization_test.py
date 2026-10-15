"""Check the complete initialization chain and paired serial GELU controls."""
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
from layers import initializers, models
from pythia_test import PythiaTest

PARENT = 'BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256SerialMLP'
ARMS = ['BamPythiaMediumPropT2048SerialMLPBamInitFix',
        'BamPythiaMediumPropT2048SerialMLPOriginalInit']

class PythiaInitializationTest(PythiaTest):
  def bam_config(self, arm, full=False):
    c = self.config(exp_class=arm, base_emb_dim=1200 if full else 300,
        base_num_decoder_layers=18 if full else 6,
        base_num_query_heads=16 if full else 4, base_num_kv_heads=16 if full else 4,
        head_dim=75, pythia_rotary_dim=18, vocab_size=50432 if full else 32,
        scan_layers=False)
    if full:
      return c
    return pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        emb_bam_num_head=4, mlp_dim_by_block=[256,224,256],
        bam_local_full_m_read_bottleneck_dim=64, bam_write_v_bottleneck_dim=64,
        bam_mlp_write_address_rank=64, bam_splash_attention=False,
        bam_layer_modes=c.bam_layer_modes[:6])))

  def test_parameter_roles_and_unchanged_semantic_initializers(self):
    configs = [self.bam_config(a) for a in [PARENT]+ARMS]
    trees = [flatten_dict(nn.unbox(self.init(c)[2])) for c in configs]
    for tree in trees[1:]:
      self.assertEqual(set(tree),set(trees[0]))
    small = math.sqrt(2/(5*configs[0].emb_dim))
    wang = 2/configs[0].num_decoder_layers/math.sqrt(configs[0].emb_dim)
    changed = {'embedding':0,'bam':0,'core':0}
    for path, base in trees[0].items():
      fix, old = trees[1][path], trees[2][path]
      self.assertEqual(base.shape,fix.shape)
      self.assertEqual(base.shape,old.shape)
      embedding = path[-1]=='embedding'
      kernel = path[-1]=='kernel'
      core = (kernel and path[-2] in ('query','key','value','out','wi','wo')) or path[-1]=='logits_dense'
      bam = kernel and not core
      # Every semantic zero kernel, gate bias and normalization parameter stays exact.
      if (embedding or bam) and np.any(np.asarray(base)):
        np.testing.assert_allclose(fix,base*(.006/small),atol=1e-8,rtol=2e-6,err_msg=str(path))
        changed['embedding' if embedding else 'bam']+=1
      else:
        np.testing.assert_array_equal(fix,base,err_msg=str(path))
      if core:
        target = wang if path[-2] in ('out','wo') else small
        np.testing.assert_allclose(old,base*(.006/target),atol=1e-8,rtol=2e-6,err_msg=str(path))
        changed['core']+=1
      else:
        np.testing.assert_array_equal(old,fix,err_msg=str(path))
    self.assertTrue(all(changed.values()),changed)
    for c in configs[1:]:
      self.assertFalse(c.pythia_parallel_residual)
      self.assertEqual(c.mlp_activations,['gelu_exact'])
      self.assertEqual(c.normalization_layer_epsilon,1e-5)
      self.assertEqual(c.bam_read_key_epsilon,1e-4)
      self.assertEqual(initializers.get_bam_write_epsilon(c),1e-6)
      self.assertFalse(c.record_training_health_metrics)

  def test_full_size_parameter_budget(self):
    reference = None
    for arm in [PARENT]+ARMS:
      c=self.bam_config(arm,full=True);m,mesh=self.model(c)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), contextlib.redirect_stdout(io.StringIO()):
        p=jax.eval_shape(lambda k:m.init(k,*self.inputs(),enable_dropout=False)['params'],jax.random.key(19))
      shapes={k:(v.shape,v.dtype) for k,v in flatten_dict(nn.unbox(p)).items()}
      self.assertEqual(sum(math.prod(v[0]) for v in shapes.values()),432356690)
      self.assertEqual(c.mlp_dim_by_block,[5832,5643,5832])
      if reference is None:reference=shapes
      self.assertEqual(shapes,reference)
      print('INIT_ARM_PARAMS',arm,432356690,flush=True)

  def test_first_matrix_reproduces_original_scale(self):
    # Actual full-width seed module, paired Gaussian embedding rows and parameter RNG.
    c=self.bam_config(ARMS[0],full=True)
    cfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        dtype=jnp.bfloat16, normalization_layer_epsilon=1e-5)))
    legacy=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),
        normalization_layer_epsilon=1e-6,bam_write_rms_epsilon=None)))
    x=(jax.random.normal(jax.random.key(19),(1,32,1200))*.006).astype(jnp.bfloat16)
    def seed(config):
      return models.EmbeddingBamWrite(config,16,jnp.bfloat16,jnp.float32,None,
          initializers.get_bam_init(config))
    m,old=seed(cfg),seed(legacy)
    with nn.partitioning.axis_rules([]):
      p=m.init(jax.random.key(119),x)
      a,b=m.apply(p,x),old.apply(p,x)
    np.testing.assert_array_equal(a,b)
    rms=float(jnp.sqrt(jnp.mean(a.astype(jnp.float32)**2)))
    self.assertGreater(rms,.01);self.assertLess(rms,.03)
    print('REPAIRED_INITIAL_M_RMS',rms,flush=True)

  def test_forward_gradients_and_causality(self):
    for arm in ARMS:
      c=self.bam_config(arm);m,mesh,p=self.init(c);args=self.inputs()
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        f=jax.jit(lambda p,t:m.apply({'params':p},t,*args[1:],enable_dropout=False)[0])
        loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,args[0])**2)))(p)
        self.assertTrue(np.isfinite(loss))
        flat=flatten_dict(nn.unbox(g))
        self.assertTrue(all(np.isfinite(a).all() for a in flat.values()))
        for fragment in ('W_emb_u','full_m_read_down','mlp_address_down','M_shift_gate','wi'):
          values=[v for k,v in flat.items() if fragment in k]
          self.assertTrue(values,fragment)
          self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in values),0,fragment)
        np.testing.assert_allclose(f(p,args[0])[:,:6],f(p,args[0].at[:,6:].set(17))[:,:6],atol=2e-6)

  def test_health_option_is_wired(self):
    import ast
    from pathlib import Path
    tree=ast.parse(Path('MaxText/train.py').read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
    calls=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='compute_params_norm']
    self.assertEqual(len(calls),2)
    guards=[n for n in ast.walk(train) if isinstance(n,ast.If)
            and 'record_training_health_metrics' in ast.unparse(n.test)]
    self.assertEqual(len(guards),1)
    self.assertTrue(all(call in list(ast.walk(guards[0])) for call in calls))

if __name__=='__main__':unittest.main()

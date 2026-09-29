"""Focused coverage for plain-JAX matrix-read normalization variants."""
import contextlib
import io
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest, EXP
from layers import rmt

MED = 'RMTMediumPropT4096TruePileAllLocalK48EmbedUnembedDirect32NoO'

class ReadNormTest(XLPropTest):
  def test_full_parameter_trees(self):
    for base, count in [(MED,431773472),(EXP,1432430680)]:
      trees=[]
      for suffix in ['', 'QKPreNorm', 'MPreNorm']:
        cfg=self.config(base+suffix)
        assert cfg.rmt_vector_pre_norm and not cfg.rmt_block_scan
        assert cfg.DATASET_VARIANT == 'truepile4096'
        for key in ('rmt_pallas_write','rmt_fused_attention_read','rmt_fused_write_mlp_read'):
          assert not cfg.get_keys().get(key,False)
        model,args=self.model_args(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
          tree=jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1))
        self.assertEqual(sum(math.prod(p.shape) for p in jax.tree.leaves(tree)),count)
        trees.append(jax.tree.map(lambda p:(p.shape,str(p.dtype)),tree))
      self.assertEqual(trees[0],trees[1]);self.assertEqual(trees[0],trees[2])

  def test_scanned_forward_gradient_and_health(self):
    for suffix in ['QKPreNorm','MPreNorm']:
      cfg=self.config(EXP+suffix,base_num_decoder_layers=2,base_emb_dim=640,
                      head_dim=32,base_mlp_dim=128,vocab_size=128)
      cfg.get_keys().update(dtype=jnp.float32,rmt_record_dynamic_health=True)
      model,args=self.model_args(cfg)
      with contextlib.redirect_stdout(io.StringIO()):
        params=nn.unbox(model.init(jax.random.key(2),**args)['params'])
        def objective(p):return jnp.sum(model.apply({'params':p},**args)[0])
        value,gradient=jax.jit(jax.value_and_grad(objective))(params)
      self.assertTrue(np.isfinite(value))
      self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves(gradient)))
      for module in ['dynamic_embedding_write','dynamic_unembedding_read']:
        self.assertGreater(sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(gradient['decoder'][module])),0.)
      if suffix=='QKPreNorm':
        cfg.get_keys().update(rmt_matrix_read_norm='none',rmt_probe_qk_matrix_rms=True)
        with contextlib.redirect_stdout(io.StringIO()):
          expected,expected_grad=jax.jit(jax.value_and_grad(objective))(params)
        np.testing.assert_allclose(value,expected,rtol=2e-5,atol=2e-5)
        for a,b in zip(jax.tree.leaves(gradient),jax.tree.leaves(expected_grad)):
          # Contraction order differs; compare normwise to avoid cancellation
          # near individual zero-valued gradient coordinates.
          self.assertLess(float(jnp.linalg.norm(a-b)),
                          3e-5 * max(1.,float(jnp.linalg.norm(b))))
      jax.clear_caches()

  def test_matrix_norm_axes_and_scale(self):
    m=jax.random.normal(jax.random.key(4),(2,3,48,75))*4
    out=rmt.matrix_read_rms_norm(m,1e-6)
    np.testing.assert_allclose(jnp.mean(out**2,axis=(-2,-1)),1,rtol=1e-6)
    self.assertEqual(out.shape,m.shape)

if __name__=='__main__':
  unittest.main(defaultTest='ReadNormTest')

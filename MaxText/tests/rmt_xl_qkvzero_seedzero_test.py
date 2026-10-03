"""XL zero-init transfer: budget, carry statistics, gradients and retention."""
import contextlib
import io
import math
import tempfile
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import rmt
from tests import rmt_xlprop_test

EXP='RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOSharedWriteNormQKVZeroInitEmbedSeedZero'

class XLSeedZeroTest(unittest.TestCase):
  config=rmt_xlprop_test.XLPropTest.config
  model_args=rmt_xlprop_test.XLPropTest.model_args

  def test_full_config_and_shapes(self):
    cfg=self.config(EXP)
    self.assertEqual(cfg.rmt_matrix_read_norm,'none')
    self.assertFalse(cfg.rmt_matrix_read_learned_scale)
    self.assertFalse(cfg.rmt_unembedding_vector_skip)
    self.assertEqual(cfg.rmt_dynamic_write_bottleneck_dim,384)
    self.assertEqual(cfg.rmt_carry_health_layers,[1,7,14,21,25,27])
    self.assertEqual(cfg.base_mlp_dim,6643)
    self.assertEqual(cfg.keep_period,2000)
    self.assertEqual(cfg.max_to_keep,2)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    print('XL_PARAMS',sum(math.prod(v.shape) for v in jax.tree.leaves(tree)))
    layers=tree['decoder']['layers']
    self.assertNotIn('attn_norm',layers)
    self.assertNotIn('mlp_norm',layers)
    self.assertIn('final_matrix_norm',tree['decoder'])
    self.assertIn('unembedding_vector_norm',tree['decoder'])

  def test_global_carry_statistics(self):
    # Different constant values across batch distinguish global from per-example means.
    m=jnp.array([1.,-1.]).reshape(2,1,1,1)*jnp.ones((2,3,2,4))
    np.testing.assert_allclose(rmt.carry_shared_health(m),[0.,1.])
    np.testing.assert_allclose(rmt.carry_shared_health(jnp.ones_like(m)*3),[1.,3.])
    np.testing.assert_allclose(rmt.carry_shared_health(jnp.zeros_like(m)),[0.,0.])
    y=m+1
    np.testing.assert_allclose(rmt.carry_shared_health(y),[.5,np.sqrt(2)],rtol=1e-6)

  def test_forward_grad_and_selected_layer_export(self):
    cfg=self.config(EXP,base_num_decoder_layers=2,base_emb_dim=640,head_dim=32,
                    base_mlp_dim=128,vocab_size=128)
    cfg.get_keys().update(dtype=jnp.float32,rmt_carry_health_layers=[1])
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(3),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.mean(out[0]),aux
      (value,aux),grad=jax.jit(jax.value_and_grad(loss,has_aux=True))(params)
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree.leaves((value,aux,grad))))
    stats=aux['intermediates']['decoder']['layers']['rmt_carry_health'][0]
    self.assertEqual(stats.shape,(2,2))
    np.testing.assert_array_equal(stats[0],0.)
    self.assertGreater(float(stats[1,1]),0.)
    self.assertGreaterEqual(float(stats[1,0]),0.)
    self.assertLessEqual(float(stats[1,0]),1.00001)
    layers=params['decoder']['layers']
    np.testing.assert_array_equal(layers['qkv_key'],0.)
    from train import record_rmt_dynamic_health_metrics, compute_params_norm
    metrics={'scalar':{}}
    record_rmt_dynamic_health_metrics(metrics,aux,cfg)
    self.assertIn('rmt/carry/layer_001/output_raw_rms',metrics['scalar'])
    norms=compute_params_norm({'params':grad},cfg,prefix='raw_grads')
    self.assertTrue(any('dynamic_embedding_write/address_up_bias' in k for k in norms))
    self.assertTrue(any('lm_head/' in k for k in norms))

if __name__=='__main__':unittest.main()

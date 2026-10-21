"""SoftCapSiLU gate law, full semantic scope, unchanged budget, causal gradients."""
import math
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import bam_gates
from layers.models import Transformer

PARENT = 'BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateTruePile'
CHILD = 'BamMediumPropMShiftReadG050FullQKNormNoQKStaticGateSoftCapSiLUTruePile'


class SoftCapGateTest(MLPWriteTest, unittest.TestCase):
  def mesh(self, c):
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)), c.mesh_axes)

  def test_law_bias_and_stronger_slope(self):
    f = bam_gates.softcap_silu
    x = jnp.linspace(-10., 10., 1001)
    np.testing.assert_allclose(f(x), 2*jnp.tanh(x*jax.nn.sigmoid(x)/2), atol=2e-7)
    self.assertEqual(float(f(jnp.array(0.))), 0.)
    self.assertLess(float(jnp.min(f(x))), -.27)
    self.assertLess(float(jnp.max(f(x))), 2.)
    for opening, slope in ((.05, .54731895), (.1, .58970894)):
      b = bam_gates.bias_for_opening(opening, 'softcap_silu')
      self.assertAlmostEqual(float(f(jnp.array(b))), opening, places=6)
      self.assertAlmostEqual(float(jax.grad(f)(jnp.array(b))), slope, places=6)
      self.assertGreater(slope, 6*opening*(1-opening))
    b = bam_gates.bias_for_opening(.1, 'sigmoid')
    self.assertEqual(b, math.log(.1/.9))
    for dtype in (jnp.float32, jnp.bfloat16):
      y=f(x.astype(dtype));self.assertEqual(y.dtype,dtype)
      self.assertTrue(np.isfinite(np.asarray(y,dtype=np.float32)).all())

  def test_formal_budget_and_controls(self):
    for exp in (PARENT, CHILD):
      c=self.config(exp); args,*_=train_compile.get_shaped_inputs(self.mesh(c),c)
      flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()),431422604)
      self.assertEqual(c.mlp_dim_by_block,[3802,3676,3802])
      self.assertFalse(c.bam_qk_static_read_gate)
      self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat and c.scan_layers)
      self.assertEqual(c.bam_m_shift_init,.5)
      self.assertEqual(c.bam_static_read_gate_init,.99)
      self.assertEqual(c.bam_read_gate_init,.05);self.assertEqual(c.bam_write_eps,.1)
      self.assertEqual(c.bam_embedding_write_eps,.1)
      self.assertFalse(c.record_training_health_metrics or c.bam_record_concat_health or c.bam_record_m_shift_health)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertEqual(c.bam_read_gate_activation,'softcap_silu')
    self.assertEqual(c.bam_write_gate_activation,'softcap_silu')
    print('SOFTCAP_GATE_BUDGET_OK 431422604 delta=0 W_Q=0',flush=True)

  def test_seeded_tree_scope_gradients_and_causality(self):
    c=self.config(PARENT,bam_splash_attention=False,base_emb_dim=150,base_num_query_heads=2,
                  base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,
                  dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
                        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
                        bam_layer_modes=c.bam_layer_modes[:3],mlp_dim_by_block=[32,24,32])
    mesh=self.mesh(c);tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      oldmodel=Transformer(c,mesh,quant=None)
      old=nn.unbox(oldmodel.init(jax.random.key(24),tok,pos,tok,mask,mask,enable_dropout=False)['params'])
      c.get_keys().update(bam_read_gate_activation='softcap_silu',bam_write_gate_activation='softcap_silu')
      model=Transformer(c,mesh,quant=None)
      new=nn.unbox(model.init(jax.random.key(24),tok,pos,tok,mask,mask,enable_dropout=False)['params'])
      of,nf=flatten_dict(old),flatten_dict(new);self.assertEqual(of.keys(),nf.keys())
      roles=set()
      for p,v in nf.items():
        if p[-1] in ('gw_b0','mlp_write_gate_bias','emb_gw_b0'):
          b=bam_gates.bias_for_opening(.1,'softcap_silu');roles.add(p[-1])
          np.testing.assert_allclose(v,b,atol=1e-7)
        elif p[-1].startswith('W_') and p[-1].endswith('_gate_b0'):
          b=bam_gates.bias_for_opening(.05,'softcap_silu');roles.add('dynamic_read')
          np.testing.assert_allclose(v,b,atol=1e-7,err_msg='/'.join(p))
        else:
          # In particular static V/O gates and M-shift are bitwise unchanged.
          np.testing.assert_array_equal(v,of[p],err_msg='/'.join(p))
      self.assertEqual(roles,{'gw_b0','mlp_write_gate_bias','emb_gw_b0','dynamic_read'})
      f=lambda p,t:model.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False)[0]
      loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,tok)**2)))(new)
      gf=flatten_dict(nn.unbox(g));self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in gf.values()))
      for name in ('W_lq_gate','W_lk_gate','W_gw','mlp_write_gate','W_emb_g'):
        self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in gf.items() if name in p),0,name)
      np.testing.assert_allclose(f(new,tok)[:,:3],f(new,tok.at[:,3].set(5))[:,:3],atol=1e-6,rtol=1e-6)
    print('SOFTCAP_ALL_READ_WRITE_SCOPE_GRAD_OK',flush=True)


if __name__=='__main__':
  unittest.main()

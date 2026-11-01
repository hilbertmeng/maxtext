"""Relocated last O: actual budgets, final-M shift, masks and fused gradients."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions
from layers.bam_final_local_o import FinalLocalO
from layers.models import Transformer

PREFIX = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixed'
RUNS = [PREFIX + suffix + 'TruePile' for suffix in (
    'FinalLocalORelocatedLastWrites', 'FinalLocalORelocatedLastWritesShift',
    'FinalLocalORelocatedLastWritesSelfAttention',
    'FinalLocalORelocatedLastWritesShiftSelfAttention', 'SplitLastOutputProjections')]


class RelocatedLocalOTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, index, length=4):
    c = self.config(RUNS[index], base_emb_dim=150, base_num_query_heads=2,
        base_num_kv_heads=2, base_num_decoder_layers=3, base_mlp_dim=32,
        vocab_size=32, dtype='float32', weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False, bam_pallas_write=False,
        bam_splash_attention=False, bam_memory_mlp_dim=16, bam_last_mlp_dim=32,
        bam_layer_modes=['local_qk+local_v+local_o'] * 3,
        bam_attention_write_layers=list(range(3 if index < 4 else 2)),
        bam_mlp_write_layers=None, emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8,
        max_target_length=length, max_prefill_predict_length=length,
        query_chunk_size=length)
    return c

  def test_formal_budget_and_pruned_scope(self):
    for i, (width, count) in enumerate(((2036,432112611), (2025,432111443),
                                      (2036,432112611), (2025,432111443),
                                      (3504,432112210))):
      c = self.config(RUNS[i])
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      p = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in p.values()), count)
      self.assertEqual(c.bam_last_mlp_dim, width)
      self.assertEqual(c.mlp_dim,2510)
      self.assertEqual(c.bam_memory_mlp_dim,1128)
      last = {k:v for k,v in p.items() if 'final_local_layer' in k}
      for name in ('W_R_up', 'W_R_gate', 'W_R_gate_b0', 'static_o_key',
                   'general_vo_pre_bias', 'general_o_static_gate'):
        self.assertEqual(any(name in k for k in last), i == 4, name)
      self.assertTrue(any('W_lv_c8_up' in k for k in last))
      self.assertTrue(any('static_v_key' in k for k in last))
      self.assertEqual(any('memory_mlp' in k for k in last), i < 4)
      self.assertEqual(any('W_gw' in k for k in last), i < 4)
      self.assertEqual(any('final_local_o' in k for k in p), i < 4)
      self.assertEqual(any('local_o_out' in k for k in last), i == 4)
      self.assertTrue(c.scan_layers and c.bam_final_local_layer_count == 1)

  def test_exit_shift_reference_all_gradients_and_boundaries(self):
    c = self.tiny(1)
    module = FinalLocalO(c)
    x = jax.random.normal(jax.random.key(1), (1,4,150))
    matrix = jax.random.normal(jax.random.key(2), (1,4,75,32))
    seg = jnp.array([[1,1,2,2]])
    pos = jnp.array([[0,1,0,1]])
    p = nn.unbox(module.init(jax.random.key(3), x, matrix, seg, pos)['params'])
    p['static_o_key'] = jax.random.normal(jax.random.key(4),(32,2)) * .03
    p['m_shift_gate']['kernel'] = jax.random.normal(jax.random.key(5),(150,32)) * .01
    raw_c = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(), bam_final_local_o_shift=False)))
    raw_module = FinalLocalO(raw_c)
    def reference(p, x, matrix):
      norm = x * jax.lax.rsqrt(jnp.mean(x*x, -1, keepdims=True) + c.normalization_layer_epsilon)
      norm *= 1 + p['input_norm']['scale']
      gate = jax.nn.sigmoid(norm @ p['m_shift_gate']['kernel'] + p['m_shift_bias'])
      previous = jnp.concatenate((matrix[:,:1],matrix[:,:-1]),1)
      mixed = (1-gate[...,None,:])*matrix + gate[...,None,:]*previous
      mixed = jnp.where(jnp.array([[False,True,False,True]])[...,None,None],mixed,matrix)
      params = {k:v for k,v in p.items() if k not in ('m_shift_gate','m_shift_bias')}
      return raw_module.apply({'params':params}, x, mixed, seg, pos)
    actual = lambda p,x,m: module.apply({'params':p},x,m,seg,pos)
    target = jax.random.normal(jax.random.key(6),x.shape)
    def loss(fn,p,x,m):
      y = fn(p,x,m)
      return jnp.sum(y*target),y
    (_,a),ga = jax.value_and_grad(lambda p,x,m:loss(actual,p,x,m),argnums=(0,1,2),has_aux=True)(p,x,matrix)
    (_,b),gb = jax.value_and_grad(lambda p,x,m:loss(reference,p,x,m),argnums=(0,1,2),has_aux=True)(p,x,matrix)
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=3e-6)
    for u,v in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
      np.testing.assert_allclose(u,v,rtol=5e-4,atol=5e-6)
    for name in ('m_shift_gate','m_shift_bias'):
      self.assertGreater(sum(float(jnp.sum(t*t)) for t in jax.tree.leaves(ga[0][name])),0)

  def test_mask_scope_writer_gradients_and_causality(self):
    for i in (0,2,4):
      c = self.tiny(i)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model = Transformer(c,mesh,quant=None)
      tokens = jnp.array([[1,2,3,4]],jnp.int32)
      pos = jnp.arange(4)[None]
      seg = jnp.ones_like(tokens)
      seen = []
      original = attentions.BamAttention._attention_block
      def observe(module,*args,**kwargs):
        seen.append((module.layer_inx,module.config.bam_attention_negative_diagonal))
        return original(module,*args,**kwargs)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(attentions.BamAttention,'_attention_block',observe):
        p = nn.unbox(model.init(jax.random.key(10),tokens,pos,seg,tokens))
        def call(p,tokens):return model.apply(p,tokens,pos,seg,tokens)[0]
        y,g = jax.value_and_grad(lambda p:jnp.mean(call(p,tokens)**2))(p)
        np.testing.assert_allclose(call(p,tokens)[:,:3],call(p,tokens.at[:,3].set(5))[:,:3],rtol=2e-5,atol=2e-6)
      self.assertTrue(np.isfinite(float(y)))
      self.assertTrue(all(np.isfinite(np.asarray(t)).all() for t in jax.tree.leaves(g)))
      for layer,negative in seen:
        self.assertEqual(negative, not(i == 2 and layer == 2))
      last = {k:v for k,v in flatten_dict(g).items() if 'final_local_layer' in k}
      if i < 4:
        for name in ('W_gw','memory_mlp'):
          self.assertGreater(sum(float(jnp.sum(v*v)) for k,v in last.items() if name in k),0,name)
      else:
        attn = p['params']['decoder']['final_local_layer']['block']['self_attention']
        self.assertFalse(np.array_equal(attn['out']['kernel'],attn['local_o_out']['kernel']))

  def fused_compare(self,index):
    c = self.tiny(index,128)
    pc = pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_write=True,bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    native,fused = Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
    tok = jax.random.randint(jax.random.key(20),(1,128),0,32)
    pos = (jnp.arange(128)%64)[None]
    seg = jnp.where(jnp.arange(128)<64,1,2)[None]
    original = splash.make_splash_mha
    def interpret(*args,**kwargs):return original(*args,**dict(kwargs,interpret=True))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
      p,pp = (nn.unbox(m.init(jax.random.key(21),tok,pos,seg,tok)) for m in (native,fused))
      for path,value in flatten_dict(p).items():np.testing.assert_array_equal(value,flatten_dict(pp)[path])
      def loss(model,p):
        y = model.apply(p,tok,pos,seg,tok)[0]
        return jnp.mean(y*y),y
      (_,a),ga = jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
      (_,b),gb = jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(p)
    np.testing.assert_allclose(a,b,rtol=4e-4,atol=3e-6)
    for path,value in flatten_dict(ga).items():
      np.testing.assert_allclose(value,flatten_dict(gb)[path],rtol=3e-3,atol=5e-6,err_msg=str(path))
    print('RELOCATED_O_FULL_PALLAS_GRADIENTS_OK',index,flush=True)

  def test_a_raw_masked_native_pallas(self):self.fused_compare(0)
  def test_b_shift_masked_native_pallas(self):self.fused_compare(1)
  def test_c_raw_self_native_pallas(self):self.fused_compare(2)
  def test_d_shift_self_native_pallas(self):self.fused_compare(3)
  def test_e_split_last_native_pallas(self):self.fused_compare(4)


if __name__ == '__main__':
  unittest.main()

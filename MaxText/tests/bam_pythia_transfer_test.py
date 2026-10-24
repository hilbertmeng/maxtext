"""2.8B VM and independent M-MLP transfers: budgets and live write placement."""
import math
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from bam_prop_test_utils import MLPWriteTest
from layers import bam_mudd
from layers.models import Transformer
import max_utils
import train_compile

PARENT = 'BamPythia2p8BSotaIndependentR768PallasShortTrain'
VM = PARENT + 'MuddL27L31VML2'
RATIO = PARENT + 'MemoryMLPRatio2to1'
RATIO3 = PARENT + 'MemoryMLPRatio3to1'


class PythiaTransferTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budgets_and_live_parameter_placement(self):
    for exp, total in ((VM, 2774921974), (RATIO, 2774920842),
                       (RATIO3, 2774917892)):
      cfg = self.config(exp, target_train_batch_size=None)
      cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                            query_chunk_size=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      shaped, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(shaped[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
      self.assertLess(abs(total - 2774919648), 2561)
      self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_fused_inputs)
      self.assertEqual((cfg.bam_pallas_body, cfg.bam_pallas_vmem_mib), ('v8t', 56))
      self.assertTrue(cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics)
      tail = ('params', 'decoder', 'final_local_layer_1', 'block')
      if exp == VM:
        self.assertEqual([i for i in range(32) if bam_mudd.is_value_layer(cfg, i)], [27, 31])
        self.assertEqual(cfg.bam_mlp_write_layers, list(range(1, 28, 3)) + [29])
        pre = ('params', 'decoder', 'pre_tail_layers')
        self.assertFalse(any(p[:len(pre)] == pre and 'local_1' in p
                             and 'mlp_address_down' in p for p in flat))
        self.assertTrue(any(p[:len(pre)] == pre and 'fetch_2' in p
                            and 'mlp_address_down' in p for p in flat))
        self.assertFalse(any(p[:len(tail)] == tail and 'P_loc_up' in p for p in flat))
        self.assertTrue(any(p[:len(tail)] == tail and 'static_o_key' in p for p in flat))
        self.assertEqual(flat[tail + ('self_attention', 'query', 'kernel')].shape[-1], 20)
        routers = [p for p in flat if 'qkvm_compose' in p and 'dynamic_dense_conn2' in p]
        self.assertEqual(len(routers), 2)
        for p in routers:
          self.assertEqual(flat[p].shape[-2:], (2, 4))
      else:
        self.assertFalse(any(p[:len(tail)] == tail and 'memory_mlp' in p for p in flat))
        ordinary = 5121 * cfg.base_mlp_dim + 2560
        memory = 5120 * cfg.bam_memory_mlp_dim + 3622944
        target, tolerance = (2., .002) if exp == RATIO else (3., .08)
        self.assertAlmostEqual(ordinary / memory, target, delta=tolerance)
        self.assertEqual(cfg.bam_memory_mlp_dim % 128, 0)
        self.assertEqual(cfg.base_mlp_dim % 128, 0)
        if exp == RATIO3:
          self.assertEqual(cfg.bam_last_mlp_dim, 7172)
          # One final GELU channel costs5121 parameters: the chosen width
          # must be the closest match to the original BAM parent's budget.
          error = total - 2774919648
          self.assertLess(abs(error), abs(error - 5121))
          self.assertLess(abs(error), abs(error + 5121))

  def tiny(self, exp):
    cfg = self.config(exp, target_train_batch_size=None, base_emb_dim=64,
        base_num_decoder_layers=11, base_num_query_heads=2, base_num_kv_heads=2,
        base_mlp_dim=32, head_dim=32, vocab_size=32, dtype='float32',
        weight_dtype='float32', attention='dot_product', bam_splash_attention=False)
    cfg.get_keys().update(bam_pallas_core=False, bam_k=32, bam_v=16,
        emb_bam_num_head=2, pythia_rotary_dim=8, bam_standard_qk_dim=8,
        bam_partial_rope_nope_dim=24, bam_local_qk_col_output_dim=24,
        bam_layer_modes=['local_qk+local_v+local_o'] * 11,
        bam_local_full_m_read_bottleneck_dim=16, bam_write_v_bottleneck_dim=16,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16,
        mlp_dim_by_block=[32, 32, 32], bam_final_local_mlp_dim=32,
        bam_last_mlp_dim=40, bam_mudd_tail_mlp_dims=[32, 32, 32],
        bam_mudd_pre_tail_vm_mlp_dim=28)
    if exp == VM:
      cfg.get_keys()['bam_mlp_write_layers'] = [1, 4, 8]  # Move L7 write to L8.
    else:
      cfg.get_keys()['bam_memory_mlp_dim'] = 16
    return cfg

  def check_forward_backward(self, exp):
    cfg = self.tiny(exp)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    positions = jnp.arange(4)[None, :]
    segments = jnp.ones_like(tokens)
    rngs = {'params': jax.random.key(7), 'dropout': jax.random.key(8)}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p, t=tokens):
        return model.apply({'params': p}, t, positions, t, segments,
            decoder_segment_ids=segments, enable_dropout=False, rngs=rngs)[0]
      params = nn.unbox(model.init(rngs, tokens, positions, tokens, segments,
          decoder_segment_ids=segments, enable_dropout=False)['params'])
      flat = flatten_dict(params)
      for p, value in flat.items():
        if 'dynamic_dense_conn2' in p:
          flat[p] = value + .01 * jax.random.normal(jax.random.key(9), value.shape)
      params = unflatten_dict(flat)
      expected = forward(params)
      np.testing.assert_allclose(forward(params, tokens.at[:, 2:].set(9))[:, :2],
                                 expected[:, :2], rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      gf = flatten_dict(grad)
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in gf.values()))
      branches = (['qkvm_compose', 'compose_final_three_source'] if exp == VM
                  else ['memory_mlp'])
      for branch in branches + ['mlp_address_up']:
        leaves = [v for p, v in gf.items() if branch in p]
        self.assertTrue(leaves, branch)
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves), 0., branch)
      if exp == VM:
        pre = ('decoder', 'pre_tail_layers')
        leaves = [v for p, v in gf.items() if p[:2] == pre and 'fetch_2' in p
                  and 'mlp_address_up' in p]
        self.assertTrue(leaves)
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves), 0.)

  def test_vm_explicit_tail_causality_and_moved_write_gradients(self):
    self.check_forward_backward(VM)

  def test_independent_memory_mlp_causality_and_gradients(self):
    self.check_forward_backward(RATIO)


if __name__ == '__main__':
  unittest.main()

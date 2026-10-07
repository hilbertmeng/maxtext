"""Final-block independent C8 reads: budget, schedule, health and gradients."""
import functools
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions, quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest

PARENT = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLastBlockSeparateVOTruePile'


class LastBlockVOTest(unittest.TestCase):
  setUp = MLPWriteTest.setUp
  tearDown = MLPWriteTest.tearDown
  config = MLPWriteTest.config

  def test_full_budget_schedule_and_health(self):
    for exp, count in ((PARENT, 432096128), (EXP, 432092528)):
      c = self.config(exp)
      self.assertEqual(c.DATASET_VARIANT, 'truepile4096')
      self.assertEqual((c.bam_k, c.bam_v, c.bam_local_qk_col_output_dim), (75, 32, 57))
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, kw, sharding, model = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()), count)
      separate = [(p, v.shape) for p, v in flat.items() if 'W_R_v' in p]
      self.assertEqual(len(separate), 3 if exp == EXP else 0)
      for p, shape in separate:
        self.assertIn('final_block', p)
        self.assertEqual(shape, (1200, 16, 1, 8))
      if exp == EXP:
        prefix_keys = [v for p, v in flat.items() if 'layers' in p and 'W_R' in p]
        self.assertEqual(len(prefix_keys), 3)
        self.assertTrue(all(v.shape[1] == 5 for v in prefix_keys))
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
          metrics = jax.eval_shape(functools.partial(train.train_step, model, c, sharding), *args, **kw)[1]['scalar']
        for l in range(18):
          self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in metrics, l in range(1, 18, 3))
          self.assertIn(f'bam/concat/local_v_gate/layer_{l:03d}/mean', metrics)
          self.assertIn(f'bam/concat/local_o_gate/layer_{l:03d}/mean', metrics)
          self.assertEqual(f'bam/concat/vo_read_pair/layer_{l:03d}/cosine' in metrics, l >= 15)
      print('FINAL_BLOCK_VO_FULL_BUDGET_HEALTH_OK', exp, count, flush=True)

  def test_actual_forward_and_consumed_gradients(self):
    c = self.config(EXP, dtype='float32', weight_dtype='float32')
    c.get_keys().update(emb_dim=150, num_query_heads=2, num_kv_heads=2,
        base_num_decoder_layers=6, num_decoder_layers=6, mlp_dim=64,
        mlp_dim_by_block=[64]*3, bam_final_block_mlp_dim_by_block=[62]*3,
        vocab_size=32, bam_layer_modes=['local_qk+local_v+local_o']*6,
        bam_write_v_bottleneck_dim=16, emb_bam_num_head=2,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    model = Transformer(c, mesh, quantizations.configure_quantization(c))
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    args = (tokens, jnp.arange(4)[None], tokens, jnp.ones_like(tokens), jnp.ones_like(tokens))
    seen = []
    original = attentions._attention_op
    def checked(q, k, v, *a, **kw):
      self.assertEqual((q.shape[-1], k.shape[-1], v.shape[-1]), (75, 75, 75))
      seen.append(q.shape)
      return original(q, k, v, *a, **kw)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), mock.patch.object(attentions, '_attention_op', checked):
      params = model.init({'params': jax.random.key(3), 'dropout': jax.random.key(4), 'aqt': jax.random.key(5)}, *args, enable_dropout=False)['params']
      def loss(p):
        return jnp.mean(model.apply({'params': p}, *args, enable_dropout=False, rngs={'aqt': jax.random.key(5)})[0]**2)
      value, grad = jax.jit(jax.value_and_grad(loss))(params)
    self.assertTrue(seen)
    self.assertTrue(np.isfinite(float(value)))
    self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad)))
    flat = flatten_dict(nn.unbox(grad))
    separate = [(p, v) for p, v in flat.items() if 'W_R_v' in p]
    self.assertEqual(len(separate), 3)
    for p, v in separate:
      self.assertGreater(float(jnp.sum(v*v)), 0., str(p))
    for name in ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate'):
      self.assertGreater(sum(float(jnp.sum(v*v)) for p, v in flat.items() if name in p), 0., name)
    print('FINAL_BLOCK_VO_FINITE_CONSUMED_GRADIENTS_OK', float(value), flush=True)


if __name__ == '__main__':
  unittest.main()

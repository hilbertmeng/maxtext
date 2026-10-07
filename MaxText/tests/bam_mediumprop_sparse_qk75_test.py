"""QK93 configuration-only regression: equal budget, full reads, finite training."""
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
EXP = 'BamMediumPropK75EmbedVOnlyQK75AllLocalMLPWriteIndependentEveryThirdTruePile'


class SparseQK75Test(unittest.TestCase):
  setUp = MLPWriteTest.setUp
  tearDown = MLPWriteTest.tearDown
  config = MLPWriteTest.config

  def test_full_budget_and_health(self):
    shapes = []
    for exp in (PARENT, EXP):
      c = self.config(exp)
      self.assertEqual(c.mlp_dim_by_block, [3901, 3774, 3901])
      self.assertEqual((c.head_dim, c.bam_k, c.bam_v, c.bam_standard_qk_dim), (75, 75, 32, 18))
      self.assertEqual(c.DATASET_VARIANT, 'truepile4096')
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, kw, sharding, model = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()), 432096128)
      shapes.append({p: v.shape for p, v in flat.items()})
      if exp == EXP:
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
          metrics = jax.eval_shape(functools.partial(train.train_step, model, c, sharding), *args, **kw)[1]['scalar']
        for l in range(18):
          self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in metrics, l in range(1, 18, 3))
        self.assertTrue(any('extra_scores' in k for k in metrics), sorted(metrics))
    self.assertEqual(shapes[0], shapes[1])
    print('QK93_EQUAL_PARAMETER_TREE_HEALTH_OK 432096128', flush=True)

  def test_actual_qk93_forward_and_gradient(self):
    c = self.config(EXP, dtype='float32', weight_dtype='float32')
    c.get_keys().update(emb_dim=150, num_query_heads=2, num_kv_heads=2,
        base_num_decoder_layers=3, num_decoder_layers=3, mlp_dim=64,
        mlp_dim_by_block=[64]*3, vocab_size=32,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_write_v_bottleneck_dim=16, emb_bam_num_head=2,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    model = Transformer(c, mesh, quantizations.configure_quantization(c))
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    args = (tokens, jnp.arange(4)[None], tokens, jnp.ones_like(tokens), jnp.ones_like(tokens))
    observed = []
    original = attentions._attention_op
    def checked(q, k, v, *a, **kw):
      self.assertEqual((q.shape[-1], k.shape[-1], v.shape[-1]), (93, 93, 75))
      observed.append(q.shape)
      return original(q, k, v, *a, **kw)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules), mock.patch.object(attentions, '_attention_op', checked):
      params = model.init({'params': jax.random.key(3), 'dropout': jax.random.key(4), 'aqt': jax.random.key(5)}, *args, enable_dropout=False)['params']
      def loss(p):
        return jnp.mean(model.apply({'params': p}, *args, enable_dropout=False, rngs={'aqt': jax.random.key(5)})[0]**2)
      value, grad = jax.jit(jax.value_and_grad(loss))(params)
    self.assertTrue(observed)
    self.assertTrue(np.isfinite(float(value)))
    self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad)))
    flat = flatten_dict(nn.unbox(grad))
    for name in ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate', 'static_q_key', 'static_k_key'):
      self.assertGreater(sum(float(jnp.sum(x*x)) for p, x in flat.items() if name in p), 0., name)
    print('ACTUAL_QK93_V75_FINITE_GRADIENT_OK', float(value), flush=True)


if __name__ == '__main__':
  unittest.main()

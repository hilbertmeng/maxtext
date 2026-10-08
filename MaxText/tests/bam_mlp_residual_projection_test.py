"""Focused residual-side projection, exact-budget and scanned-gradient gates."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train
import train_compile
from layers import attentions, fusion, quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest

PARENT = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
SHARED = PARENT.replace('TruePile', 'ResidualWOTruePile')
SEPARATE = PARENT.replace('TruePile', 'ResidualProjectionTruePile')


class ResidualProjectionTest(unittest.TestCase):
  setUp = MLPWriteTest.setUp
  tearDown = MLPWriteTest.tearDown
  config = MLPWriteTest.config

  def small_config(self, name):
    c = self.config(name, dtype='float32', weight_dtype='float32')
    c.get_keys().update(base_emb_dim=150, emb_dim=150,
        base_num_query_heads=2, num_query_heads=2, base_num_kv_heads=2, num_kv_heads=2,
        base_num_decoder_layers=6, num_decoder_layers=6, base_mlp_dim=128, mlp_dim=128,
        mlp_dim_by_block=[128]*3, bam_layer_modes=['local_qk+local_v+local_o']*6,
        vocab_size=128, bam_write_v_bottleneck_dim=32, emb_bam_num_head=2,
        emb_bam_v_bottleneck_dim=32)
    return c

  def test_exact_budget_and_training_graph(self):
    for name in (SHARED, SEPARATE):
      c = self.config(name)
      self.assertEqual(c.bam_mlp_write_content_projection, 'none')
      self.assertEqual(c.mlp_dim_by_block, [3901, 3374 if name == SEPARATE else 3774, 3901])
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, kw, sharding, model = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(args[0].params)
      count = sum(int(np.prod(v.shape)) for v in flat.values())
      self.assertEqual(count, 432096128)
      kernels = [v for p, v in flat.items() if 'mlp_residual_kernel' in p]
      self.assertEqual(bool(kernels), name == SEPARATE)
      if kernels:
        self.assertEqual(len(kernels), 1)
        self.assertEqual(sorted(kernels[0].shape), sorted((6, 16, 75, 1200)))
      self.assertFalse(any('mlp_write_content_kernel' in p for p in flat))
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        scalar = jax.eval_shape(functools.partial(train.train_step, model, c, sharding), *args, **kw)[1]['scalar']
      for i in range(18):
        self.assertEqual(f'bam/concat/mlp_write_gate/layer_{i:03d}/mean' in scalar, i % 3 == 1)
      print('RESIDUAL_FULL_BUDGET_TRAINSTEP_OK', name, count, flush=True)

  def test_forward_wo_not_adjoint_and_live_gradient(self):
    c = self.small_config(SHARED)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    a = attentions.BamAttention(config=c, num_query_heads=2, num_kv_heads=2,
        head_dim=75, bam_k=75, bam_v=32, max_target_length=4,
        max_prefill_predict_length=4, mesh=mesh, attention_kernel='dot_product_chunk',
        dtype=c.dtype, layer_mode='local_qk+local_v+local_o', read_side='col',
        attention_type=c.attention_type)
    x = jax.random.normal(jax.random.key(10), (1, 4, 150))
    w = jax.random.normal(jax.random.key(11), (2, 75, 150))
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      variables = a.init(jax.random.key(12), jnp.zeros((1, 4, 2, 75)), x,
          jnp.zeros((1, 4, 75, 32)), method=a._write)
      params = dict(nn.unbox(variables['params']), out={'kernel': w})
      actual = a.apply({'params': params}, x, method=a.project_mlp_residual)
      reference = lambda k: jnp.einsum('btnk,nkd->btd', x.reshape(1, 4, 2, 75), k)
      np.testing.assert_allclose(actual, reference(w), rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(actual, x @ w.reshape(150, 150), rtol=1e-6, atol=1e-6)
      self.assertGreater(float(jnp.max(jnp.abs(actual - x @ w.reshape(150, 150).T))), 1.)
      grad = jax.grad(lambda k: jnp.sum(a.apply({'params': dict(params, out={'kernel': k})}, x,
          method=a.project_mlp_residual)**2))(w)
      np.testing.assert_allclose(grad, jax.grad(lambda k: jnp.sum(reference(k)**2))(w), rtol=1e-5, atol=1e-5)
      self.assertGreater(float(jnp.sum(grad**2)), 0.)
      c.get_keys()['bam_mlp_residual_projection'] = 'independent'
      v = nn.unbox(a.init(jax.random.key(12), x, method=a.project_mlp_residual))
      v['params']['mlp_residual_kernel'] = w
      np.testing.assert_allclose(a.apply(v, x, method=a.project_mlp_residual), actual, rtol=0, atol=0)
    print('RESIDUAL_WO_FORWARD_DIRECTION_AND_GRADIENT_OK', flush=True)

  def test_single_layer_projection_does_not_change_matrix_write(self):
    c = self.small_config(SEPARATE)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    layer = fusion.SubDecoderLayer(config=c, mesh=mesh, layer_inx=1)
    x = jax.random.normal(jax.random.key(1), (1, 4, 150))
    tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
    matrix = jax.random.normal(jax.random.key(2), (1, 4, 75, 32))
    kw = dict(inputs=x, decoder_segment_ids=jnp.ones_like(tok),
        decoder_positions=jnp.arange(4)[None], decoder_input_tokens=tok,
        deep_embedding=None, deterministic=True, model_mode='train', eos_sum=None,
        M_in=matrix, layer_index=1)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      params = nn.unbox(layer.init({'params': jax.random.key(3), 'aqt': jax.random.key(4)}, **kw)['params'])
      kernel = params['self_attention']['mlp_residual_kernel']
      outputs = []
      for scale in (0., 1., 2.):
        p = dict(params, self_attention=dict(params['self_attention'], mlp_residual_kernel=scale*kernel))
        outputs.append(layer.apply({'params': p}, **kw))
    for _, updated_matrix in outputs[1:]:
      np.testing.assert_allclose(updated_matrix, outputs[0][1], rtol=0, atol=0)
    np.testing.assert_allclose(outputs[2][0] - outputs[0][0],
        2*(outputs[1][0] - outputs[0][0]), rtol=2e-5, atol=2e-5)
    self.assertGreater(float(jnp.max(jnp.abs(outputs[1][0] - outputs[0][0]))), 1e-4)
    print('RESIDUAL_ONLY_CHANGES_VECTOR_BRANCH_MATRIX_UNCHANGED_OK', flush=True)

  def test_scanned_consumed_gradients(self):
    for name in (SHARED, SEPARATE):
      c = self.small_config(name)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      model = Transformer(c, mesh, quantizations.configure_quantization(c))
      tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
      call = (tok, jnp.arange(4)[None], tok, jnp.ones_like(tok), jnp.ones_like(tok))
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        params = model.init({'params': jax.random.key(3), 'dropout': jax.random.key(4), 'aqt': jax.random.key(5)},
            *call, enable_dropout=False)['params']
        loss = lambda p: jnp.mean(model.apply({'params': p}, *call, enable_dropout=False,
            rngs={'aqt': jax.random.key(5)})[0])
        value, grad = jax.jit(jax.value_and_grad(loss))(params)
      self.assertTrue(np.isfinite(float(value)))
      self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
      flat = flatten_dict(nn.unbox(grad))
      for needle in ['out', 'mlp_address_down', 'mlp_write_gate'] + (['mlp_residual_kernel'] if name == SEPARATE else []):
        gs = [g for p, g in flat.items() if needle in p]
        self.assertTrue(gs)
        self.assertGreater(sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in gs), 0.)
      print('RESIDUAL_SCANNED_CONSUMED_GRADIENT_OK', name, float(value), flush=True)

  def test_invalid_configuration(self):
    from bam_config import validate_bam_config
    for update in ({'bam_mlp_residual_projection': 'wrong'}, {'bam_mlp_write_every': 0}, {'bam_k': 64}):
      c = self.config(SHARED)
      c.get_keys().update(update)
      with self.assertRaises(ValueError):
        validate_bam_config(c)


if __name__ == '__main__':
  unittest.main()

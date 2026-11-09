"""Matched TruePile Llama Full AttnRes: budgets, causality and live gradients."""
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import residual_connections_test as ref
import train_compile
from layers.residual_connections import FullAttentionResidual

BASE = 'BamMHAMediumPropC256TruePile'
RUN = BASE + 'AttnRes'


class LlamaAttnResTest(unittest.TestCase):
  setUp = ref.ResidualModelTest.setUp
  tearDown = ref.ResidualModelTest.tearDown
  config = ref.MLPWriteTest.config
  tiny = ref.ResidualModelTest.tiny

  def test_separate_history_matches_stacked_forward_and_gradients(self):
    cfg = ref.connection_config()
    module = FullAttentionResidual(cfg)
    rng = np.random.default_rng(91)
    sources = tuple(jnp.asarray(rng.normal(size=(2, 3, 16)), jnp.float32) for _ in range(7))
    params = {'query': jnp.asarray(rng.normal(size=16), jnp.float32),
              'key_scale': jnp.asarray(rng.uniform(.5, 1.5, size=16), jnp.float32)}
    def stacked(p, s):
      values = jnp.stack(s, axis=-2).astype(jnp.float32)
      keys = values * jax.lax.rsqrt(jnp.mean(values**2, axis=-1, keepdims=True) + cfg.attnres_norm_epsilon)
      scores = jnp.einsum('...sd,d->...s', keys * p['key_scale'], p['query'])
      return jnp.sum(jax.nn.softmax(scores, axis=-1)[..., None] * values, axis=-2)
    def actual(p, s):
      return module.apply({'params': p}, s)
    np.testing.assert_allclose(actual(params, sources), stacked(params, sources), rtol=2e-5, atol=2e-6)
    a = jax.grad(lambda p, s: jnp.square(actual(p, s)).sum(), argnums=(0, 1))(params, sources)
    b = jax.grad(lambda p, s: jnp.square(stacked(p, s)).sum(), argnums=(0, 1))(params, sources)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
      np.testing.assert_allclose(x, y, rtol=3e-5, atol=2e-5)

  def test_formal_recipe_and_full_parameter_tree(self):
    base, cfg = self.config(BASE), self.config(RUN)
    for key in ('init_method', 'learning_rate', 'steps', 'init_weights_seed',
                'attention', 'query_chunk_size', 'max_target_length', 'emb_dim',
                'num_decoder_layers', 'head_dim', 'adam_weight_decay'):
      self.assertEqual(getattr(base, key), getattr(cfg, key), key)
    self.assertEqual((cfg.mlp_dim, cfg.emb_dim, cfg.num_decoder_layers), (3199, 1200, 18))
    self.assertFalse(cfg.scan_layers)
    self.assertTrue(cfg.record_training_health_metrics)
    self.assertFalse(cfg.record_internal_nn_metrics)
    self.assertGreater(cfg.tensorboard_gcs_rotate_period, 0)
    cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
    mesh = jax.sharding.Mesh(ref.max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, *_ = train_compile.get_shaped_inputs(mesh, cfg)
    flat = flatten_dict(nn.unbox(shaped[0].params))
    total = sum(math.prod(v.shape) for v in flat.values())
    self.assertEqual(total, 432142800)
    queries = [v for p, v in flat.items() if p[-1] == 'query']
    self.assertEqual(len(queries), 36)
    self.assertTrue(all(v.shape == (1200,) for v in queries))

  def test_causal_bf16_forward_and_router_gradients(self):
    cfg = self.tiny(RUN)
    cfg.get_keys().update(dtype=jnp.bfloat16, remat_policy='full')
    mesh = jax.sharding.Mesh(ref.max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = ref.Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    positions, segments = jnp.arange(4)[None], jnp.ones_like(tokens)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = model.init(jax.random.key(29), tokens, positions, tokens,
                          segments, segments, enable_dropout=False)['params']
      def forward(p, t):
        return model.apply({'params': p}, t, positions, t, segments, segments,
                           enable_dropout=False)[0]
      loss, grad = jax.value_and_grad(lambda p: forward(p, tokens).mean())(params)
      np.testing.assert_allclose(forward(params, tokens)[:, :2],
                                 forward(params, tokens.at[:, 2:].set(7))[:, :2],
                                 rtol=2e-3, atol=2e-3)
    self.assertTrue(np.isfinite(loss))
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(nn.unbox(grad))))
    queries = [x for p, x in flatten_dict(nn.unbox(grad)).items() if p[-1] == 'query']
    self.assertGreater(sum(float(jnp.sum(x*x)) for x in queries), 0)


if __name__ == '__main__':
  unittest.main()

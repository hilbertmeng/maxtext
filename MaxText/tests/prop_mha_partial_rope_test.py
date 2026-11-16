"""Standard odd-head Prop MHA against an independent tail-RoPE reference."""
import unittest

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from bam_prop_test_utils import MLPWriteTest
import max_utils
from layers.attentions import Attention


class ReferenceAttention(Attention):
  def apply_rotary_embedding(self, inputs, inputs_positions, name, embedding_dims=None):
    # Explicit RoPE74, retaining the first NoPE coordinate of each 75-wide head.
    width = inputs.shape[-1] - 1
    timescale = self.config.rope_min_timescale * (
        self.config.rope_max_timescale / self.config.rope_min_timescale
    ) ** (2 * jnp.arange(width // 2, dtype=jnp.float32) / width)
    angle = inputs_positions[:, :, None, None] / timescale
    a, b = jnp.split(inputs[..., 1:], 2, axis=-1)
    return jnp.concatenate((inputs[..., :1], a * jnp.cos(angle) - b * jnp.sin(angle),
                            b * jnp.cos(angle) + a * jnp.sin(angle)), axis=-1)


class PropMHAPartialRoPETest(MLPWriteTest, unittest.TestCase):
  def test_odd_head_forward_and_all_gradients(self):
    cfg = self.config('BamMHAMediumPropL12TruePile', dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(emb_dim=150, num_query_heads=2, num_kv_heads=2, head_dim=75,
                          num_decoder_layers=1, mlp_dim=24, vocab_size=32, attention='dot_product')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x = jax.random.normal(jax.random.key(3), (2, 4, 150))
    positions = jnp.array([[0, 1, 2, 3], [3, 4, 5, 6]], dtype=jnp.int32)
    segments = jnp.ones((2, 4), dtype=jnp.int32)
    kwargs = dict(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=75,
                  max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
                  attention_kernel='dot_product', dtype=jnp.float32, weight_dtype=jnp.float32)
    actual = Attention(**kwargs)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(actual.init(jax.random.key(4), x, x, positions, segments))
      y = actual.apply(params, x, x, positions, segments)
      grad = jax.grad(lambda p, z: jnp.sum(actual.apply(p, z, z, positions, segments)**2),
                      argnums=(0, 1))(params, x)
      cfg.get_keys().update(bam_partial_rope=False)
      reference = ReferenceAttention(**kwargs)
      expected = reference.apply(params, x, x, positions, segments)
      expected_grad = jax.grad(lambda p, z: jnp.sum(reference.apply(p, z, z, positions, segments)**2),
                               argnums=(0, 1))(params, x)
    np.testing.assert_allclose(y, expected, rtol=2e-5, atol=2e-6)
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(expected_grad)):
      np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)


if __name__ == '__main__':
  unittest.main()

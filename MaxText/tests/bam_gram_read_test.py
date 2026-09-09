"""Scheme C vs explicit normalized effective keys, including nonzero VJPs."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from layers.attentions import _effective_key_bam_read


class GramReadTest(absltest.TestCase):
  def test_packed_qkv_module(self):
    from bam_local_fetch_test import LocalFetchTest
    import max_utils
    from layers.attentions import BamAttention
    helper = LocalFetchTest()
    self.addCleanup(helper.doCleanups)
    cfg = helper.config('BamMediumIndependentLLFGramMulMix')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    module = BamAttention(
        config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
        max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
        attention_kernel='dot_product_chunk', dtype=cfg.dtype,
        layer_mode='local_qk+local_o', attention_type=cfg.attention_type)
    x = jnp.ones((1, 8, 128), cfg.dtype)
    M = jnp.ones((1, 8, 32, 32), cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    params = module.init({'params': jax.random.key(5), 'aqt': jax.random.key(6)},
                        *args, M_in=M, deterministic=True, layer_index=2)
    for arm in ('q', 'k', 'v'):
      self.assertEqual(params['params'][f'W_l{arm}_gate_b0'].value.shape, (2, 2))
    def loss(p):
      y, m = module.apply({'params': p}, *args, M_in=M, deterministic=True, layer_index=2)
      return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(m.astype(jnp.float32)**2)
    grads = jax.grad(loss)(params['params'])
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads)))

  def test_forward_and_gradients(self):
    for rank in (1, 2, 4):
      rng = jax.random.split(jax.random.key(rank), 5)
      M = jax.random.normal(rng[0], (2, 3, 7, 5))
      key = jax.random.normal(rng[1], (2, 3, rank, 12))
      mix = jax.random.normal(rng[2], (2, 3, 4, 2, rank))
      gate = jax.random.normal(rng[3], (2, 3, 4, 2))
      cotangent = jax.random.normal(rng[4], (2, 3, 4, 12))

      def reference(M, key, mix, gate):
        row, col = jnp.split(key, [7], -1)
        def normalized(A, H, g):
          p = jnp.einsum('btnr,btrk->btnk', H, A)
          return 2 * jax.nn.sigmoid(g)[..., None] * p * jax.lax.rsqrt(
              jnp.mean(p * p, -1, keepdims=True) + 1e-4)
        row = normalized(row, mix[..., 0, :], gate[..., 0])
        col = normalized(col, mix[..., 1, :], gate[..., 1])
        u = jnp.einsum('btkv,btnv->btnk', M, col)
        v = jnp.einsum('btkv,btnk->btnv', M, row)
        return jnp.concatenate((u, v), -1)

      def objective(fn, *args):
        return jnp.sum(fn(*args) * cotangent)
      args = (M, key, mix, gate)
      expected = reference(*args)
      expected_grad = jax.grad(lambda *a: objective(reference, *a), argnums=(0, 1, 2, 3))(*args)
      for gram in ('dot', 'mul_reduce'):
        for placement in ('mix', 'output'):
          for second in ('dot', 'mul_reduce'):
            def actual(M, key, mix, gate):
              return _effective_key_bam_read(
                  M, key, mix, gate, rms_epsilon=1e-4, key_scale=2.0,
                  implementation='mul_reduce_btn', second_implementation=second,
                  gram_implementation=gram, scale_placement=placement,
                  read_side='both', v_projection=None, return_sides=False)
            np.testing.assert_allclose(actual(*args), expected, atol=2e-5, rtol=2e-5)
            grad = jax.grad(lambda *a: objective(actual, *a), argnums=(0, 1, 2, 3))(*args)
            for g, ref in zip(grad, expected_grad):
              np.testing.assert_allclose(g, ref, atol=2e-4, rtol=2e-4)
            zero_args = (M, jnp.zeros_like(key), mix, gate)
            zero_grad = jax.grad(lambda *a: objective(actual, *a), argnums=(0, 1, 2, 3))(*zero_args)
            self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in zero_grad))


if __name__ == '__main__':
  absltest.main()

"""Compact column-only parameters match bilateral parameters with row reads disabled."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from layers.attentions import BamAttention, _packed_local_layout
import bam_local_fetch_test


class ColOnlyBudgetTest(absltest.TestCase):
  config = bam_local_fetch_test.LocalFetchTest.config
  def test_compact_modules_match_masked_bilateral(self):
    for mode in ('local_qk+local_o', 'local_qk+full'):
      cfg = self.config('BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow')
      cfg.get_keys()['bam_fetched_read_side'] = 'col'
      cfg.get_keys()['bam_prune_all_row_reads'] = False
      cfg.get_keys()['bam_local_v_share_output_coordinates'] = False
      import max_utils
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      kwargs = dict(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
                    max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
                    attention_kernel='dot_product_chunk', dtype=cfg.dtype,
                    layer_mode=mode, read_side='col', attention_type=cfg.attention_type)
      full = BamAttention(**kwargs)
      x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
      m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), dtype=cfg.dtype)
      args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
      call = dict(M_in=m, deterministic=True, layer_index=3)
      params = full.init({'params': jax.random.key(3)}, *args, **call)['params']
      arms = full.apply({'params': params}, method=lambda mod: tuple(mod._local_arms.values()))
      # Exercise nonzero reads, not only their zero-initialized forward.
      flat = flatten_dict(params)
      for index, (path, leaf) in enumerate(flat.items()):
        if path[0] in ('W_R', 'W_lq_bias', 'W_lk_bias', 'W_lv_bias'):
          value = leaf.value
          flat[path] = leaf.replace(value=.1 * jax.random.normal(
              jax.random.key(index+20), value.shape, value.dtype))
      params = unflatten_dict(flat)
      expected = full.apply({'params': params}, *args, **call)
      old_layout, _ = _packed_local_layout(arms)

      def compact(p):
        new = {}
        for path, leaf in flatten_dict(p).items():
          name = path[0]
          value = leaf.value
          if name == 'abs_v_row_decoder':
            continue
          if name == 'W_R': value = value[..., 32:]
          elif name in ('W_R_gate', 'W_R_gate_b0'): value = value[..., 1:2]
          elif name in ('W_lq_bias','W_lk_bias','W_lv_bias'): value = value[..., 32:]
          elif name in ('W_lq_gate_b0','W_lk_gate_b0','W_lv_gate_b0'): value = value[..., 1:2]
          elif name == 'W_local_packed':
            pieces = []
            for arm, (bs, gs, hs) in zip(arms, old_layout):
              lead = value.shape[:-1]
              pieces.extend((value[..., bs].reshape(lead+arm.key_shape)[...,32:].reshape(lead+(-1,)),
                             value[..., gs].reshape(lead+arm.gate_shape)[...,1:2].reshape(lead+(-1,)),
                             value[..., hs].reshape(lead+arm.mix_shape)[...,1:2,:].reshape(lead+(-1,))))
            value = jnp.concatenate(pieces, -1)
          new[path] = leaf.replace(value=value)
        return unflatten_dict(new)

      cfg.get_keys()['bam_prune_all_row_reads'] = True
      small = BamAttention(**kwargs)
      small_params = compact(params)
      init = small.init({'params': jax.random.key(3)}, *args, **call)['params']
      self.assertEqual(jax.tree.structure(init), jax.tree.structure(small_params))
      self.assertEqual([a.shape for a in jax.tree.leaves(init)],
                       [a.shape for a in jax.tree.leaves(small_params)])
      actual = small.apply({'params': small_params}, *args, **call)
      for a, b in zip(expected, actual):
        np.testing.assert_allclose(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), atol=.002, rtol=.002)
      grad = jax.grad(lambda p: sum(jnp.mean(z.astype(jnp.float32)**2)
                       for z in small.apply({'params': p}, *args, **call)))(small_params)
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(grad)))
      self.assertGreater(float(jnp.linalg.norm(grad['W_R']['kernel'].value.astype(jnp.float32))), 0)


if __name__ == '__main__':
  absltest.main()

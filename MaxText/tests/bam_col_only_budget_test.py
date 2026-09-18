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

  def test_no_local_qk_modes_keep_output_reads_and_write(self):
    import max_utils
    cfg = self.config('BamMediumIndependentLLFMLPPerLayerColOnlyK48PartialRoPE')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
    m = jax.random.normal(jax.random.key(2), (1, 8, 48, 32), dtype=cfg.dtype)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    for mode in ('local_o', 'full'):
      module = BamAttention(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
          bam_k=48, bam_v=32, max_target_length=8,
          max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode=mode, read_side='col', attention_type=cfg.attention_type)
      variables = module.init(
          {'params': jax.random.key(3)}, *args, M_in=m,
          deterministic=True, layer_index=2)
      arm_names = module.apply(
          variables, method=lambda mod: tuple(mod._local_arms))
      self.assertEqual(arm_names, ('v',) if mode == 'local_o' else ())
      out, m_out = module.apply(
          variables, *args, M_in=m, deterministic=True, layer_index=2)
      self.assertEqual(out.shape, x.shape)
      self.assertEqual(m_out.shape, m.shape)
      self.assertFalse(bool(jnp.array_equal(m_out, m)))

  def test_k48_column_only_uses_no_extra_parameters(self):
    import max_utils
    trees = []
    for exp_name, k_dim in (
        ('BamMediumIndependentLLFMLPPerLayerColOnly', 32),
        ('BamMediumIndependentLLFMLPPerLayerColOnlyK48', 48)):
      cfg = self.config(exp_name)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      module = BamAttention(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
          bam_k=cfg.bam_k, bam_v=cfg.bam_v,
          max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode='local_qk+full', read_side='col',
          attention_type=cfg.attention_type)
      x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
      m = jax.random.normal(
          jax.random.key(2), (1, 8, k_dim, 32), dtype=cfg.dtype)
      args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
      variables = module.init(
          {'params': jax.random.key(3)}, *args, M_in=m,
          deterministic=True, layer_index=2)
      out, m_out = module.apply(
          variables, *args, M_in=m, deterministic=True, layer_index=2)
      self.assertEqual(out.shape, x.shape)
      self.assertEqual(m_out.shape, m.shape)
      trees.append(variables['params'])
    self.assertEqual(
        sum(x.size for x in jax.tree.leaves(trees[0])),
        sum(x.size for x in jax.tree.leaves(trees[1])))

  def test_k64_qk48_project_starts_as_truncate(self):
    import max_utils
    names = (
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE',
        'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48ProjectPartialRoPE')
    modules, variables, outputs = [], [], []
    x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=jnp.bfloat16)
    m = jax.random.normal(jax.random.key(2), (1, 8, 64, 32), dtype=jnp.bfloat16)
    args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
    for name in names:
      cfg = self.config(name)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      module = BamAttention(
          config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
          bam_k=cfg.bam_k, bam_v=cfg.bam_v,
          max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
          attention_kernel='dot_product_chunk', dtype=cfg.dtype,
          layer_mode='local_qk+full', read_side='col',
          attention_type=cfg.attention_type)
      value = module.init(
          {'params': jax.random.key(3)}, *args, M_in=m,
          deterministic=True, layer_index=2)
      modules.append(module)
      variables.append(value)
      outputs.append(module.apply(
          value, *args, M_in=m, deterministic=True, layer_index=2))

    truncate = flatten_dict(variables[0]['params'])
    project = flatten_dict(variables[1]['params'])
    projection_names = ('local_q_col_projection', 'local_k_col_projection')
    project_control = {p: v for p, v in project.items()
                       if p[0] not in projection_names}
    self.assertEqual(truncate.keys(), project_control.keys())
    for path in truncate:
      np.testing.assert_array_equal(
          np.asarray(truncate[path].value), np.asarray(project_control[path].value))
    selector = np.eye(64, 48, dtype=np.float32)
    for name in projection_names:
      np.testing.assert_array_equal(
          np.asarray(project[(name,)].value, dtype=np.float32), selector)
    self.assertEqual(
        sum(x.size for x in jax.tree.leaves(variables[1]['params']))
        - sum(x.size for x in jax.tree.leaves(variables[0]['params'])),
        2 * 64 * 48)
    for truncate_value, project_value in zip(outputs[0], outputs[1]):
      np.testing.assert_array_equal(
          np.asarray(truncate_value), np.asarray(project_value))

  def test_o_only_preserves_qkv(self):
    import max_utils
    for mode in ('local_qk+local_o', 'local_qk+full'):
      cfg = self.config('BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow')
      cfg.get_keys()['bam_fetched_read_side'] = 'col'
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      kwargs = dict(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=64,
                    max_target_length=8, max_prefill_predict_length=8, mesh=mesh,
                    attention_kernel='dot_product_chunk', dtype=cfg.dtype,
                    layer_mode=mode, read_side='both', attention_type=cfg.attention_type)
      full = BamAttention(**kwargs)
      x = jax.random.normal(jax.random.key(1), (1, 8, 128), dtype=cfg.dtype)
      m = jax.random.normal(jax.random.key(2), (1, 8, 32, 32), dtype=cfg.dtype)
      args = (x, x, jnp.arange(8)[None], jnp.ones((1, 8), jnp.int32))
      call = dict(M_in=m, deterministic=True, layer_index=3)
      params = full.init({'params': jax.random.key(3)}, *args, **call)['params']
      flat = flatten_dict(params)
      for i, (path, leaf) in enumerate(flat.items()):
        if path[0] in ('W_R', 'W_lq_bias', 'W_lk_bias', 'W_lv_bias'):
          flat[path] = leaf.replace(value=.1 * jax.random.normal(
              jax.random.key(i+20), leaf.value.shape, leaf.value.dtype))
      params = unflatten_dict(flat)
      expected = full.apply({'params': params}, *args, **call)
      cfg.get_keys()['bam_fetched_read_side'] = 'both'
      cfg.get_keys()['bam_prune_o_row_reads'] = True
      small = BamAttention(**kwargs)
      compact = {}
      for path, leaf in flat.items():
        if path[0] == 'abs_v_row_decoder': continue
        if path[0] == 'W_R': leaf = leaf.replace(value=leaf.value[..., 32:])
        elif path[0] in ('W_R_gate', 'W_R_gate_b0'):
          leaf = leaf.replace(value=leaf.value[..., 1:2])
        compact[path] = leaf
      compact = unflatten_dict(compact)
      init = small.init({'params': jax.random.key(3)}, *args, **call)['params']
      self.assertEqual(jax.tree.structure(init), jax.tree.structure(compact))
      self.assertEqual([a.shape for a in jax.tree.leaves(init)],
                       [a.shape for a in jax.tree.leaves(compact)])
      actual = small.apply({'params': compact}, *args, **call)
      for a, b in zip(expected, actual):
        np.testing.assert_allclose(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32), atol=.002, rtol=.002)
      grad = jax.grad(lambda p: sum(jnp.mean(z.astype(jnp.float32)**2)
                       for z in small.apply({'params': p}, *args, **call)))(compact)
      self.assertTrue(all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(grad)))

  def test_compact_modules_match_masked_bilateral(self):
    self._check_compact_local(keep_o=False)

  def test_qkv_only_preserves_o(self):
    self._check_compact_local(keep_o=True)

  def _check_compact_local(self, keep_o):
    for mode in ('local_qk+local_o', 'local_qk+full'):
      cfg = self.config('BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow')
      cfg.get_keys()['bam_fetched_read_side'] = 'both' if keep_o else 'col'
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
          if name == 'abs_v_row_decoder' and not keep_o:
            continue
          if name == 'W_R' and not keep_o: value = value[..., 32:]
          elif name in ('W_R_gate', 'W_R_gate_b0') and not keep_o: value = value[..., 1:2]
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

      cfg.get_keys()['bam_prune_local_row_reads' if keep_o else 'bam_prune_all_row_reads'] = True
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

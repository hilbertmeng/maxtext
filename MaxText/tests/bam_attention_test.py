"""Focused tests for BAM runtime read-key transforms."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from layers import initializers
from layers import normalizations

from layers.attentions import (
    GroupedRMSNorm,
    _attention_op,
    _bam_fetch_op,
    _dynamic_bam_fetch_mix_weights,
    _fit_bam_read_to_head,
    _packed_factorized_local_qk_init,
    _dynamic_mixed_bam_fetch_alpha,
    _mix_bam_write_v,
    _select_bam_write_source,
    _shared_bam_fetch_alpha,
    _sliding_window_bam_fetch_alpha,
    _temporal_block_bam_fetch,
    _transform_bam_read_key,
    _update_bam_matrix,
    bam_read,
    codebook_read,
    factorized_head_bam_read,
)

_RMS_EPSILON = normalizations.DEFAULT_RMS_EPSILON


class BamReadKeyTransformTest(absltest.TestCase):

  def test_attention_op_matches_dense_and_chunk_values_and_gradients(self):
    b, t, n, d, chunk_size = 2, 6, 3, 4, 2
    keys = jax.random.split(jax.random.PRNGKey(79), 6)
    args = (
        jax.random.normal(keys[0], (b, t, n, d)),
        jax.random.normal(keys[1], (b, t, n, d)),
        jax.random.normal(keys[2], (b, t, n, d)),
    )
    segment_ids = jnp.asarray([[1, 1, 1, 2, 2, 2], [3, 3, 4, 4, 4, 4]])
    causal = jnp.arange(t)[None, :] <= jnp.arange(t)[:, None]
    valid = causal[None] & (
        segment_ids[:, :, None] == segment_ids[:, None, :])
    output_weight = jax.random.normal(keys[3], (b, t, n, d))
    alpha_weight = jax.random.normal(keys[4], (b, n, t, t))

    def dense(values):
      return _attention_op(
          *values, valid, attn_logits_soft_cap=3.0, float32_logits=True,
      )

    def reference(values):
      query, key, value = values
      logits = jnp.einsum('btnd,bsnd->bnts', query, key)
      logits = jnp.tanh(logits / 3.0) * 3.0
      logits = jnp.where(valid[:, None], logits, -1e30).astype(jnp.float32)
      alpha = jax.nn.softmax(logits, axis=-1)
      return jnp.einsum('bnts,bsnd->btnd', alpha, value), alpha

    expected_y, expected_alpha = reference(args)
    actual_y, actual_alpha = dense(args)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_alpha, expected_alpha, rtol=1e-6, atol=1e-6)
    chunk_outputs = []
    chunk_alphas = []
    for q0 in range(0, t, chunk_size):
      q1 = q0 + chunk_size
      chunk_y, chunk_alpha = _attention_op(
          args[0][:, q0:q1], args[1][:, :q1], args[2][:, :q1],
          valid[:, q0:q1, :q1], attn_logits_soft_cap=3.0,
          float32_logits=True)
      chunk_outputs.append(chunk_y)
      chunk_alphas.append(jnp.pad(
          chunk_alpha, ((0, 0), (0, 0), (0, 0), (0, t - q1))))
    np.testing.assert_allclose(
        jnp.concatenate(chunk_outputs, axis=1), expected_y,
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.concatenate(chunk_alphas, axis=2), expected_alpha,
        rtol=1e-6, atol=1e-6)

    def objective(function, values):
      y, alpha = function(values)
      return jnp.sum(y * output_weight) + jnp.sum(alpha * alpha_weight)

    expected_value, expected_grad = jax.value_and_grad(
        lambda values: objective(reference, values))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda values: objective(dense, values))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
    for got, expected in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  def test_bam_fetch_op_matches_v2_fetch_values_and_gradients(self):
    b, n, t, k, v = 2, 3, 5, 4, 6
    keys = jax.random.split(jax.random.PRNGKey(83), 4)
    args = (
        jax.nn.softmax(jax.random.normal(keys[0], (b, n, t, t)), axis=-1),
        jax.random.normal(keys[1], (b, t, n)),
        jax.random.normal(keys[2], (b, t, k, v)),
    )
    upstream = jax.random.normal(keys[3], (b, t, k, v))
    diagonal = jnp.arange(t)

    def reference(values):
      alpha, mix_weights, fetch_state = values
      fetch_alpha = jnp.einsum('bnts,btn->bts', alpha, mix_weights)
      pre_diagonal = fetch_alpha
      fetch_alpha = fetch_alpha.at[:, diagonal, diagonal].set(1)
      Mbar = jnp.einsum('bts,bskv->btkv', fetch_alpha, fetch_state)
      return Mbar, fetch_alpha, pre_diagonal

    def actual(values):
      alpha, mix_weights, fetch_state = values
      return _bam_fetch_op(
          alpha, fetch_state, mix_weights=mix_weights,
          diagonal_indices=(diagonal, diagonal))

    expected = reference(args)
    got = actual(args)
    for got_item, expected_item in zip(got, expected):
      np.testing.assert_allclose(got_item, expected_item, rtol=1e-6, atol=1e-6)
    masked = _bam_fetch_op(
        args[0], args[2], mix_weights=args[1],
        diagonal_mask=jnp.eye(t, dtype=bool))[0]
    np.testing.assert_allclose(
        masked, expected[0], rtol=1e-6, atol=1e-6)
    dense = _bam_fetch_op(expected[1][:, None], args[2])[0]
    np.testing.assert_allclose(
        dense[:, 0], expected[0], rtol=1e-6, atol=1e-6)

    expected_value, expected_grad = jax.value_and_grad(
        lambda values: jnp.sum(reference(values)[0] * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda values: jnp.sum(actual(values)[0] * upstream))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
    for got_item, expected_item in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got_item, expected_item, rtol=1e-6, atol=1e-6)

  def test_bam_read_head_mapping_pads_or_adapts_only_v_side(self):
    direct = jnp.arange(96, dtype=jnp.float32).reshape(1, 1, 1, 96)
    padded = _fit_bam_read_to_head(direct, bam_k=64, head_dim=128)
    np.testing.assert_array_equal(padded[..., :96], direct)
    np.testing.assert_array_equal(padded[..., 96:], 0)

    adapter = jnp.zeros((1, 64, 32), dtype=jnp.float32)
    adapter = adapter.at[0, :32].set(jnp.eye(32))
    wide = jnp.arange(96, dtype=jnp.float32).reshape(1, 1, 1, 96)
    adapted = _fit_bam_read_to_head(
        wide, bam_k=32, head_dim=64, v_adapter=adapter)
    np.testing.assert_array_equal(adapted[..., :32], wide[..., :32])
    np.testing.assert_array_equal(adapted[..., 32:], wide[..., 32:64])

  def test_bam_read_head_mapping_rejects_wide_v_without_adapter(self):
    with self.assertRaisesRegex(ValueError, 'without an adapter'):
      _fit_bam_read_to_head(
          jnp.zeros((1, 1, 1, 96)), bam_k=32, head_dim=64)

  def test_dynamic_bam_fetch_rms_mix_weights(self):
    logits = jax.random.normal(jax.random.PRNGKey(13), (2, 4, 3))
    _, weights = _dynamic_bam_fetch_mix_weights(
        logits, jnp.bfloat16, 'rms', rms_epsilon=_RMS_EPSILON)
    self.assertEqual(weights.shape, logits.shape)
    self.assertEqual(weights.dtype, jnp.bfloat16)

  def test_packed_local_qk_preserves_segment_initializers(self):
    embed, heads, key_width = 64, 4, 12
    regular_init = initializers.nd_dense_init(
        1.0, 'fan_in', 'truncated_normal')
    init = _packed_factorized_local_qk_init(
        regular_init, heads, key_width)
    kernel = init(
        jax.random.PRNGKey(0),
        (embed, 2 * (key_width + 2 + 2 * heads)), jnp.float32)
    q_key, q_gate, q_mix, k_key, k_gate, k_mix = jnp.split(
        kernel,
        (key_width, key_width + 2, key_width + 2 + 2 * heads,
         2 * key_width + 2 + 2 * heads,
         2 * key_width + 4 + 2 * heads),
        axis=-1)
    for zero_segment in (q_key, q_gate, k_key, k_gate):
      np.testing.assert_array_equal(zero_segment, 0)
    self.assertGreater(float(jnp.linalg.norm(q_mix)), 0.0)
    self.assertGreater(float(jnp.linalg.norm(k_mix)), 0.0)
    self.assertFalse(np.array_equal(q_mix, k_mix))

  def test_rmsnorm_supports_nontrailing_axis(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4, 2))
    norm = normalizations.RMSNorm(axis=-2)
    variables = norm.init(jax.random.PRNGKey(1), x)
    actual = norm.apply(variables, x)
    expected = normalizations.rms_norm(
        x, dtype=x.dtype, epsilon=_RMS_EPSILON, axis=-2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.mean(actual ** 2, axis=-2), jnp.ones((2, 3, 2)),
        rtol=1e-5, atol=1e-5)

  def test_factorized_head_read_implementations_match_gradients(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(59), 7)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, k + v)),
        jax.random.normal(random[3], (e, n, 2)),
        jax.random.normal(random[4], (b, t, 2)),
    )
    upstream = jax.random.normal(random[5], (b, n, t, k + v))

    def output(values, implementation):
      M, x, key_kernel, mix_kernel, gates = values
      projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)
      head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)
      return factorized_head_bam_read(
          M, x, projection, head_projection, key_mode='rms_gate',
          key_scale=2.0, rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation=implementation)

    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'dot_bnt') * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'mul_reduce_btn') * upstream))(args)
    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
    for got, expected in zip(actual_grad, reference_grad):
      np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_batched_factorized_qk_read_matches_separate_reads(self):
    b, t, qk, n, k, v, e = 2, 3, 2, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(71), 6)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, qk, k + v)),
        jax.random.normal(random[3], (e, qk, n, 2)),
        jax.random.normal(random[4], (b, t, qk, 2)),
    )
    upstream = jax.random.normal(random[5], (b, t, qk, n, k + v))

    def combined(values):
      M, x, key_kernel, mix_kernel, gates = values
      projected_key = jnp.einsum('bte,eqd->btqd', x, key_kernel)
      raw_mix = jnp.einsum('bte,eqnr->btqnr', x, mix_kernel)
      y_u, y_v = bam_read(
          M, x, lambda _x: projected_key, None, key_mode='rms_gate',
          key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gates, implementation='mul_reduce_btn',
          return_sides=True)
      mix = normalizations.rms_norm(
          raw_mix, dtype=y_u.dtype, epsilon=_RMS_EPSILON, axis=-2)
      row_mix, col_mix = mix[..., 0], mix[..., 1]
      y_u = jnp.einsum('btqk,btqn->btqnk', y_u, col_mix)
      y_v = jnp.einsum('btqv,btqn->btqnv', y_v, row_mix)
      return jnp.concatenate((y_u, y_v), axis=-1)

    def separate(values):
      M, x, key_kernel, mix_kernel, gates = values
      outputs = []
      for index in range(qk):
        projection = lambda z, i=index: jnp.einsum(
            'bte,ed->btd', z, key_kernel[:, i])
        mix_projection = lambda z, i=index: jnp.einsum(
            'bte,enr->btnr', z, mix_kernel[:, i])
        outputs.append(factorized_head_bam_read(
            M, x, projection, mix_projection, key_mode='rms_gate',
            key_scale=2.0, rms_epsilon=_RMS_EPSILON,
            key_gate_logits=gates[:, :, index],
            implementation='mul_reduce_btn', output_layout='btn'))
      return jnp.stack(outputs, axis=2)

    reference = separate(args)
    actual = combined(args)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(separate(z) * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(combined(z) * upstream))(args)
    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
    for got, expected in zip(actual_grad, reference_grad):
      np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_factorized_head_read_supports_shared_learned_key_norm(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(61), 5)
    M = jax.random.normal(random[0], (b, t, k, v))
    x = jax.random.normal(random[1], (b, t, e))
    key_kernel = jax.random.normal(random[2], (e, k + v))
    mix_kernel = jax.random.normal(random[3], (e, n, 2))
    gates = jax.random.normal(random[4], (b, t, 2))
    projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)
    head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gates, implementation='mul_reduce_btn')
    baseline = factorized_head_bam_read(
        M, x, projection, head_projection, **kwargs)
    identity_norm = lambda z: normalizations.rms_norm(
        z, dtype=z.dtype, epsilon=_RMS_EPSILON)
    identity = factorized_head_bam_read(
        M, x, projection, head_projection, key_row_norm=identity_norm,
        key_col_norm=identity_norm, use_learned_key_norm=True, **kwargs)
    np.testing.assert_allclose(identity, baseline, rtol=1e-6, atol=1e-6)

    scaled_norm = lambda z: 1.5 * normalizations.rms_norm(
        z, dtype=z.dtype, epsilon=_RMS_EPSILON)
    scaled = factorized_head_bam_read(
        M, x, projection, head_projection, key_row_norm=scaled_norm,
        key_col_norm=scaled_norm, use_learned_key_norm=True, **kwargs)
    np.testing.assert_allclose(scaled, 1.5 * baseline, rtol=2e-5, atol=2e-5)

  def test_codebook_source_and_destination_implementations_match_gradients(self):
    b, t, n, c, k, v, e = 2, 3, 4, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(53), 8)
    args = (
        jax.nn.softmax(jax.random.normal(random[0], (b, 1, t, t)), axis=-1),
        jax.random.normal(random[1], (b, t, k, v)),
        jax.random.normal(random[2], (b, t, e)),
        jax.random.normal(random[3], (c, k)),
        jax.random.normal(random[4], (c, v)),
        jax.random.normal(random[5], (e, n, 1, 2 * c)),
        jax.random.normal(random[6], (b, t, n, 2)),
    )
    upstream = jax.random.normal(random[7], (b, t, n, k + v))

    def output(values, source_implementation, read_implementation):
      alpha, M, x, rho_u, rho_v, kernel, gates = values
      projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)
      return codebook_read(
          alpha, M, x, rho_u, rho_v, projection,
          key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gates, source_implementation=source_implementation,
          read_implementation=read_implementation)

    reference = output(args, 'dot', 'dot_btn')
    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, 'dot', 'dot_btn') * upstream))(args)
    for source_implementation, read_implementation in (
        ('mul_reduce', 'dot_btn'),
        ('dot', 'mul_reduce_btn'),
        ('mul_reduce', 'mul_reduce_btn'),
    ):
      actual = output(args, source_implementation, read_implementation)
      actual_value, actual_grad = jax.value_and_grad(
          lambda z: jnp.sum(output(
              z, source_implementation, read_implementation) * upstream))(args)
      np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
      for got, expected in zip(actual_grad, reference_grad):
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_bam_read_implementations_match_values_and_gradients(self):
    b, t, n, f, k, v, e = 2, 3, 4, 2, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(41), 7)

    for fetched in (False, True):
      M_shape = (b, f, t, k, v) if fetched else (b, t, k, v)
      key_shape = (e, n, f, k + v) if fetched else (e, n, k + v)
      gate_shape = (b, t, n, f, 2) if fetched else (b, t, n, 2)
      args = (
          jax.random.normal(random[0], M_shape),
          jax.random.normal(random[1], (b, t, e)),
          jax.random.normal(random[2], key_shape),
          jax.random.normal(random[3], gate_shape),
      )
      upstream = jax.random.normal(random[4], (b, t, n, k + v))

      def output(values, implementation):
        M, x, kernel, gates = values
        projection = lambda z: jnp.einsum(
            'bte,enfD->btnfD', z, kernel) if fetched else jnp.einsum(
                'bte,enD->btnD', z, kernel)
        y = bam_read(
            M, x, projection, None, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
            implementation=implementation)
        return rearrange(y, 'b n t d -> b t n d') if implementation == 'dot_bnt' else y

      reference = output(args, 'dot_bnt')
      reference_value, reference_grad = jax.value_and_grad(
          lambda z: jnp.sum(output(z, 'dot_bnt') * upstream))(args)
      for implementation in ('dot_btn', 'mul_reduce_btn'):
        actual = output(args, implementation)
        actual_value, actual_grad = jax.value_and_grad(
            lambda z: jnp.sum(output(z, implementation) * upstream))(args)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_value, reference_value, rtol=1e-5, atol=1e-5)
        for got, expected in zip(actual_grad, reference_grad):
          np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

  def test_bam_read_return_sides_matches_joined_output(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(73), 6)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, n, k + v)),
        jax.random.normal(random[3], (b, t, n, 2)),
    )
    upstream = jax.random.normal(random[4], (b, t, n, k + v))

    def output(values, return_sides):
      M, x, kernel, gates = values
      projection = lambda z: jnp.einsum('bte,end->btnd', z, kernel)
      y = bam_read(
          M, x, projection, None, key_mode='rms_gate', key_scale=2.0,
          rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation='mul_reduce_btn', return_sides=return_sides)
      return jnp.concatenate(y, axis=-1) if return_sides else y

    reference_value, reference_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, False) * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, True) * upstream))(args)
    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-6, atol=1e-6)
    for got, expected in zip(actual_grad, reference_grad):
      np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

  def test_one_sided_bam_reads_keep_selected_half(self):
    b, t, n, f, k, v, e = 2, 3, 4, 2, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(67), 6)
    x = jax.random.normal(random[0], (b, t, e))
    gates = jax.random.normal(random[1], (b, t, n, f, 2))
    M = jax.random.normal(random[2], (b, f, t, k, v))
    kernel = jax.random.normal(random[3], (e, n, f, k + v))
    projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)

    for implementation in ('dot_bnt', 'dot_btn', 'mul_reduce_btn'):
      outputs = {}
      for read_side in ('both', 'row', 'col'):
        y = bam_read(
            M, x, projection, None, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
            implementation=implementation, read_side=read_side)
        outputs[read_side] = (
            rearrange(y, 'b n t d -> b t n d')
            if implementation == 'dot_bnt' else y)
      np.testing.assert_array_equal(outputs['row'][..., :k], 0)
      np.testing.assert_allclose(
          outputs['row'][..., k:], outputs['both'][..., k:], rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(
          outputs['col'][..., :k], outputs['both'][..., :k], rtol=1e-5, atol=1e-5)
      np.testing.assert_array_equal(outputs['col'][..., k:], 0)

    local_M = jax.random.normal(random[4], (b, t, k, v))
    key_kernel = jax.random.normal(random[3], (e, k + v))
    mix_kernel = jax.random.normal(random[5], (e, n, 2))
    key_projection = lambda z: jnp.einsum('bte,eD->btD', z, key_kernel)
    mix_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)
    local_gates = gates[..., 0, :]
    both = factorized_head_bam_read(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn')
    row = factorized_head_bam_read(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn', read_side='row')
    col = factorized_head_bam_read(
        local_M, x, key_projection, mix_projection, key_mode='rms_gate',
        key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=local_gates,
        implementation='mul_reduce_btn', read_side='col')
    np.testing.assert_array_equal(row[..., :k], 0)
    np.testing.assert_allclose(row[..., k:], both[..., k:], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(col[..., :k], both[..., :k], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(col[..., k:], 0)

  def test_single_fetch_axis_squeeze_matches_values_and_gradients(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(47), 5)
    args = (
        jax.random.normal(random[0], (b, 1, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, n, 1, k + v)),
        jax.random.normal(random[3], (b, t, n, 1, 2)),
    )
    upstream = jax.random.normal(random[4], (b, t, n, k + v))

    def output(values, squeeze):
      M, x, kernel, gates = values
      if squeeze:
        projection = lambda z: jnp.einsum('bte,enD->btnD', z, kernel[:, :, 0])
        return bam_read(
            M[:, 0], x, projection, None, key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON, key_gate_logits=gates[..., 0, :],
            implementation='dot_btn')
      projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)
      return bam_read(
          M, x, projection, None, key_mode='rms_gate', key_scale=2.0,
          rms_epsilon=_RMS_EPSILON, key_gate_logits=gates,
          implementation='dot_btn')

    expected = output(args, False)
    actual = output(args, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    expected_value, expected_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, False) * upstream))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: jnp.sum(output(z, True) * upstream))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    for got, expected_grad_item in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got, expected_grad_item, rtol=2e-5, atol=2e-5)

  def test_factorized_head_read_matches_explicit_rank_one_keys(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    keys = jax.random.split(jax.random.PRNGKey(29), 6)
    M = jax.random.normal(keys[0], (b, t, k, v))
    x = jax.random.normal(keys[1], (b, t, e))
    key_kernel = jax.random.normal(keys[2], (e, k + v))
    mix_kernel = jax.random.normal(keys[3], (e, n, 2))
    gate_kernel = jax.random.normal(keys[4], (e, 2))
    gate_bias = jax.random.normal(keys[5], (2,))
    projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)
    head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)
    gate_logits = jnp.einsum('bte,er->btr', x, gate_kernel) + gate_bias
    kwargs = dict(
        key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
        key_gate_logits=gate_logits)

    actual = factorized_head_bam_read(
        M, x, projection, head_projection, **kwargs)
    actual_mul = factorized_head_bam_read(
        M, x, projection, head_projection, **kwargs,
        implementation='mul_reduce_btn')
    raw_row, raw_col = jnp.split(projection(x), [k], axis=-1)
    row_gate, col_gate = jnp.split(gate_logits, 2, axis=-1)
    row = _transform_bam_read_key(
        raw_row, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=row_gate)
    col = _transform_bam_read_key(
        raw_col, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=col_gate)
    raw_mix = head_projection(x)
    mix = normalizations.rms_norm(
        raw_mix, dtype=raw_mix.dtype, epsilon=_RMS_EPSILON, axis=-2)
    explicit_row = row[:, :, None, :] * mix[..., 0, None]
    explicit_col = col[:, :, None, :] * mix[..., 1, None]
    expected = jnp.concatenate([
        jnp.einsum('btkv,btnv->bntk', M, explicit_col),
        jnp.einsum('btkv,btnk->bntv', M, explicit_row),
    ], axis=-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_mul, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        jnp.mean(mix ** 2, axis=-2), jnp.ones((b, t, 2)),
        rtol=2e-3, atol=2e-3)

  def test_factorized_head_read_zero_key_starts_dormant_but_has_key_gradient(self):
    b, t, n, k, v, e = 1, 3, 4, 3, 5, 7
    M = jax.random.normal(jax.random.PRNGKey(31), (b, t, k, v))
    x = jax.random.normal(jax.random.PRNGKey(32), (b, t, e))
    mix_kernel = jax.random.normal(jax.random.PRNGKey(33), (e, n, 2))
    upstream = jax.random.normal(jax.random.PRNGKey(34), (b, n, t, k + v))
    head_projection = lambda z: jnp.einsum('bte,enr->btnr', z, mix_kernel)
    gate_init = np.sqrt(_RMS_EPSILON) / 2.0
    gate_logits = jnp.full((b, t, 2), np.log(gate_init / (1.0 - gate_init)))

    def objective(key_kernel):
      projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)
      y = factorized_head_bam_read(
          M, x, projection, head_projection, key_mode='rms_gate',
          key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gate_logits)
      return jnp.sum(y * upstream), y

    (value, y), grad = jax.value_and_grad(objective, has_aux=True)(
        jnp.zeros((e, k + v)))
    self.assertEqual(float(value), 0.0)
    np.testing.assert_array_equal(y, jnp.zeros_like(y))
    self.assertGreater(float(jnp.linalg.norm(grad)), 0.0)

  def test_combined_shared_read_matches_separate_value_and_gradients(self):
    """Read(F, r) + Read(L, r) == Read(F + L, r), including shared-key grads."""
    b, t, n, k, v, e = 2, 5, 3, 4, 4, 7
    keys = jax.random.split(jax.random.PRNGKey(17), 7)

    def make_inputs(dtype):
      return (
          jax.random.normal(keys[0], (b, t, k, v), dtype=dtype),
          jax.random.normal(keys[1], (b, 1, t, t), dtype=jnp.float32),
          jax.random.normal(keys[2], (b, t, e), dtype=dtype),
          jax.random.normal(keys[3], (e, n, 1, k + v), dtype=jnp.float32),
          jax.random.normal(keys[4], (e, n, 1, 2), dtype=jnp.float32),
      )

    upstream = jax.random.normal(
        keys[5], (b, n, t, k + v), dtype=jnp.bfloat16).astype(jnp.float32)
    gate_bias = jax.random.normal(keys[6], (n, 1, 2), dtype=jnp.float32)

    def read_output(args, combine):
      Mh, fetch_alpha, x, read_kernel, gate_kernel = args
      compute_dtype = x.dtype
      projection = lambda z: jnp.einsum(
          'bte,enfD->btnfD', z.astype(compute_dtype), read_kernel.astype(compute_dtype))
      gate_logits = jnp.einsum(
          'bte,enfg->btnfg', x, gate_kernel.astype(compute_dtype)) + gate_bias
      eye = jnp.eye(t, dtype=fetch_alpha.dtype)[None, None]
      offdiag_alpha = fetch_alpha * (1 - eye)
      routed_alpha = offdiag_alpha + eye if combine == 'diag_one' else offdiag_alpha
      Mbar = jnp.einsum('bfts,bskv->bftkv', routed_alpha, Mh)
      kwargs = dict(
          key_mode='rms_gate', key_scale=2.0, rms_epsilon=_RMS_EPSILON,
          key_gate_logits=gate_logits)
      if combine == 'diag_one':
        y = bam_read(Mbar, x, projection, None, **kwargs)
      elif combine:
        y = bam_read(Mbar + Mh[:, None], x, projection, None, **kwargs)
      else:
        y = bam_read(Mbar, x, projection, None, **kwargs)
        y += bam_read(
            Mh, x, lambda z: jnp.squeeze(projection(z), axis=-2), None,
            key_mode='rms_gate', key_scale=2.0,
            rms_epsilon=_RMS_EPSILON,
            key_gate_logits=jnp.squeeze(gate_logits, axis=-2))
      return y

    def objective(args, combine):
      return jnp.sum(jnp.asarray(read_output(args, combine), jnp.float32) * upstream)

    backend = jax.default_backend()
    for dtype, cpu_relative_limit in (
        (jnp.float32, 2e-6),
        (jnp.bfloat16, 2e-2),
    ):
      # TPU's default dot precision may use reduced-precision products even for
      # float32 operands.  Keep this test diagnostic on TPU while retaining a
      # strict algebraic check on CPU.
      relative_limit = cpu_relative_limit if backend == 'cpu' else 2e-2
      args = make_inputs(dtype)
      old_output = np.asarray(jax.jit(lambda values: read_output(values, False))(args), np.float32)
      new_output = np.asarray(jax.jit(lambda values: read_output(values, True))(args), np.float32)
      output_diff = new_output - old_output
      output_relative_l2 = np.linalg.norm(output_diff) / max(np.linalg.norm(old_output), 1e-12)
      print(
          f'combined_read backend={backend} dtype={dtype} '
          f'output_rel_l2={output_relative_l2:.3e} '
          f'output_max_abs={np.max(np.abs(output_diff)):.3e}')
      self.assertLess(output_relative_l2, relative_limit)
      separate = jax.jit(jax.value_and_grad(lambda values: objective(values, False)))
      combined = jax.jit(jax.value_and_grad(lambda values: objective(values, True)))
      old_value, old_grads = separate(args)
      new_value, new_grads = combined(args)
      old_value_f = float(old_value)
      value_rel = abs(float(new_value) - old_value_f) / max(abs(old_value_f), 1e-12)
      print(f'combined_read dtype={dtype} value_rel={value_rel:.3e}')
      self.assertLess(value_rel, relative_limit)
      for index, (new_grad, old_grad) in enumerate(zip(new_grads, old_grads)):
        new_grad = np.asarray(new_grad, dtype=np.float32)
        old_grad = np.asarray(old_grad, dtype=np.float32)
        diff = new_grad - old_grad
        relative_l2 = np.linalg.norm(diff) / max(np.linalg.norm(old_grad), 1e-12)
        print(
            f'combined_read dtype={dtype} grad={index} '
            f'rel_l2={relative_l2:.3e} max_abs={np.max(np.abs(diff)):.3e}')
        self.assertLess(relative_l2, relative_limit)

      diag_output = np.asarray(
          jax.jit(lambda values: read_output(values, 'diag_one'))(args), np.float32)
      diag_output_diff = diag_output - new_output
      diag_output_relative_l2 = np.linalg.norm(diag_output_diff) / max(
          np.linalg.norm(new_output), 1e-12)
      print(
          f'diag_one backend={backend} dtype={dtype} '
          f'output_rel_l2_vs_add_local={diag_output_relative_l2:.3e} '
          f'output_max_abs={np.max(np.abs(diag_output_diff)):.3e}')
      self.assertLess(diag_output_relative_l2, relative_limit)
      diag_value, diag_grads = jax.jit(
          jax.value_and_grad(lambda values: objective(values, 'diag_one')))(args)
      diag_value_rel = abs(float(diag_value) - float(new_value)) / max(abs(float(new_value)), 1e-12)
      print(f'diag_one dtype={dtype} value_rel_vs_add_local={diag_value_rel:.3e}')
      self.assertLess(diag_value_rel, relative_limit)
      for index, (diag_grad, combined_grad) in enumerate(zip(diag_grads, new_grads)):
        diag_grad = np.asarray(diag_grad, dtype=np.float32)
        combined_grad = np.asarray(combined_grad, dtype=np.float32)
        diff = diag_grad - combined_grad
        relative_l2 = np.linalg.norm(diff) / max(np.linalg.norm(combined_grad), 1e-12)
        print(
            f'diag_one dtype={dtype} grad={index} '
            f'rel_l2_vs_add_local={relative_l2:.3e} max_abs={np.max(np.abs(diff)):.3e}')
        self.assertLess(relative_l2, relative_limit)

  def test_grouped_rmsnorm_has_independent_group_scales(self):
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)
    norm = GroupedRMSNorm(
        scale_shape=(3, 4), epsilon=_RMS_EPSILON, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None))
    variables = norm.init(jax.random.PRNGKey(0), x)
    scale_param = variables['params']['scale']
    self.assertEqual(scale_param.value.shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    scale = jnp.zeros((3, 4)).at[1].set(1.0)
    scaled = norm.apply({'params': {'scale': scale}}, x)
    np.testing.assert_allclose(scaled[:, 0], expected[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 1], 2.0 * expected[:, 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 2], expected[:, 2], rtol=1e-6, atol=1e-6)

  def test_grouped_rmsnorm_supports_independent_group_biases(self):
    x = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4)
    norm = GroupedRMSNorm(
        scale_shape=(3, 4), epsilon=_RMS_EPSILON, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None), use_bias=True)
    variables = norm.init(jax.random.PRNGKey(0), x)
    self.assertEqual(variables['params']['bias'].value.shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    bias = jnp.zeros((3, 4)).at[1].set(jnp.arange(4, dtype=jnp.float32))
    biased = norm.apply(
        {'params': {'scale': jnp.zeros((3, 4)), 'bias': bias}}, x)
    np.testing.assert_allclose(biased[:, 0], expected[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        biased[:, 1], expected[:, 1] + bias[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(biased[:, 2], expected[:, 2], rtol=1e-6, atol=1e-6)

  def test_constant_matrix_update_matches_existing_decay(self):
    M_in = jnp.arange(12, dtype=jnp.float32).reshape(1, 1, 3, 4)
    dM = jnp.ones_like(M_in)
    M_out, forget_gate = _update_bam_matrix(M_in, dM, 0.75)
    np.testing.assert_array_equal(M_out, 0.75 * M_in + dM)
    self.assertIsNone(forget_gate)

  def test_dynamic_forget_starts_at_requested_retention_and_has_gradient(self):
    forget_init = 0.01
    logits = jnp.full((1, 2, 1), np.log(forget_init / (1.0 - forget_init)))
    M_in = jnp.ones((1, 2, 3, 4))
    dM = jnp.zeros_like(M_in)
    M_out, forget_gate = _update_bam_matrix(M_in, dM, 1.0, logits)
    np.testing.assert_allclose(forget_gate, forget_init, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(M_out, 1.0 - forget_init, rtol=1e-6, atol=1e-7)
    grad = jax.grad(
        lambda z: _update_bam_matrix(M_in, dM, 1.0, z)[0].sum())(logits)
    self.assertTrue(np.all(np.asarray(grad) < 0.0))

  def test_write_v_mix_starts_as_local_projection(self):
    x_v = jax.random.normal(jax.random.PRNGKey(3), (2, 5, 4, 3))
    o_head = jax.random.normal(jax.random.PRNGKey(4), (2, 5, 4, 6))
    scale = jnp.tile(jnp.array([[1.0, 0.0]]), (4, 1))
    bias = jnp.zeros((4, 3))
    mixed = _mix_bam_write_v(x_v, o_head, 3, scale, bias)
    np.testing.assert_array_equal(mixed, x_v)

  def test_write_v_mix_selects_output_tail_and_bias(self):
    x_v = jnp.zeros((1, 2, 2, 3))
    o_head = jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 2, 3)
    scale = jnp.tile(jnp.array([[0.0, 1.0]]), (2, 1))
    bias = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    mixed = _mix_bam_write_v(x_v, o_head, 0, scale, bias)
    np.testing.assert_array_equal(mixed, o_head + bias)

  def test_shared_fetch_implementations_match_values_and_gradients(self):
    q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8)) / np.sqrt(8)
    k = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4, 8))
    causal = jnp.where(jnp.tril(jnp.ones((1, 1, 5, 5), dtype=bool)), 0.0, -jnp.inf)
    weights = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 5, 5))

    def objective(query, mode, soft_cap):
      logits = jnp.einsum('btnd,bsnd->bnts', query, k)
      if soft_cap:
        logits = jnp.tanh(logits / soft_cap) * soft_cap
      logits = jnp.where(causal == 0, logits, causal)
      alpha = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
      fetch = _shared_bam_fetch_alpha(
          alpha, query, k, causal, 2, mode, True, soft_cap, True)
      return jnp.sum(fetch * weights)

    for soft_cap in (0.0, 3.0):
      values = [objective(q, mode, soft_cap) for mode in ('legacy', 'compact', 'recompute')]
      grads = [jax.grad(objective)(q, mode, soft_cap)
               for mode in ('legacy', 'compact', 'recompute')]
      for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-6, atol=1e-6)
      for grad in grads[1:]:
        np.testing.assert_allclose(grad, grads[0], rtol=1e-5, atol=1e-6)

  def test_dynamic_fetch_is_tokenwise_convex_head_mix(self):
    alpha = jnp.arange(1, 49, dtype=jnp.float32).reshape(1, 3, 4, 4)
    alpha = alpha / alpha.sum(axis=-1, keepdims=True)
    logits = jnp.zeros((1, 4, 3), dtype=jnp.float32)
    mixed = _dynamic_mixed_bam_fetch_alpha(
        alpha, logits, False, rms_epsilon=_RMS_EPSILON)
    np.testing.assert_allclose(mixed[:, 0], alpha.mean(axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mixed.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)

    yielded = _dynamic_mixed_bam_fetch_alpha(
        alpha, logits, True, rms_epsilon=_RMS_EPSILON)
    diagonal = jnp.diagonal(yielded[:, 0], axis1=-2, axis2=-1)
    np.testing.assert_array_equal(diagonal, jnp.zeros_like(diagonal))

  def test_dynamic_fetch_supports_signed_rms_head_mix(self):
    alpha = jnp.arange(1, 25, dtype=jnp.float32).reshape(1, 3, 2, 4)
    logits = jnp.array([[[1.0, -2.0, 3.0], [-3.0, 2.0, -1.0]]])
    mixed = _dynamic_mixed_bam_fetch_alpha(
        alpha, logits, False, weight_mode='rms',
        rms_epsilon=_RMS_EPSILON)
    weights = logits * jax.lax.rsqrt(
        jnp.mean(logits ** 2, axis=-1, keepdims=True)
        + _RMS_EPSILON) / jnp.sqrt(logits.shape[-1])
    expected = jnp.einsum('bnts,btn->bts', alpha, weights)
    np.testing.assert_allclose(mixed[:, 0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(weights ** 2, axis=-1)), 1.0, rtol=1e-6, atol=1e-6)

    mixed_aux, raw_logits, actual_weights, pre_diagonal = _dynamic_mixed_bam_fetch_alpha(
        alpha, logits, False, weight_mode='rms',
        rms_epsilon=_RMS_EPSILON, return_aux=True)
    np.testing.assert_array_equal(mixed_aux, mixed)
    np.testing.assert_array_equal(raw_logits, logits)
    np.testing.assert_allclose(actual_weights, weights, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(pre_diagonal, mixed)

  def test_temporal_block_fetch_is_causal_and_segment_aware(self):
    alpha = jnp.tril(jnp.ones((1, 1, 8, 8), dtype=jnp.float32))
    alpha = alpha / alpha.sum(axis=-1, keepdims=True)
    matrix = jnp.arange(8, dtype=jnp.float32).reshape(1, 8, 1, 1)
    positions = jnp.arange(8, dtype=jnp.int32)[None]
    segments = jnp.ones_like(positions)

    mean = _temporal_block_bam_fetch(
        alpha, matrix, positions, segments, 4, 'mean')
    # Current block is exact. At t=4, block 0 has mean 1.5 and token 4 is exact.
    np.testing.assert_allclose(mean[0, 0, :4, 0, 0],
                               jnp.einsum('ts,s->t', alpha[0, 0, :4], matrix[0, :, 0, 0]))
    np.testing.assert_allclose(mean[0, 0, 4, 0, 0], (4 * 1.5 + 4.0) / 5.0)

    # A new packed segment must not contaminate the first segment's block summary.
    packed_positions = jnp.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=jnp.int32)
    packed_segments = jnp.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=jnp.int32)
    packed_alpha = alpha.at[:, :, 4:, :4].set(0)
    packed_alpha = packed_alpha / jnp.maximum(packed_alpha.sum(axis=-1, keepdims=True), 1)
    packed = _temporal_block_bam_fetch(
        packed_alpha, matrix, packed_positions, packed_segments, 4, 'mean')
    exact = jnp.einsum('bfts,bskv->bftkv', packed_alpha, matrix)
    np.testing.assert_allclose(packed, exact)

  def test_temporal_linear_reconstructs_linear_matrix_history(self):
    alpha = jnp.tril(jnp.ones((1, 1, 8, 8), dtype=jnp.float32))
    alpha = alpha / alpha.sum(axis=-1, keepdims=True)
    positions = jnp.arange(8, dtype=jnp.int32)[None]
    segments = jnp.ones_like(positions)
    within = (positions % 4).astype(jnp.float32)
    matrix = (3.0 + 2.0 * within)[..., None, None]
    actual = _temporal_block_bam_fetch(
        alpha, matrix, positions, segments, 4, 'linear')
    exact = jnp.einsum('bfts,bskv->bftkv', alpha, matrix)
    np.testing.assert_allclose(actual, exact, rtol=1e-5, atol=1e-5)

  def test_fetch_sliding_window_masks_post_mix_without_renormalizing(self):
    mixed = jax.random.normal(jax.random.PRNGKey(7), (2, 1, 6, 6))
    actual = _sliding_window_bam_fetch_alpha(mixed, 3)
    target = jnp.arange(6)[:, None]
    source = jnp.arange(6)[None, :]
    sliding = (source <= target) & (source > target - 3)
    expected = jnp.where(sliding, mixed, 0)
    np.testing.assert_array_equal(actual, expected)

    positions = jnp.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]])
    with_prefix = _sliding_window_bam_fetch_alpha(mixed, 3, 2, positions)
    prefix = positions[:, None, None, :] < 2
    expected_with_prefix = jnp.where(sliding[None, None] | prefix, mixed, 0)
    np.testing.assert_array_equal(with_prefix, expected_with_prefix)
    with self.assertRaisesRegex(ValueError, 'source_positions'):
      _sliding_window_bam_fetch_alpha(mixed, 3, 2)

  def test_write_source_selection(self):
    terms = [jnp.full((2,), value) for value in (1.0, 2.0, 4.0, 8.0)]
    np.testing.assert_array_equal(
        _select_bam_write_source('std', *terms), jnp.full((2,), 1.0))
    np.testing.assert_array_equal(
        _select_bam_write_source('std+cross', *terms), jnp.full((2,), 7.0))
    np.testing.assert_array_equal(
        _select_bam_write_source('std+cross+local_o', *terms), jnp.full((2,), 15.0))
    y_all = jnp.full((2,), 16.0)
    self.assertIs(_select_bam_write_source('std+cross+local_o', *terms, y_all), y_all)

  def test_soft_rms_cap_is_identity_at_zero_and_bounded(self):
    scale = 2.0
    jacobian = jax.jacfwd(
        lambda z: _transform_bam_read_key(
            z, 'soft_rms_cap', scale,
            rms_epsilon=_RMS_EPSILON))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-6, atol=1e-6)

    large = jnp.array([30.0, 40.0])
    transformed = _transform_bam_read_key(
        large, 'soft_rms_cap', scale, rms_epsilon=_RMS_EPSILON)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    self.assertLess(float(transformed_rms), scale)
    self.assertGreater(float(transformed_rms), 0.99 * scale)

  def test_rms_gate_bias_calibration_preserves_zero_jacobian(self):
    scale = 2.0
    initial_gate = np.sqrt(_RMS_EPSILON) / scale
    gate_logits = jnp.full((1,), np.log(initial_gate / (1.0 - initial_gate)))
    jacobian = jax.jacfwd(
        lambda z: _transform_bam_read_key(
            z, 'rms_gate', scale, rms_epsilon=_RMS_EPSILON,
            gate_logits=gate_logits))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-5, atol=1e-5)

  def test_rms_gate_has_requested_rms(self):
    scale = 2.0
    gate = 0.25
    gate_logits = jnp.full((1,), np.log(gate / (1.0 - gate)))
    transformed = _transform_bam_read_key(
        jnp.array([3.0, 4.0]), 'rms_gate', scale,
        rms_epsilon=_RMS_EPSILON, gate_logits=gate_logits)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    np.testing.assert_allclose(transformed_rms, scale * gate, rtol=1e-6, atol=1e-6)

  def test_rms_gate_learned_norm_is_a_paired_identity_control(self):
    r = jnp.array([[3.0, 4.0]], dtype=jnp.float32)
    gate_logits = jnp.zeros((1, 1), dtype=jnp.float32)
    learned_norm = lambda z: 1.5 * normalizations.rms_norm(
        z, dtype=z.dtype, epsilon=_RMS_EPSILON)
    baseline = _transform_bam_read_key(
        r, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=gate_logits)
    dormant = _transform_bam_read_key(
        r, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=gate_logits, learned_rms_norm=learned_norm,
        use_learned_rms=False)
    active = _transform_bam_read_key(
        r, 'rms_gate', 2.0, rms_epsilon=_RMS_EPSILON,
        gate_logits=gate_logits, learned_rms_norm=learned_norm,
        use_learned_rms=True)
    np.testing.assert_array_equal(dormant, baseline)
    np.testing.assert_allclose(active, 1.5 * baseline, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()

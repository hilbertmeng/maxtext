"""Focused tests for BAM runtime read-key transforms."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from layers.attentions import (
    GroupedRMSNorm,
    _dynamic_mixed_bam_fetch_alpha,
    _mix_bam_write_v,
    _rms,
    _select_bam_write_source,
    _shared_bam_fetch_alpha,
    _transform_bam_read_key,
    _update_bam_matrix,
    bam_read,
    factorized_head_bam_read,
    packed_qk_bam_read,
)


class BamReadKeyTransformTest(absltest.TestCase):

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
            key_eps=1e-4, key_gate_logits=gates, implementation=implementation)
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

  def test_packed_qk_read_matches_two_direct_layout_reads_and_gradients(self):
    b, t, n, k, v, e = 2, 3, 4, 3, 5, 7
    random = jax.random.split(jax.random.PRNGKey(43), 8)
    args = (
        jax.random.normal(random[0], (b, t, k, v)),
        jax.random.normal(random[1], (b, t, e)),
        jax.random.normal(random[2], (e, n, k + v)),
        jax.random.normal(random[3], (e, n, k + v)),
        jax.random.normal(random[4], (b, t, n, 2)),
        jax.random.normal(random[5], (b, t, n, 2)),
    )
    q_upstream = jax.random.normal(random[6], (b, t, n, k + v))
    k_upstream = jax.random.normal(random[7], (b, t, n, k + v))

    def outputs(values, packed):
      M, x, q_kernel, k_kernel, q_gates, k_gates = values
      q_projection = lambda z: jnp.einsum('bte,enD->btnD', z, q_kernel)
      k_projection = lambda z: jnp.einsum('bte,enD->btnD', z, k_kernel)
      q_kwargs = dict(
          key_mode='rms_gate', key_scale=2.0, key_eps=1e-4,
          key_gate_logits=q_gates)
      k_kwargs = dict(
          key_mode='rms_gate', key_scale=2.0, key_eps=1e-4,
          key_gate_logits=k_gates)
      if packed:
        return packed_qk_bam_read(
            M, x, q_projection, k_projection, q_kwargs, k_kwargs)
      return (
          bam_read(M, x, q_projection, None, **q_kwargs, implementation='dot_btn'),
          bam_read(M, x, k_projection, None, **k_kwargs, implementation='dot_btn'),
      )

    def objective(values, packed):
      q, k_out = outputs(values, packed)
      return jnp.sum(q * q_upstream) + jnp.sum(k_out * k_upstream)

    expected_q, expected_k = outputs(args, False)
    actual_q, actual_k = outputs(args, True)
    np.testing.assert_allclose(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_k, expected_k, rtol=1e-5, atol=1e-5)
    expected_value, expected_grad = jax.value_and_grad(
        lambda z: objective(z, False))(args)
    actual_value, actual_grad = jax.value_and_grad(
        lambda z: objective(z, True))(args)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    for got, expected in zip(actual_grad, expected_grad):
      np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)

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
            key_eps=1e-4, key_gate_logits=gates[..., 0, :],
            implementation='dot_btn')
      projection = lambda z: jnp.einsum('bte,enfD->btnfD', z, kernel)
      return bam_read(
          M, x, projection, None, key_mode='rms_gate', key_scale=2.0,
          key_eps=1e-4, key_gate_logits=gates, implementation='dot_btn')

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
        key_mode='rms_gate', key_scale=2.0, key_eps=1e-4,
        key_gate_logits=gate_logits)

    actual = factorized_head_bam_read(
        M, x, projection, head_projection, **kwargs)
    raw_row, raw_col = jnp.split(projection(x), [k], axis=-1)
    row_gate, col_gate = jnp.split(gate_logits, 2, axis=-1)
    row = _transform_bam_read_key(raw_row, 'rms_gate', 2.0, 1e-4, row_gate)
    col = _transform_bam_read_key(raw_col, 'rms_gate', 2.0, 1e-4, col_gate)
    mix = _rms(head_projection(x), 1e-4, axis=-2)
    explicit_row = row[:, :, None, :] * mix[..., 0, None]
    explicit_col = col[:, :, None, :] * mix[..., 1, None]
    expected = jnp.concatenate([
        jnp.einsum('btkv,btnv->bntk', M, explicit_col),
        jnp.einsum('btkv,btnk->bntv', M, explicit_row),
    ], axis=-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
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
    gate_init = np.sqrt(1e-4) / 2.0
    gate_logits = jnp.full((b, t, 2), np.log(gate_init / (1.0 - gate_init)))

    def objective(key_kernel):
      projection = lambda z: jnp.einsum('bte,ed->btd', z, key_kernel)
      y = factorized_head_bam_read(
          M, x, projection, head_projection, key_mode='rms_gate',
          key_scale=2.0, key_eps=1e-4, key_gate_logits=gate_logits)
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
          key_mode='rms_gate', key_scale=2.0, key_eps=1e-4,
          key_gate_logits=gate_logits)
      if combine == 'diag_one':
        y = bam_read(Mbar, x, projection, None, **kwargs)
      elif combine:
        y = bam_read(Mbar + Mh[:, None], x, projection, None, **kwargs)
      else:
        y = bam_read(Mbar, x, projection, None, **kwargs)
        y += bam_read(
            Mh, x, lambda z: jnp.squeeze(projection(z), axis=-2), None,
            key_mode='rms_gate', key_scale=2.0, key_eps=1e-4,
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
        scale_shape=(3, 4), epsilon=1e-6, dtype=jnp.float32,
        weight_dtype=jnp.float32, kernel_axes=(None, None))
    variables = norm.init(jax.random.PRNGKey(0), x)
    self.assertEqual(variables['params']['scale'].shape, (3, 4))
    expected = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(norm.apply(variables, x), expected, rtol=1e-6, atol=1e-6)

    scale = jnp.zeros((3, 4)).at[1].set(1.0)
    scaled = norm.apply({'params': {'scale': scale}}, x)
    np.testing.assert_allclose(scaled[:, 0], expected[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 1], 2.0 * expected[:, 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scaled[:, 2], expected[:, 2], rtol=1e-6, atol=1e-6)

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
    mixed = _dynamic_mixed_bam_fetch_alpha(alpha, logits, False)
    np.testing.assert_allclose(mixed[:, 0], alpha.mean(axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mixed.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)

    yielded = _dynamic_mixed_bam_fetch_alpha(alpha, logits, True)
    diagonal = jnp.diagonal(yielded[:, 0], axis1=-2, axis2=-1)
    np.testing.assert_array_equal(diagonal, jnp.zeros_like(diagonal))

  def test_dynamic_fetch_supports_signed_rms_head_mix(self):
    alpha = jnp.arange(1, 25, dtype=jnp.float32).reshape(1, 3, 2, 4)
    logits = jnp.array([[[1.0, -2.0, 3.0], [-3.0, 2.0, -1.0]]])
    mixed = _dynamic_mixed_bam_fetch_alpha(
        alpha, logits, False, weight_mode='rms', epsilon=1e-8)
    weights = logits * jax.lax.rsqrt(
        jnp.mean(logits ** 2, axis=-1, keepdims=True) + 1e-8) / jnp.sqrt(logits.shape[-1])
    expected = jnp.einsum('bnts,btn->bts', alpha, weights)
    np.testing.assert_allclose(mixed[:, 0], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(weights ** 2, axis=-1)), 1.0, rtol=1e-6, atol=1e-6)

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
        lambda z: _transform_bam_read_key(z, 'soft_rms_cap', scale))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-6, atol=1e-6)

    large = jnp.array([30.0, 40.0])
    transformed = _transform_bam_read_key(large, 'soft_rms_cap', scale)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    self.assertLess(float(transformed_rms), scale)
    self.assertGreater(float(transformed_rms), 0.99 * scale)

  def test_rms_gate_bias_calibration_preserves_zero_jacobian(self):
    scale = 2.0
    eps = 1e-4
    initial_gate = np.sqrt(eps) / scale
    gate_logits = jnp.full((1,), np.log(initial_gate / (1.0 - initial_gate)))
    jacobian = jax.jacfwd(
        lambda z: _transform_bam_read_key(
            z, 'rms_gate', scale, eps, gate_logits))(jnp.zeros((4,)))
    np.testing.assert_allclose(jacobian, np.eye(4), rtol=1e-5, atol=1e-5)

  def test_rms_gate_has_requested_rms(self):
    scale = 2.0
    gate = 0.25
    gate_logits = jnp.full((1,), np.log(gate / (1.0 - gate)))
    transformed = _transform_bam_read_key(
        jnp.array([3.0, 4.0]), 'rms_gate', scale, 1e-8, gate_logits)
    transformed_rms = jnp.sqrt(jnp.mean(transformed ** 2))
    np.testing.assert_allclose(transformed_rms, scale * gate, rtol=1e-6, atol=1e-6)

  def test_rms_gate_learned_norm_is_a_paired_identity_control(self):
    r = jnp.array([[3.0, 4.0]], dtype=jnp.float32)
    gate_logits = jnp.zeros((1, 1), dtype=jnp.float32)
    learned_norm = lambda z: 1.5 * _rms(z, 1e-8)
    baseline = _transform_bam_read_key(
        r, 'rms_gate', 2.0, 1e-8, gate_logits)
    dormant = _transform_bam_read_key(
        r, 'rms_gate', 2.0, 1e-8, gate_logits, learned_norm, False)
    active = _transform_bam_read_key(
        r, 'rms_gate', 2.0, 1e-8, gate_logits, learned_norm, True)
    np.testing.assert_array_equal(dormant, baseline)
    np.testing.assert_allclose(active, 1.5 * baseline, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()

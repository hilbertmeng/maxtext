"""FP32 SwiGLU chain-rule checks at a captured, fixed local operating point.

This is a local mathematical reference, not a whole-network replay. Returned
errors against the captured BF16 forward/VJP must accompany its interpretation.
"""


BOUNDED_CANDIDATES = (
    ('gpt_oss_7', 'pre_hard', 7., 7., 1.702, 1.),
    ('step_7', 'post_silu_hard', 7., 7., 1., 0.),
    ('deepseek_10', 'pre_hard', 10., 10., 1., 0.),
    ('dual_hard_2', 'pre_hard', 2., 2., 1., 0.),
    ('dual_hard_1', 'pre_hard', 1., 1., 1., 0.),
    ('situ_4_25', 'situ', 4., 25., 1., 0.),
    ('situ_2_2', 'situ', 2., 2., 1., 0.),
    ('product_hard_2', 'product_hard', 2., 2., 1., 0.),
    ('product_tanh_2', 'product_tanh', 2., 2., 1., 0.),
)


def bounded_activation_terms(u, v, kind, gate_limit, value_limit, alpha=1., offset=0.):
  """Return hidden values and their two partial derivatives, including clip ties.

  Published-model arms reproduce their activation formula only; this does not
  transplant their weights, expert routing, normalization, or training recipe.
  """
  import jax
  import jax.numpy as jnp
  below = lambda a, b: jnp.where(a < b, 1., jnp.where(a == b, .5, 0.))
  inside = lambda a, b: jnp.where(jnp.abs(a) < b, 1., jnp.where(jnp.abs(a) == b, .5, 0.))
  if kind == 'situ':
    su, tu, tv = jax.nn.sigmoid(u), jnp.tanh(u / gate_limit), jnp.tanh(v / value_limit)
    a, b = gate_limit * tu * su, value_limit * tv
    da = (1 - tu * tu) * su + gate_limit * tu * su * (1 - su)
    db = 1 - tv * tv
    return a * b, da * b, a * db
  z = jnp.minimum(u, gate_limit) if kind == 'pre_hard' else u
  s = jax.nn.sigmoid(alpha * z)
  a, da = z * s, s + alpha * z * s * (1 - s)
  if kind == 'pre_hard':
    b = jnp.clip(v, -value_limit, value_limit) + offset
    return a * b, da * below(u, gate_limit) * b, a * inside(v, value_limit)
  if kind == 'post_silu_hard':
    b = jnp.clip(v, -value_limit, value_limit)
    return jnp.minimum(a, gate_limit) * b, da * below(a, gate_limit) * b, jnp.minimum(a, gate_limit) * inside(v, value_limit)
  product = a * v
  if kind == 'product_hard':
    slope = inside(product, gate_limit)
    return jnp.clip(product, -gate_limit, gate_limit), slope * da * v, slope * a
  if kind == 'product_tanh':
    t = jnp.tanh(product / gate_limit)
    slope = 1 - t * t
    return gate_limit * t, slope * da * v, slope * a
  raise ValueError(kind)


def chain_rule(x, dy, wg, wu, wo):
  import jax
  import jax.numpy as jnp
  mm = lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
  u, v = mm(x, wg), mm(x, wu)
  sigmoid = jax.nn.sigmoid(u)
  activation = u * sigmoid
  derivative = sigmoid + u * sigmoid * (1 - sigmoid)
  hidden_cotangent = mm(dy, wo.T)
  gate_cotangent = hidden_cotangent * derivative * v
  value_cotangent = hidden_cotangent * activation
  dx_gate, dx_value = mm(gate_cotangent, wg.T), mm(value_cotangent, wu.T)
  return dict(y=mm(activation * v, wo), dx=dx_gate + dx_value,
              dx_gate=dx_gate, dx_value=dx_value, u=u, v=v,
              activation=activation, derivative=derivative,
              gate_cotangent=gate_cotangent, value_cotangent=value_cotangent)


def analyze(params, x, dy, actual_dx, actual_y, target):
  import jax
  import jax.numpy as jnp
  import numpy as np
  weights = [params[n]['kernel'] for n in ('wi_0', 'wi_1', 'wo')]
  x, dy, actual_dx, actual_y = [jnp.asarray(a, jnp.float32).reshape((-1, a.shape[-1]))
                                for a in (x, dy, actual_dx, actual_y)]

  def summary(x, dy, dx_ref, y_ref, wg, wu, wo):
    mm = lambda a, b: jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
    r = chain_rule(x, dy, wg, wu, wo)
    l2 = lambda a: jnp.sqrt(jnp.sum(a * a))
    token_dx = jnp.linalg.norm(r['dx'], axis=-1)
    token_dy = jnp.linalg.norm(dy, axis=-1)
    peak = jnp.argmax(token_dx)
    indexes = jnp.stack((jnp.int32(target), peak, jnp.int32(max(0, target - 1)),
                         jnp.int32(min(x.shape[0] - 1, target + 1))))
    bounded = []
    # Full captured-sequence output changes, but fixed upstream cotangents.
    # These are local interventions, not a reevaluated network loss or trajectory.
    for _, kind, cg, cv, alpha, offset in BOUNDED_CANDIDATES:
      hidden, du, dv = bounded_activation_terms(r['u'], r['v'], kind, cg, cv, alpha, offset)
      changed_y = mm(hidden, wo)
      hg = mm(dy[indexes], wo.T)
      changed_dx = mm(hg * du[indexes], wg.T) + mm(hg * dv[indexes], wu.T)
      bounded.append(dict(
          sequence_output_relative_change=l2(changed_y - r['y']) / jnp.maximum(l2(r['y']), 1e-30),
          sequence_output_rms_ratio=l2(changed_y) / jnp.maximum(l2(r['y']), 1e-30),
          selected_input_vjp_gains=jnp.linalg.norm(changed_dx, axis=-1) / jnp.maximum(token_dy[indexes], 1e-30),
          selected_output_relative_changes=jnp.linalg.norm(changed_y[indexes] - r['y'][indexes], axis=-1) / jnp.maximum(jnp.linalg.norm(r['y'][indexes], axis=-1), 1e-30),
          selected_vjp_cosines=jnp.sum(changed_dx * r['dx'][indexes], axis=-1) / jnp.maximum(jnp.linalg.norm(changed_dx, axis=-1) * token_dx[indexes], 1e-30),
          selected_gate_over_limit_fractions=jnp.mean(r['u'][indexes] > cg, axis=-1),
          selected_value_over_limit_fractions=jnp.mean(jnp.abs(r['v'][indexes]) > cv, axis=-1)))
    rows = []
    for index in indexes:
      z = {k: v[index] for k, v in r.items()}
      # J[input, output]; no finite-difference step size or perturbed network.
      j_gate = mm(wg * (z['derivative'] * z['v'])[None, :], wo)
      j_value = mm(wu * z['activation'][None, :], wo)
      jacobian = j_gate + j_value
      norm2 = jnp.sum(z['dx'] ** 2)
      projection = (z['gate_cotangent'] * mm(wg.T, z['dx'])
                    + z['value_cotangent'] * mm(wu.T, z['dx'])) / jnp.maximum(norm2, 1e-30)
      top_projection, top_units = jax.lax.top_k(projection, 16)
      geometry = {}
      for name, matrix in (('gate', wg), ('value', wu), ('output', wo.T)):
        norms = jnp.linalg.norm(matrix, axis=0)
        selected_weights = matrix[:, top_units[:4]]
        selected_norms = norms[top_units[:4]]
        geometry[name] = dict(norms=selected_norms,
            norm_percentile=jnp.mean(norms[None, :] <= selected_norms[:, None], axis=1),
            cosine=mm(selected_weights.T, selected_weights)
                / jnp.maximum(selected_norms[:, None] * selected_norms[None, :], 1e-30),
            all_norm_quantiles=jnp.quantile(norms, jnp.asarray([.5, .95, .99, 1.])))

      interventions = []
      random_units = jax.random.choice(jax.random.PRNGKey(91), wg.shape[1], (4,), replace=False)
      for name, units, cap, detach in (
          ('remove_top4_hidden_units', top_units[:4], None, False),
          ('remove_random4_hidden_units', random_units, None, False),
          ('cap_silu_at_1', None, 1., False),
          ('cap_silu_at_2', None, 2., False),
          ('detach_gate_derivative', None, None, True)):
        act, derivative = z['activation'], z['derivative']
        if units is not None:
          act, derivative = act.at[units].set(0), derivative.at[units].set(0)
        if cap is not None:
          derivative = jnp.where(jnp.abs(act) < cap, derivative, 0)
          act = jnp.clip(act, -cap, cap)
        if detach:
          derivative = jnp.zeros_like(derivative)
        hidden_g = mm(dy[index], wo.T)
        changed_dx = (mm(hidden_g * derivative * z['v'], wg.T)
                      + mm(hidden_g * act, wu.T))
        changed_y = mm(act * z['v'], wo)
        interventions.append(dict(
            input_vjp_gain=l2(changed_dx) / jnp.maximum(token_dy[index], 1e-30),
            output_relative_change=l2(changed_y - z['y']) / jnp.maximum(l2(z['y']), 1e-30)))

      def power(seed):
        q = jax.random.normal(jax.random.PRNGKey(seed), (jacobian.shape[1],))
        q = q / jnp.linalg.norm(q)
        def body(_, q):
          q = mm(jacobian.T, mm(jacobian, q))
          return q / jnp.maximum(jnp.linalg.norm(q), 1e-30)
        q = jax.lax.fori_loop(0, 80, body, q)
        return jnp.linalg.norm(mm(jacobian, q))

      rows.append(dict(token=index, input_gradient_l2=token_dx[index],
          upstream_gradient_l2=token_dy[index],
          directional_gain=token_dx[index] / jnp.maximum(token_dy[index], 1e-30),
          gain_with_target_upstream=l2(mm(jacobian, dy[target])) / jnp.maximum(token_dy[target], 1e-30),
          random_direction_rms_gain=l2(jacobian) / jnp.sqrt(jnp.float32(jacobian.shape[1])),
          spectral_gain_lower_estimates=jnp.stack([power(seed) for seed in range(3)]),
          gate_branch_gradient_l2=l2(z['dx_gate']), value_branch_gradient_l2=l2(z['dx_value']),
          gate_preactivation_absmax=jnp.max(jnp.abs(z['u'])),
          value_preactivation_absmax=jnp.max(jnp.abs(z['v'])),
          gate_activation_absmax=jnp.max(jnp.abs(z['activation'])),
          gate_derivative_absmax=jnp.max(jnp.abs(z['derivative'])),
          projected_fraction_sum=jnp.sum(projection),
          positive_projected_fraction=jnp.sum(jnp.maximum(projection, 0)),
          negative_projected_fraction=jnp.sum(jnp.minimum(projection, 0)),
          top4_weight_geometry=geometry, local_hidden_interventions=interventions,
          top4_same_unit_gate_value_cosines=jnp.sum(wg[:, top_units[:4]] * wu[:, top_units[:4]], axis=0) / jnp.maximum(jnp.linalg.norm(wg[:, top_units[:4]], axis=0) * jnp.linalg.norm(wu[:, top_units[:4]], axis=0), 1e-30),
          top16_projected_fractions=top_projection, top16_hidden_units=top_units,
          top16_gate_preactivations=z['u'][top_units],
          top16_value_preactivations=z['v'][top_units]))
    return dict(output_relative_error=l2(r['y'] - y_ref) / jnp.maximum(l2(y_ref), 1e-30),
                input_vjp_relative_error=l2(r['dx'] - dx_ref) / jnp.maximum(l2(dx_ref), 1e-30),
                global_directional_gain=l2(r['dx']) / jnp.maximum(l2(dy), 1e-30),
                bounded_candidate_token_indexes=indexes, bounded_candidates=bounded,
                selected_tokens=rows)

  compiled = jax.jit(summary)
  results = []
  for quantized in (True, False):
    w = [a.astype(jnp.bfloat16).astype(jnp.float32) if quantized else a.astype(jnp.float32)
         for a in weights]
    result = jax.device_get(compiled(x, dy, actual_dx, actual_y, *w))
    result = jax.tree.map(lambda a: np.asarray(a).tolist(), result)
    for entry, spec in zip(result['bounded_candidates'], BOUNDED_CANDIDATES):
      entry['name'] = spec[0]
    for row in result['selected_tokens']:
      for entry, name in zip(row['local_hidden_interventions'],
          ('remove_top4_hidden_units', 'remove_random4_hidden_units', 'cap_silu_at_1',
           'cap_silu_at_2', 'detach_gate_derivative')):
        entry['name'] = name
    results.append(dict(effective_bf16_weights=quantized, **result))
  return dict(scope='Fixed captured MLP inputs/upstream cotangents; FP32 chain rule, no network reevaluation',
              arms=results)

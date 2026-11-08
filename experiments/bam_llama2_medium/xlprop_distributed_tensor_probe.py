"""Formal-mesh frozen-state cotangents; called by bounded_aot_replay.py.

No model edits, optimizer updates, callbacks or checkpoint writes. Zero additive
probes are limited to one global sequence. Full parameter differentiation and
formal input/parameter shardings are retained. Recompilation can change BF16
lowering: forward loss AND parameter-gradient controls gate interpretation.
"""
import json
import time


def run_probe(train, captured, reference, output, sequence, target_layer):
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from layers import attentions, fusion

  config, mesh = captured['config'], captured['mesh']
  model, _, state_shardings = captured['functional_train'].args
  assert jax.device_count() == 16 and jax.process_count() == 4
  assert config.global_batch_size_to_train_on == 128
  assert config.remat_policy == 'full'
  assert config.scan_layers and not config.mix_attn
  state = captured['state']
  data = dict(captured['data'])
  # The selected target interval was already probed with the original AOT;
  # preserve that mask rather than inferring it from the full captured batch.
  token = captured['probe_token']
  @jax.jit
  def selected_segments(segments):
    rows = jnp.arange(segments.shape[0])[:, None]
    mask = (rows == sequence) & (jnp.arange(config.max_target_length)[None, :] == token)
    return jnp.where(mask, segments, 0)
  data['targets_segmentation'] = selected_segments(data['targets_segmentation'])
  layer_stack, attention_stack = [], []
  active_probes = None
  original_splash = attentions._bam_splash_attention

  def stable_rms(x):
    x = x.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(x))
    return maximum * jnp.sqrt(jnp.mean((x / jnp.maximum(maximum, 1e-30)) ** 2))

  def inject(x, addition):
    selected = (jnp.arange(x.shape[0]) == sequence).reshape((-1,) + (1,) * (x.ndim - 1))
    return x + jnp.where(selected, addition, jnp.zeros((), x.dtype))

  def captured_splash(q, k, v, segments, config, mesh, window=None, seq_minor=False):
    selected = jnp.asarray(layer_stack[-1]) == target_layer
    q, k, v = (inject(x, jnp.where(selected, active_probes[name], 0))
               for x, name in zip((q, k, v), ('q', 'k', 'v')))
    for name, x in zip(('q', 'k', 'v'), (q, k, v)):
      attention_stack[-1].sow('intermediates', 'formal_probe_' + name, x[sequence:sequence + 1])
    result = original_splash(q, k, v, segments, config, mesh, window=window, seq_minor=seq_minor)
    attention_stack[-1].sow('intermediates', 'formal_probe_context', result[sequence:sequence + 1])
    return inject(result, jnp.where(selected, active_probes['context'], 0))

  def intercept(next_fun, call_args, kwargs, context):
    if isinstance(context.module, attentions.BamAttention) and context.method_name == '__call__':
      attention_stack.append(context.module)
      try:
        return next_fun(*call_args, **kwargs)
      finally:
        attention_stack.pop()
    if isinstance(context.module, fusion.SubDecoderLayer) and context.method_name == '__call__':
      layer = kwargs.get('layer_index', context.module.layer_inx)
      layer = context.module.layer_inx if layer is None else layer
      local_args, local_kwargs = list(call_args), dict(kwargs)
      local_args[0] = inject(local_args[0], active_probes['residual'][layer])
      if local_kwargs.get('M_in') is not None:
        local_kwargs['M_in'] = inject(local_kwargs['M_in'], active_probes['M'][layer])
        value = local_kwargs['M_in'][sequence:sequence + 1]
        context.module.sow('intermediates', 'formal_probe_M_rms', stable_rms(value))
      value = local_args[0][sequence:sequence + 1]
      context.module.sow('intermediates', 'formal_probe_residual_rms', stable_rms(value))
      layer_stack.append(layer)
      try:
        return next_fun(*local_args, **local_kwargs)
      finally:
        layer_stack.pop()
    return next_fun(*call_args, **kwargs)

  def objective(params, probes, batch, rng):
    nonlocal active_probes
    active_probes = probes
    try:
      with nn.intercept_methods(intercept):
        loss, aux = train.loss_fn(model, config, dict(batch), rng, params, is_train=True)
    finally:
      active_probes = None
    decoder = aux['intermediate_outputs']['intermediates']['decoder']
    body = decoder['layers']['block']
    values = {name: body['self_attention']['formal_probe_' + name][0][target_layer]
              for name in ('q', 'k', 'v', 'context')}
    for name in ('M', 'residual'):
      key = 'formal_probe_' + name + '_rms'
      values[name + '_rms'] = jnp.concatenate((body[key][0],
          jnp.reshape(decoder['final_local_layer']['block'][key][0], (1,))))
    return loss, values

  def measure(params, probes, batch, rng):
    (loss, values), (parameter_grads, probe_grads) = jax.value_and_grad(
        objective, argnums=(0, 1), has_aux=True)(params, probes, batch, rng)
    norms = {name: jax.vmap(stable_rms)(probe_grads[name]) for name in ('M', 'residual')}
    peaks = {}
    for name, axes in [('M', (-2, -1)), ('residual', (-1,))]:
      token_rms = jnp.sqrt(jnp.mean(probe_grads[name].astype(jnp.float32) ** 2, axis=axes))
      token_rms = token_rms.reshape((config.num_decoder_layers, -1))
      peaks[name] = dict(peak_rms=token_rms.max(axis=1), peak_token=token_rms.argmax(axis=1))
    summary = dict(loss=loss, raw_grad_norm=train.max_utils.l2norm_pytree(parameter_grads),
                   norms=norms, peaks=peaks)
    bias_grad = parameter_grads['params']['decoder']['embedding_bam_write']['W_emb_v_up']['bias']
    summary['embedding_address_bias_l2'] = jnp.linalg.norm(bias_grad.astype(jnp.float32))
    return summary, values, {name: probe_grads[name] for name in ('q', 'k', 'v', 'context')}

  l, t = config.num_decoder_layers, config.max_target_length
  probes = dict(M=np.zeros((l, 1, t, config.bam_v, config.bam_k), config.dtype),
                residual=np.zeros((l, 1, t, config.emb_dim), config.dtype))
  for name, width in [('q', config.bam_partial_rope_nope_dim + config.bam_standard_qk_dim),
                      ('k', config.bam_partial_rope_nope_dim + config.bam_standard_qk_dim),
                      ('v', config.head_dim), ('context', config.head_dim)]:
    probes[name] = np.zeros((1, t, config.num_query_heads, width), config.dtype)
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*config.data_sharding))
  probe_shardings = jax.tree.map(lambda _: replicated, probes)
  probes = jax.tree.map(lambda x: jax.make_array_from_callback(x.shape, replicated, lambda index: x[index]), probes)
  started = time.monotonic()
  attentions._bam_splash_attention = captured_splash
  try:
    with mesh, nn.partitioning.axis_rules(config.logical_axis_rules):
      compiled = jax.jit(measure, in_shardings=(state_shardings.params, probe_shardings, data_sharding, None),
                         out_shardings=replicated)
      summary, values, gradients = jax.device_get(compiled(state.params, probes, data, captured['rng']))
  finally:
    attentions._bam_splash_attention = original_splash
  observed_ce, observed_norm = float(summary['loss']), float(summary['raw_grad_norm'])
  expected_ce, expected_norm = reference['learning/loss'], reference['learning/raw_grad_norm']
  fidelity = dict(selected_ce=observed_ce, reference_ce=expected_ce,
      raw_norm=observed_norm, reference_raw_norm=expected_norm,
      forward_abs_error=abs(observed_ce - expected_ce),
      gradient_relative_error=abs(observed_norm / expected_norm - 1),
      seconds=time.monotonic() - started, global_sequence=sequence, target_token=token,
      target_layer=target_layer, devices=jax.device_count(), processes=jax.process_count(),
      embedding_address_bias_l2=float(summary['embedding_address_bias_l2']))
  fidelity['passed'] = fidelity['forward_abs_error'] < 1e-3 and fidelity['gradient_relative_error'] < .05
  if jax.process_index() == 0:
    (output / 'tensor-fidelity.json').write_text(json.dumps(fidelity, indent=2) + '\n')
    rows = [dict(layer=i, **{name + '_forward_rms': float(values[name + '_rms'][i]) for name in ('M', 'residual')},
                 **{name + '_cotangent_rms': float(summary['norms'][name][i]) for name in ('M', 'residual')},
                 **{name + '_' + key: float(value[i]) for name, fields in summary['peaks'].items()
                    for key, value in fields.items()}) for i in range(l)]
    (output / 'layer-cotangents.json').write_text(json.dumps(rows, indent=2) + '\n')
    # Retain conditional tensors even when fidelity fails, with the failed gate
    # beside them; never silently treat another operating point as reproduction.
    for name in ('q', 'k', 'v', 'context'):
      np.save(output / ('forward-' + name + '.npy'), np.asarray(values[name], np.float32))
      np.save(output / ('backward-' + name + '.npy'), np.asarray(gradients[name], np.float32))
    print('FORMAL_TENSOR_PROBE_COMPLETE', json.dumps(fidelity), flush=True)

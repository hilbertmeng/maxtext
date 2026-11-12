"""Frozen full-batch sink interventions following the 37-case layer ranking.

Model files stay at d54c4c5. Original-AOT parameter splits run first; a separate
instrumented full-remat objective must pass forward AND gradient fidelity gates.
Value clamps keep same-layer Q/K unchanged, but later routing can respond.
"""
from pathlib import Path
import runpy


def parameter_plans():
  v, o, w = 'attention_local_v', 'attention_local_o', 'attention_output_projection'
  p = [dict(name='peak_control', families=[], direction='revert'),
       dict(name='old_control', families=['ALL'], direction='revert')]
  for layer in (3, 0, 12):
    for label, families in [('V', [v]), ('O', [o]), ('WO', [w]),
                             ('QK', ['attention_standard_qk', 'attention_local_qk']),
                             ('write', ['attention_write']),
                             ('V_O_WO', [v, o, w]), ('all', ['ATTENTION'])]:
      p.append(dict(name=f'revert_L{layer:02d}_{label}', families=families,
                    direction='revert', layer=layer))
  for label, families in [('QK', ['attention_standard_qk', 'attention_local_qk']),
                           ('V_WO', [v,w]), ('all', ['ATTENTION'])]:
    p.append(dict(name=f'revert_L00_05_{label}', families=families,
                  direction='revert', layer=list(range(6))))
  p.append(dict(name='revert_all_QK', families=['attention_standard_qk', 'attention_local_qk'], direction='revert'))
  return p


def patch_value(v, donor, layer, target, mode, position):
  """Negative target encodes a layer bitmask; modes 4/5 zero/freeze 8 tokens."""
  import jax
  import jax.numpy as jnp
  chosen = (layer == target) | ((target < 0) & jnp.asarray(((-target) >> layer) & 1, dtype=bool))
  indices = jnp.arange(v.shape[1])
  width = jnp.where(mode >= 4, 8, 1)
  selected = (indices >= position) & (indices < position + width)
  mask = chosen & (mode != 0) & selected[None, :, None, None]
  replacement = jnp.where((mode == 1) | (mode == 4), jnp.zeros_like(v),
                jnp.where(mode == 2, donor[:, None], jax.lax.stop_gradient(v)))
  return jnp.where(mask, replacement, v)


def run_probe(train, captured, reference, output, sequence, target_layer):
  import hashlib
  import json
  import os
  import subprocess
  import time
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn, serialization
  from layers import attentions, fusion

  helper = runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
  helper['run_probe'].__globals__['case_plans'] = parameter_plans
  helper['run_probe'](train, captured, reference, output, sequence, target_layer)
  coarse = json.loads((output/'loss-localization.json').read_text())
  l3 = next(x for x in coarse['cases'] if x['name'] == 'revert_L03_all')['metrics']
  assert abs(l3['learning/loss'] - 2.1567070484161377) < 1e-5
  assert abs(l3['learning/raw_grad_norm']/8204.7158203125 - 1) < 1e-5
  upload = helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  cfg, mesh, state = captured['config'], captured['mesh'], captured['state']
  model, _, state_shardings = captured['functional_train'].args
  assert cfg.scan_layers and cfg.param_scan_axis == 1 and cfg.remat_policy == 'full'
  assert jax.device_count() == 16 and jax.process_count() == 4
  assert cfg.global_batch_size_to_train_on == 128
  source = os.environ['BAM_DIAG_REFERENCE_CHECKPOINT']
  abstract = {'params': jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state.params)}
  cp = ocp.PyTreeCheckpointer()
  restored = cp.restore(source, item=abstract, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args(abstract))
  cp.close()
  old = serialization.from_state_dict(state.params, serialization.to_state_dict(restored['params']))
  del abstract, restored
  stack, modules = [], []
  active = None
  original_splash = attentions._bam_splash_attention
  query_positions = (16, 128, 512, 1220, 2048, 4095)

  def splash(q, k, v, segments, config, mesh, window=None, seq_minor=False):
    layer = jnp.asarray(stack[-1])
    target, mode, position, donor = active
    # Capture native first-token V before the clamp. Only small per-position
    # arrays leave the compiled objective; no full attention map is materialized.
    modules[-1].sow('intermediates', 'sink_v0', v[:, 0])
    v = patch_value(v, donor, layer, target, mode, position)
    def statistics(_):
      qs = q[:, jnp.asarray(query_positions)].astype(jnp.float32)
      logits = jnp.einsum('bqhd,bshd->bhqs', qs, k.astype(jnp.float32))
      qp = jnp.asarray(query_positions)
      valid = jnp.arange(k.shape[1])[None, :] < qp[:, None]
      valid = valid[None, None] & (segments[:, None, jnp.asarray(query_positions), None] == segments[:, None, None, :])
      valid = valid & (segments[:, None, jnp.asarray(query_positions), None] != 0)
      logits = jnp.where(valid, logits, -1e30)
      weights = jax.nn.softmax(logits, axis=-1)
      weights = jnp.where(valid, weights, 0.)
      # [metric, query, head]; first-source mass, entropy and routed V0 RMS.
      mass = weights[..., 0]
      entropy = -jnp.sum(jnp.where(weights > 0, weights * jnp.log(jnp.maximum(weights, 1e-30)), 0.), axis=-1)
      v0rms = jnp.sqrt(jnp.mean(v[:, 0].astype(jnp.float32)**2, axis=-1))
      contribution = mass * v0rms[:, :, None]
      return jnp.stack((mass.mean(0).T, entropy.mean(0).T, contribution.mean(0).T))
    stats = jax.lax.cond(layer == target, statistics,
                         lambda _: jnp.zeros((3, len(query_positions), cfg.num_query_heads), jnp.float32), None)
    modules[-1].sow('intermediates', 'sink_stats', stats)
    return original_splash(q, k, v, segments, config, mesh, window=window, seq_minor=seq_minor)

  def intercept(next_fun, args, kwargs, context):
    if isinstance(context.module, fusion.SubDecoderLayer) and context.method_name == '__call__':
      layer = kwargs.get('layer_index', context.module.layer_inx)
      stack.append(context.module.layer_inx if layer is None else layer)
      try: return next_fun(*args, **kwargs)
      finally: stack.pop()
    if isinstance(context.module, attentions.BamAttention) and context.method_name == '__call__':
      modules.append(context.module)
      try: return next_fun(*args, **kwargs)
      finally: modules.pop()
    return next_fun(*args, **kwargs)

  def objective(params, batch, rng, target, mode, position, donor):
    nonlocal active
    active = target, mode, position, donor
    try:
      with nn.intercept_methods(intercept):
        loss, aux = train.loss_fn(model, cfg, dict(batch), rng, params, is_train=True)
    finally: active = None
    body = aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']['self_attention']
    v0 = body['sink_v0'][0]
    stats = body['sink_stats'][0].sum(axis=0)
    return loss, (v0, stats)

  def measure(params, batch, rng, target, mode, position, donor):
    (loss, (v0, stats)), grads = jax.value_and_grad(objective, has_aux=True)(params, batch, rng, target, mode, position, donor)
    return dict(loss=loss, raw_grad_norm=train.max_utils.l2norm_pytree(grads), v0=v0, stats=stats)

  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*cfg.data_sharding))
  # V0 batch axis follows the formal batch mesh; output is replicated and small.
  donor_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*cfg.data_sharding[:1]))
  donor_np = np.zeros((128, cfg.num_query_heads, cfg.head_dim), cfg.dtype)
  donor = jax.make_array_from_callback(donor_np.shape, donor_sharding, lambda index: donor_np[index])
  compiled = jax.jit(measure, in_shardings=(state_shardings.params, data_sharding, None, None, None, None, donor_sharding), out_shardings=replicated)
  result = dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',
      old_checkpoint=source, peak_checkpoint_step=35675, batch_step=35676,
      full_batch=True, query_positions=query_positions, layers=[3, 0, 12],
      scope='Negative layer means bitmask joint intervention; modes4/5 zero/freeze eight source positions. Same-layer Q/K unchanged, later network responds. Group case first_value_rms_by_head reports native preclamp L3; group sampled stats are zero placeholders. QK parameter rollback changes routing at fixed single-layer input; joint-layer input may respond. Conditional effects not additive.',
      attention_statistics='FP32 reconstruction on six sampled queries; exclude diagonal/other segments/empty rows; not exact Splash probabilities',
      input_shards=[dict(index=str(s.index), sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest()) for s in captured['data']['inputs'].addressable_shards],
      cases=[])
  def publish():
    p = output/'sink-localization.json'; p.write_text(json.dumps(result, indent=2)+'\n')
    if jax.process_index() == 0:
      subprocess.run(['gsutil', '-q', 'cp', str(p), upload+'/sink-localization.json'], check=True, timeout=90)
  def evaluate(label, params, layer, mode=0, position=0, donated=None, expected=None):
    start = time.monotonic()
    d = donor if donated is None else jax.device_put(donated, donor_sharding)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      y = jax.device_get(compiled(params, captured['data'], captured['rng'], jnp.int32(layer), jnp.int32(mode), jnp.int32(position), d))
    case = dict(name=label, layer=layer, mode=mode, position=position, seconds=time.monotonic()-start,
                loss=float(y['loss']), raw_grad_norm=float(y['raw_grad_norm']),
                first_value_rms_by_head=np.sqrt(np.mean(y['v0'][layer if layer >= 0 else 3].astype(np.float32)**2, axis=(0, 2))).tolist(),
                sampled_routing_stats=np.asarray(y['stats']).tolist())
    assert np.isfinite(case['loss']) and np.isfinite(case['raw_grad_norm']), case
    if expected:
      case['fidelity'] = dict(loss_abs_error=abs(case['loss']-expected[0]), raw_relative_error=abs(case['raw_grad_norm']/expected[1]-1))
      case['fidelity']['passed'] = case['fidelity']['loss_abs_error'] < 1e-3 and case['fidelity']['raw_relative_error'] < .05
    result['cases'].append(case); publish()
    if jax.process_index() == 0: print('SINK_CASE', label, case['loss'], case['raw_grad_norm'], flush=True)
    if expected: assert case['fidelity']['passed'], ('Recompiled control failed: do not interpret interventions', case)
    return y['v0']
  attentions._bam_splash_attention = splash
  try:
    evaluate('peak_control', state.params, 3, expected=(2.595792531967163,809414.5625))
    old_v = evaluate('old_control', old, 3, expected=(1.9770115613937378,.21527354419231415))
    for layer in (0,12):
      evaluate(f'L{layer:02d}_old_native', old, layer)
    classify = runpy.run_path(str(Path(__file__).with_name('xlprop_parameter_family_probe.py')))['parameter_family']
    def transplant(path, before, after):
      names = tuple(getattr(k, 'key', None) for k in path)
      family = lambda _: classify(path, fine_attention=True, split_vo=True)
      yes, axis = helper['selected_parameter'](names, after.shape, family, {'attention_local_v'}, 3)
      if not yes: return after
      if axis is None: return before
      index = [slice(None)]*after.ndim; index[axis] = 3; index = tuple(index)
      return after.at[index].set(before[index])
    local_old = jax.tree.map_with_path(transplant, old, state.params)
    local_v = evaluate('L03_old_localV_parameters_same_upstream', local_old, 3)
    evaluate('L03_old_localV_first_V_same_upstream', state.params, 3, 2, donated=local_v[3])
    del local_old, local_v
    for layer in (3,0,12):
      # Native per-layer stats, zero/frozen/old first V, and ordinary source control.
      evaluate(f'L{layer:02d}_native', state.params, layer)
      evaluate(f'L{layer:02d}_freeze_first_V', state.params, layer, 3)
      evaluate(f'L{layer:02d}_zero_first_V', state.params, layer, 1)
      evaluate(f'L{layer:02d}_old_first_V', state.params, layer, 2, donated=old_v[layer])
      evaluate(f'L{layer:02d}_zero_position128_V', state.params, layer, 1, position=128)
    for label, mask in [('L00_05',63), ('L00_27',(1<<28)-1)]:
      evaluate(f'{label}_zero_first_V', state.params, -mask, 1)
      evaluate(f'{label}_freeze_first_V', state.params, -mask, 3)
      evaluate(f'{label}_zero_position128_V', state.params, -mask, 1, position=128)
      evaluate(f'{label}_zero_prefix8_V', state.params, -mask, 4)
      evaluate(f'{label}_freeze_prefix8_V', state.params, -mask, 5)
      evaluate(f'{label}_zero_interior8_V', state.params, -mask, 4, position=128)
    result['complete'] = True; publish()
  finally: attentions._bam_splash_attention = original_splash

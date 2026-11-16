"""Original-AOT parameter transplants on one frozen full-topology batch.

Called through bounded_aot_replay.py's probe interface. Compare ordinary-MLP
weights and the remaining parameter family independently between checkpoints;
optimizer state, RNG and batch remain the captured peak's. Returned updates are
discarded. This does not measure the quality of a retrained intervention.
"""


def ordinary_mlp(path):
  names = tuple(getattr(key, 'key', None) for key in path)
  return 'mlp' in names and any(name in names for name in ('wi_0', 'wi_1', 'wo'))


def parameter_family(path, fine_attention=False, split_vo=False):
  names = tuple(getattr(key, 'key', None) for key in path)
  if ordinary_mlp(path):
    return 'ordinary_mlp'
  if 'self_attention' in names:
    part = names[names.index('self_attention') + 1]
    if part in ('common_qk_read_scale', 'common_vo_read_scale'):
      return 'read_scales'
    if not fine_attention:
      return 'attention'
    if part in ('query', 'key'):
      return 'attention_standard_qk'
    if part == 'full_m_read_down':
      return 'attention_shared_read_down'
    if part.startswith(('W_lq_', 'W_lk_', 'general_q_', 'general_k_')) or part in ('static_q_key', 'static_k_key'):
      return 'attention_local_qk'
    if part.startswith(('W_lv_', 'W_R_', 'general_v_', 'general_vo_', 'general_o_')) or part in ('static_v_key', 'static_o_key'):
      if split_vo:
        is_v = part.startswith(('W_lv_', 'general_v_')) or part == 'static_v_key'
        return 'attention_local_v' if is_v else 'attention_local_o'
      return 'attention_local_vo'
    if part.startswith(('P_loc_', 'W_gw')) or part == 'gw_b0':
      return 'attention_write'
    if part.startswith('M_shift_'):
      return 'attention_mshift'
    if part == 'out':
      return 'attention_output_projection'
    raise AssertionError(('Unclassified attention parameter', names))
  if 'embedding_bam_write' in names:
    return 'embedding_write'
  if 'token_embedder' in names:
    return 'embedding'
  if 'memory_mlp' in names or any(isinstance(n, str) and n.startswith(('mlp_address_', 'mlp_write_gate')) for n in names):
    return 'memory_writer'
  if any(isinstance(n, str) and ('layer_norm' in n or n == 'decoder_norm') for n in names):
    return 'layer_norms'
  if 'lm_head' in names and 'logits_dense' in names:
    return 'unembedding'
  return 'other'


def run_probe(train, captured, reference, output, sequence, target_layer):
  import json
  import os
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import serialization

  checkpoint = os.environ['BAM_DIAG_REFERENCE_CHECKPOINT']
  assert '/diagnostics/' in checkpoint and checkpoint.endswith('/items')
  state = captured['state']
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state.params)
  target = {'params': abstract}
  cp = ocp.PyTreeCheckpointer()
  restored = cp.restore(checkpoint, item=target, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args(target))
  cp.close()
  old = serialization.from_state_dict(state.params, serialization.to_state_dict(restored['params']))
  del restored, abstract, target
  assert jax.tree.structure(old) == jax.tree.structure(state.params)
  copy = lambda tree: jax.tree.map(lambda x: x.copy(), tree)
  flags = jax.tree.map_with_path(lambda path, _: ordinary_mlp(path), state.params)
  fine_attention = os.environ.get('BAM_DIAG_FINE_ATTENTION') == '1'
  split_vo = os.environ.get('BAM_DIAG_SPLIT_VO') == '1'
  assert not split_vo or fine_attention
  families = jax.tree.map_with_path(lambda path, _: parameter_family(path, fine_attention, split_vo), state.params)
  family_names = sorted(set(jax.tree.leaves(families)))
  leaves = jax.tree.leaves(flags)
  assert any(leaves) and not all(leaves)

  @jax.jit
  def distances(old, new):
    result = []
    for is_mlp in (True, False):
      a = jax.tree.leaves(old)
      b = jax.tree.leaves(new)
      base = sum(jnp.sum(x.astype(jnp.float32) ** 2) for x, flag in zip(a, leaves) if flag == is_mlp)
      delta = sum(jnp.sum((x.astype(jnp.float32) - y.astype(jnp.float32)) ** 2)
                  for x, y, flag in zip(a, b, leaves) if flag == is_mlp)
      result.append(dict(parameter_l2=jnp.sqrt(base), delta_l2=jnp.sqrt(delta), relative_delta=jnp.sqrt(delta / base)))
    return result

  parameter_distances = jax.tree.map(float, jax.device_get(distances(old, state.params)))
  original_batch = dict(captured['data'])
  @jax.jit
  def mask(segments):
    selected = ((jnp.arange(segments.shape[0])[:, None] == sequence)
                & (jnp.arange(segments.shape[1])[None, :] == captured['probe_token']))
    return jnp.where(selected, segments, 0)
  selected_batch = dict(original_batch, targets_segmentation=mask(original_batch['targets_segmentation']))
  results = dict(reference_checkpoint=checkpoint, model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',
      scope='Original AOT; replace parameter families only, preserve peak optimizer/RNG/data; discard updates',
      parameter_distances=dict(ordinary_mlp=parameter_distances[0], rest=parameter_distances[1]), cases=[])
  results['parameter_families'] = {name: sum(x.size for x, f in zip(jax.tree.leaves(state.params),
      jax.tree.leaves(families)) if f == name) for name in family_names}
  new = state.params
  plans = []
  for mlp_new, rest_new in ((True, True), (False, False), (True, False), (False, True)):
    plans.append((dict(ordinary_mlp_from_peak=mlp_new, rest_from_peak=rest_new),
                  lambda f, m=mlp_new, r=rest_new: m if f == 'ordinary_mlp' else r))
  if os.environ.get('BAM_DIAG_EXPAND_PARAMETER_FAMILIES') == '1':
    for family in family_names:
      if family == 'ordinary_mlp':
        continue
      if fine_attention and not (family.startswith('attention_') or family == 'read_scales'):
        continue
      if split_vo and family not in ('attention_local_v', 'attention_local_o'):
        continue
      for only in (True, False):
        plans.append((dict(family=family, intervention='only_family_from_peak' if only else 'revert_family_to_old'),
                      lambda f, name=family, only=only: (f == name) if only else (f != name)))
  for case, choose_new in plans:
    params = jax.tree.map(lambda a, b, family: (b if choose_new(family) else a).copy(), old, new, families)
    for label, batch in (('full_batch', original_batch), ('isolated_target', selected_batch)):
      updated, metrics = captured['compiled'](copy(state).replace(params=copy(params)),
                                               batch, captured['rng'])
      selected = {key: value for key, value in metrics['scalar'].items()
                  if key in ('learning/loss', 'learning/raw_grad_norm', 'learning/grad_norm', 'learning/total_weights')}
      values = {key: float(np.asarray(value)) for key, value in jax.device_get(selected).items()}
      del updated
      case[label] = values
      if case.get('ordinary_mlp_from_peak') is True and case.get('rest_from_peak') is True:
        expected = captured['observed'] if label == 'full_batch' else reference
        for key in ('learning/loss', 'learning/raw_grad_norm'):
          assert abs(values[key] - expected[key]) <= 1e-5 * max(1., abs(expected[key])), (label, key, values[key], expected[key])
    results['cases'].append(case)
    if jax.process_index() == 0:
      (output / 'parameter-family-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
      print('PARAMETER_FAMILY_CHECK', {k: v for k, v in case.items() if k not in ('full_batch', 'isolated_target')},
            case['full_batch']['learning/raw_grad_norm'], flush=True)

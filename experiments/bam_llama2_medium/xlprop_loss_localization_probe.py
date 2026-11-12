"""Bounded original-AOT loss localization between pre35636 and pre35676.

Invoked by bounded_aot_replay.py. Returned optimizer updates are discarded;
checkpoint sources and the immutable model executable are never modified.
"""
from pathlib import Path
import re
import runpy


def upload_destination(upload):
  """Allow only this study's results directory, including numbered retries."""
  root = 'gs://newproject-1-llm_base_models_us-central1/log/diagnostics/'
  destination = upload.rstrip('/')
  if not re.fullmatch(re.escape(root) + r'xlprop-v5-lossloc-20261112(?:-r[1-9][0-9]*)?/results', destination):
    raise ValueError(f'Unexpected loss-localization upload destination: {upload}')
  return destination


def selected_parameter(names, shape, families, family_set, layer, scan_axis=1):
  """Return (selected, sliced_axis); final layer27 is not a scanned slice."""
  family = families(names)
  if family not in family_set:
    return False, None
  if layer is None:
    return True, None
  if 'final_local_layer' in names:
    return layer == 27, None
  if 'layers' in names:
    assert shape[scan_axis] == 27, (names, shape, scan_axis)
    return layer < 27, scan_axis
  raise AssertionError(('Unrecognized attention layer layout', names, shape))


def case_plans():
  v, o, w = 'attention_local_v', 'attention_local_o', 'attention_output_projection'
  plans = [dict(name='peak_control', families=[], direction='revert'),
           dict(name='old_control', families=['ALL'], direction='revert')]
  for label, group in [('V_WO', [v, w]), ('O_WO', [o, w]), ('V_O_WO', [v, o, w])]:
    for direction in ('revert', 'introduce'):
      plans.append(dict(name=f'{direction}_{label}', families=group, direction=direction))
  # Reproduce the existing coarse rollback as a gate for the layer ranking.
  plans.append(dict(name='revert_all_attention', families=['ATTENTION'], direction='revert'))
  for layer in range(28):
    plans.append(dict(name=f'revert_attention_L{layer:02d}', families=['ATTENTION'],
                      direction='revert', layer=layer))
  return plans


def run_probe(train, captured, reference, output, sequence, target_layer):
  import hashlib
  import json
  import os
  import subprocess
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import serialization

  classify = runpy.run_path(str(Path(__file__).with_name('xlprop_parameter_family_probe.py')))['parameter_family']
  def family(names):
    class Key:
      def __init__(self, key): self.key = key
    return classify(tuple(Key(n) for n in names), fine_attention=True, split_vo=True)

  source = os.environ['BAM_DIAG_REFERENCE_CHECKPOINT']
  assert '/diagnostics/' in source and source.endswith('/35635/items'), source
  cfg, state = captured['config'], captured['state']
  assert cfg.num_decoder_layers == 28 and cfg.param_scan_axis == 1
  target = {'params': jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state.params)}
  cp = ocp.PyTreeCheckpointer()
  restored = cp.restore(source, item=target, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args(target))
  cp.close()
  old = serialization.from_state_dict(state.params, serialization.to_state_dict(restored['params']))
  del target, restored
  assert jax.tree.structure(old) == jax.tree.structure(state.params)
  paths, leaves = zip(*jax.tree_util.tree_flatten_with_path(state.params)[0])
  names = [tuple(getattr(k, 'key', None) for k in path) for path in paths]
  labels = [family(n) for n in names]
  attention = {f for f in labels if f.startswith('attention_')}
  all_families = set(labels)
  copy = lambda tree: jax.tree.map(lambda x: x.copy(), tree)
  expected = {'peak_control': (2.595792531967163, 809414.5625),
              'old_control': (1.9770115613937378, .21527354419231415),
              'revert_all_attention': (2.0027377605438232, 1642.820068359375)}
  points = []
  result = dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',
      old_checkpoint=source, peak_checkpoint_step=35675, batch_step=35676,
      full_batch=True, layer_numbering='zero-based, scanned0..26 + final27',
      semantics='Frozen parameter interventions; discard updates; conditional effects are not additive',
      local_input_shards=[dict(index=str(s.index), sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest())
                          for s in captured['data']['inputs'].addressable_shards], cases=points)
  upload = os.environ.get('BAM_LOSS_LOCALIZATION_UPLOAD')
  upload = upload_destination(upload) if upload else None
  def publish():
    dest = output / 'loss-localization.json'
    dest.write_text(json.dumps(result, indent=2) + '\n')
    if upload and jax.process_index() == 0:
      subprocess.run(['gsutil', '-q', 'cp', str(dest), upload.rstrip('/') + '/loss-localization.json'],
                     check=True, timeout=90)
  for plan in case_plans():
    wanted = set(plan['families'])
    wanted = all_families if 'ALL' in wanted else attention if 'ATTENTION' in wanted else wanted
    selections = [selected_parameter(n, x.shape, family, wanted, plan.get('layer')) for n, x in zip(names, leaves)]
    assert plan['name'] == 'peak_control' or any(t[0] for t in selections), plan
    a, b = jax.tree.leaves(old), jax.tree.leaves(state.params)
    params, changed = [], 0
    for before, after, (yes, axis) in zip(a, b, selections):
      base, donor = (after, before) if plan['direction'] == 'revert' else (before, after)
      if yes and axis is not None:
        index = [slice(None)] * base.ndim
        index[axis] = plan['layer']
        index = tuple(index)
        value = base.at[index].set(donor[index])
        changed += base.size // base.shape[axis]
      else:
        value = donor if yes else base
        if yes: changed += base.size
      params.append(value.copy())
    params = jax.tree.unflatten(jax.tree.structure(state.params), params)
    updated, metrics = captured['compiled'](copy(state).replace(params=params), captured['data'], captured['rng'])
    values = {key: float(np.asarray(value)) for key, value in jax.device_get(metrics['scalar']).items()
              if key in ('learning/loss', 'learning/raw_grad_norm', 'learning/grad_norm', 'learning/total_weights')}
    del updated, params
    assert all(np.isfinite(v) for v in values.values()), values
    if plan['name'] in expected:
      loss, raw = expected[plan['name']]
      for k, v in [('learning/loss', loss), ('learning/raw_grad_norm', raw)]:
        assert abs(values[k] - v) <= 1e-5 * max(1., abs(v)), (plan, k, values[k], v)
    points.append(dict(**plan, selected_elements=changed, metrics=values))
    if jax.process_index() == 0:
      print('LOSS_LOCALIZATION_CASE', plan['name'], values, flush=True)
    publish()

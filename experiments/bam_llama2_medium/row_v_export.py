"""Trace the downstream consequences of denying original row input to cross-V.

All valid origins participate. Donors differ ONLY by an early V-cross input
denial, rather than by deleting the original row throughout the network.
References stay on device; only token losses and exact-null audits are saved.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

import row_consumer_positions as c
from analyze_row_mediation import stats

base, med = c.base, c.med


def recipient_arms(source, early_end):
  result = []
  groups = [(f'L{l}', [l]) for l in range(early_end + 1, min(early_end + 7, 24))]
  groups += [(f'L{early_end+1}-23', range(early_end+1, 24))]
  # Same-layer M is written after MHA V read; same-layer fetched read is before
  # its write and cannot consume that newly produced M.
  groups += [(f'L{early_end}', [early_end])]
  for label, layers in groups:
    for field in ('full_col', 'full_row', 'M', 'mlp'):
      if label == f'L{early_end}' and field in ('full_col', 'full_row'):
        continue
      control = np.zeros((24, len(med.CONTROL_NAMES)), np.float32)
      control[list(layers), med.CONTROL_NAMES.index(field)] = 1
      for world in ('block', 'rescue'):
        result.append(dict(name=f'{world}_{label}_{field}', world=world,
                           control=control))
  return result


def run(config):
  if os.environ.get('BAM_CONSUMER_BARRIER') != '1':
    raise ValueError('requires the validated post-attention boundary')
  source = int(os.environ.get('BAM_MEDIATION_SOURCE', '11'))
  early_end = int(os.environ.get('BAM_V_EXPORT_END', str(source+1)))
  if not source < early_end < 23:
    raise ValueError('early cross-V recipients must follow the source')
  component = os.environ.get('BAM_MEDIATION_COMPONENT', 'both')
  columns = {'cross': [0, 1], 'self': [2], 'both': [0, 1, 2]}[component]
  output = Path(os.environ['BAM_MEDIATION_OUTPUT']); output.mkdir(parents=True, exist_ok=True)
  path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(path) as data:
    cohort = {k: np.asarray(data[k]) for k in (
        'inputs', 'targets', 'inputs_position', 'inputs_segmentation',
        'targets_segmentation', 'sequence_hashes')}
  bs = int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  matrix = recipient_arms(source, early_end)
  names = ['clean', 'row_deleted', 'early_cross_v_denied'] + [a['name'] for a in matrix]
  start = time.perf_counter()
  rng, writer, manager, mesh, model, _, tx = base.train.setup_mesh_and_model(config)
  iterator, _ = base.create_data_iterator(config, mesh)
  state, _, _, _ = base.max_utils.setup_training_state(model, iterator, tx, config, rng, mesh, manager)
  seed = jax.jit(lambda p, b, s, u, z: c.forward(
      model, p, b, rng, config, s, b['targets_segmentation'] != 0,
      u, z, source, return_references=True))
  infer = jax.jit(lambda p, b, s, u, z, ref, pc: c.forward(
      model, p, b, rng, config, s, b['targets_segmentation'] != 0,
      u, z, source, ref, pc, return_references=True))
  scales = jnp.ones((24, 3), jnp.float32)
  deleted = scales.at[source, jnp.asarray(columns)].set(0)
  empty = jnp.zeros((24, len(c.ROW_CONSUMER_NAMES)), jnp.float32)
  deny = empty.at[source+1:early_end+1, c.ROW_CONSUMER_NAMES.index('v_cross')].set(1)
  zero_patch = jnp.zeros((24, len(med.CONTROL_NAMES)), jnp.float32)
  all_patch = zero_patch.at[source+1:, 2:6].set(1)
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT, diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      source_layer=source, source_component=component, source_positions='all valid',
      early_v_cross_layers=list(range(source+1, early_end+1)), batch_size=bs,
      requested_sequences=len(cohort['inputs']), arms=names,
      intervention='original row denied only to early cross-V; downstream exact-endpoint donor patch',
      limitation='conditioned component mediation, not additive loss shares or per-origin multi-hop lineage',
      checks=['seed_graph', 'unused_reference', 'zero_z', 'clean_self_patch', 'denied_self_patch',
              'source_residual_scope', 'source_M_scope'])
  print('V_EXPORT_RESTORED ' + json.dumps(meta), flush=True)
  records = []
  for offset in range(0, len(cohort['inputs']), bs):
    batch = {k: jnp.asarray(v[offset:offset+bs]) for k, v in cohort.items() if k != 'sequence_hashes'}
    z0 = jnp.zeros(batch['inputs'].shape + (config.emb_dim,), jnp.float32)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      initial = seed(state.params, batch, scales, empty, z0)
      clean = infer(state.params, batch, scales, empty, z0, initial[2], zero_patch)
      removed = infer(state.params, batch, deleted, empty, z0, clean[2], zero_patch)
      z = clean[2]['post_attention'][source].astype(jnp.float32) - removed[2]['post_attention'][source].astype(jnp.float32)
      denied = infer(state.params, batch, scales, deny, z, clean[2], zero_patch)
      unused = infer(state.params, batch, scales, empty, z, denied[2], zero_patch)
      null = infer(state.params, batch, scales, deny, z0, clean[2], zero_patch)
      same_clean = infer(state.params, batch, scales, empty, z0, clean[2], all_patch)
      same_denied = infer(state.params, batch, scales, deny, z, denied[2], all_patch)
      pairs = [(initial[1], clean[1]), (unused[1], clean[1]), (null[1], clean[1]),
               (same_clean[1], clean[1]), (same_denied[1], denied[1]),
               (denied[2]['post_attention'][source], clean[2]['post_attention'][source]),
               (denied[2]['M'][source], clean[2]['M'][source])]
      errors = np.asarray([float(jnp.max(abs(a.astype(jnp.float32)-b.astype(jnp.float32)))) for a, b in pairs])
      if np.any(errors != 0):
        raise ValueError(dict(zip(meta['checks'], errors.tolist())))
      losses, tokens = [], []
      for result in (clean, removed, denied):
        loss, token = jax.device_get(result[:2]); losses.append(loss); tokens.append(token)
      for arm in matrix:
        block = arm['world'] == 'block'
        result = infer(state.params, batch, scales, empty if block else deny,
                       z, denied[2] if block else clean[2], jnp.asarray(arm['control']))
        loss, token = jax.device_get(result[:2]); losses.append(loss); tokens.append(token)
    loss, token = np.stack(losses, 1), np.stack(tokens, 1)
    if not np.isfinite(token).all(): raise ValueError('nonfinite token loss')
    np.savez_compressed(output/f'batch_{offset:03d}.npz', loss=loss, token_loss=token,
        valid=np.asarray(batch['targets_segmentation']) != 0, checks=errors,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records.append(loss); a = np.concatenate(records).astype(float)
    meta.update(completed_sequences=len(a), elapsed_seconds=time.perf_counter()-start,
                results=[dict(arm=name, **stats(a[:, i]-a[:, 0])) for i, name in enumerate(names)])
    (output/'summary.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'V_EXPORT_BATCH {len(a)}/{len(cohort["inputs"])}', flush=True)
  if writer: writer.flush()
  print('V_EXPORT_COMPLETE', flush=True)


def main(argv):
  config = base.pyconfig.initialize(argv); base.train.validate_train_config(config); run(config)


if __name__ == '__main__': app.run(main)

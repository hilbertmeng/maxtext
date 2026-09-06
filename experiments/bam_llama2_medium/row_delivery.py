"""Finite-lifetime selective row delivery, using only a causal source vector.

Retain the original row vector as a private carrier: chosen consumers see h,
the others see h-z; remove the carrier after a chosen intermediate layer.
No future reference activations, loss gradients, or final-layer cancellation.
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
from row_probe_resume import resume_batches, save_batch, save_summary

base = c.base


def delivery_arms(source):
  fields = c.ROW_CONSUMER_NAMES[:9]
  policies = dict(all=fields, crossV=['v_cross'], crossV_mlp=['v_cross', 'mlp'],
      crossV_mlp_localQK=['v_cross', 'mlp', 'local_qk'],
      mha_mlp=['q', 'k', 'v_self', 'v_cross', 'mlp'])
  if os.environ.get('BAM_DELIVERY_CONTROL_ONLY') == '1':
    policies = dict(all=fields, none=[])
  arms = []
  for end in (source+1, source+2, source+4, source+6):
    if end >= 23:
      continue
    for name, kept in policies.items():
      control = np.zeros((24, len(c.ROW_CONSUMER_NAMES)), np.float32)
      for field in fields:
        if field not in kept:
          control[source+1:end+1, c.ROW_CONSUMER_NAMES.index(field)] = 1
      if 'mlp' not in kept:
        control[source, c.ROW_CONSUMER_NAMES.index('mlp')] = 1
      control[end, c.ROW_CONSUMER_NAMES.index('cut_mlp')] = 1
      arms.append(dict(name=f'keep_{name}_through_L{end}', control=control))
  return arms


def run(config):
  if os.environ.get('BAM_CONSUMER_BARRIER') != '1':
    raise ValueError('requires the validated bf16 residual boundary')
  source = int(os.environ.get('BAM_MEDIATION_SOURCE', '11'))
  component = os.environ.get('BAM_MEDIATION_COMPONENT', 'both')
  columns = {'cross': [0, 1], 'self': [2], 'both': [0, 1, 2]}[component]
  output = Path(os.environ['BAM_MEDIATION_OUTPUT']); output.mkdir(parents=True, exist_ok=True)
  path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(path) as data:
    cohort = {k: np.asarray(data[k]) for k in ('inputs', 'targets', 'inputs_position',
        'inputs_segmentation', 'targets_segmentation', 'sequence_hashes')}
  bs = int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  matrix = delivery_arms(source)
  names = ['clean', 'row_deleted'] + [a['name'] for a in matrix]
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT, diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_layer=source,
      source_component=component, source_positions='all valid', early_v_cross_layers=[],
      batch_size=bs, requested_sequences=len(cohort['inputs']), arms=names,
      controls=[a['control'].tolist() for a in matrix],
      intervention='causal source carrier with selective consumers and intermediate lifetime cutoff',
      checks=['unused_reference', 'zero_z', 'immediate_cut_equals_deletion', 'source_M_scope'],
      limitation='collective all-origin intervention; checkpoint feasibility, not retraining outcome')
  records = resume_batches(output, meta, cohort, os.environ.get('BAM_MEDIATION_RESUME_COMMIT'))
  start = time.perf_counter()
  audit_offset = os.environ.get('BAM_DELIVERY_AUDIT_OFFSET')
  if audit_offset is not None and os.environ.get('BAM_CONSUMER_AUDIT') != '1':
    raise ValueError('delivery audit requires the intermediate capture audit flag')
  rng, writer, manager, mesh, model, _, tx = base.train.setup_mesh_and_model(config)
  iterator, _ = base.create_data_iterator(config, mesh)
  state, _, _, _ = base.max_utils.setup_training_state(model, iterator, tx, config, rng, mesh, manager)
  infer = jax.jit(lambda p,b,s,u,z: c.forward(model,p,b,rng,config,s,
      b['targets_segmentation'] != 0,u,z,source))
  scales = jnp.ones((24,3), jnp.float32)
  deleted = scales.at[source,jnp.asarray(columns)].set(0)
  empty = jnp.zeros((24,len(c.ROW_CONSUMER_NAMES)), jnp.float32)
  immediate = empty.at[source,c.ROW_CONSUMER_NAMES.index('cut_attention')].set(1)
  null_control = jnp.asarray(np.maximum.reduce([a['control'] for a in matrix]))
  print('DELIVERY_RESTORED ' + json.dumps(meta), flush=True)
  for offset in range(0,len(cohort['inputs']),bs):
    if audit_offset is not None and offset != int(audit_offset):
      continue
    if offset in records:
      continue
    batch = {k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    z0 = jnp.zeros(batch['inputs'].shape+(config.emb_dim,),jnp.float32)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      clean = infer(state.params,batch,scales,empty,z0)
      removed = infer(state.params,batch,deleted,empty,z0)
      # This source output precedes every intervention, so it is available in a
      # single ordinary forward. No downstream clean trajectory is consulted.
      z = clean[2].astype(jnp.float32)-removed[2].astype(jnp.float32)
      unused = infer(state.params,batch,scales,empty,z)
      null = infer(state.params,batch,scales,null_control,z0)
      cut = infer(state.params,batch,scales,immediate,z)
      pairs = [(unused[1],clean[1]),(null[1],clean[1]),(cut[1],removed[1]),(clean[3],removed[3])]
      checks = np.asarray([float(jnp.max(abs(a.astype(jnp.float32)-b.astype(jnp.float32)))) for a,b in pairs])
      if audit_offset is not None:
        def difference(a, b):
          a, b = np.asarray(a), np.asarray(b)
          return dict(max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))),
                      differing_coordinates=int(np.count_nonzero(a != b)))
        reconstructed = (clean[2].astype(jnp.float32)-z).astype(clean[2].dtype)
        audit = dict(metadata=meta, offset=offset,
            sequence_hashes=cohort['sequence_hashes'][offset:offset+bs].tolist(),
            endpoint_checks=dict(zip(meta['checks'],checks.tolist())),
            source_dtype=str(clean[2].dtype),
            reconstructed_source=difference(reconstructed, removed[2]),
            cut_source=difference(cut[2], clean[2]),
            unused_source=difference(unused[2], clean[2]),
            cut_deleted_mean_loss=difference(cut[0], removed[0]))
        for i, name in enumerate(('post_cut', 'mlp_input', 'mlp_output')):
          audit[name] = difference(cut[4][i], removed[4][i])
        save_summary(output/'audit.json', audit)
        print('DELIVERY_AUDIT '+json.dumps(audit), flush=True)
        return
      if np.any(checks != 0):
        raise ValueError(dict(zip(meta['checks'],checks.tolist())))
      values = [jax.device_get(x[:2]) for x in (clean,removed)]
      for arm in matrix:
        values.append(jax.device_get(infer(state.params,batch,scales,jnp.asarray(arm['control']),z)[:2]))
    loss = np.stack([x[0] for x in values],1)
    token = np.stack([x[1] for x in values],1)
    if not np.isfinite(token).all():
      raise ValueError('nonfinite token loss')
    save_batch(output/f'batch_{offset:03d}.npz', loss=loss, token_loss=token,
        checks=checks, valid=np.asarray(batch['targets_segmentation'])!=0,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records[offset] = loss
    a = np.concatenate([records[k] for k in sorted(records)]).astype(float)
    meta.update(completed_sequences=len(a),elapsed_seconds=time.perf_counter()-start,
        results=[dict(arm=name,**stats(a[:,i]-a[:,0])) for i,name in enumerate(names)])
    save_summary(output/'summary.json',meta)
    print(f'DELIVERY_BATCH {len(a)}/{len(cohort["inputs"])}',flush=True)
  if writer: writer.flush()
  print('DELIVERY_COMPLETE',flush=True)


def main(argv):
  config = base.pyconfig.initialize(argv); base.train.validate_train_config(config); run(config)


if __name__ == '__main__': app.run(main)

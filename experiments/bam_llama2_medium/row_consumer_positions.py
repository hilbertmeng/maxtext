"""Point-source row-self/cross input-denial probes; no vector artifacts."""
import hashlib
import json
import os
from pathlib import Path
import time

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

import row_mediation as med
from layers.attentions import ROW_CONSUMER_NAMES

base = med.base


def origin_positions(cohort):
  """Stable hash-based positions, shared across source components and reruns."""
  positions = []
  for i, digest in enumerate(cohort['sequence_hashes']):
    valid = np.asarray(cohort['targets_segmentation'][i]) != 0
    candidates = np.flatnonzero(valid)
    candidates = candidates[(candidates >= 64) & (candidates < len(valid) - 256)]
    if not len(candidates):
      raise ValueError('sequence has no valid origin with the required future context')
    key = hashlib.sha256(str(digest).encode() + b'row-origin-v1').digest()
    positions.append(candidates[int.from_bytes(key[:8], 'little') % len(candidates)])
  return np.asarray(positions, np.int32)


def intervention_tree(params, scales, mask, controls, z, scanned):
  tree = traverse_util.flatten_dict(med.source_controls(params, scales, scanned))
  paths = [p[:-1] for p in tree if p[-1] == 'row_sign_scales']
  for attn in paths:
    layer = base._layer_from_path(attn)
    tree[attn + ('row_source_mask',)] = (
        jnp.broadcast_to(mask, (24,) + mask.shape) if scanned else mask)
    c = controls if scanned else controls[layer]
    tree[attn + ('row_consumers',)] = c
    tree[attn[:-1] + ('row_consumers',)] = c
    tree[attn[:-1] + ('row_consumer_z',)] = (
        jnp.broadcast_to(z, (24,) + z.shape) if scanned else z)
  if not paths:
    raise ValueError('no BAM attention paths found')
  return traverse_util.unflatten_dict(tree)


def forward(model, params, batch, rng, config, scales, mask, controls, z, source):
  variables = dict(params)
  variables['causal_ablation'] = intervention_tree(
      params, scales, mask, controls, z, config.scan_layers)
  r1, r2 = jax.random.split(rng)
  (token_loss, _, _), captured = model.apply(
      variables, batch['inputs'], batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
      enable_dropout=False, rngs={'dropout': r1, 'params': r2},
      mutable=['mediation_capture'])
  refs = med.stack_capture(captured)
  # Always return the same outputs, including for null and intervention arms:
  # source capture and measurement therefore use ONE compiled executable.
  return (base._sequence_mean(token_loss, batch['targets_segmentation'] != 0),
          token_loss, refs['post_attention'][source], refs['M'][source])


def arms(source):
  result = []
  def add(name, layers=(), fields=(), **kwargs):
    c = np.zeros((24, len(ROW_CONSUMER_NAMES)), np.float32)
    for layer in layers:
      for field in fields:
        if layer < source or (layer == source and field in ROW_CONSUMER_NAMES[:8]):
          raise ValueError('cannot remove source information before it exists')
        c[layer, ROW_CONSUMER_NAMES.index(field)] = 1
    result.append(dict(name=name, control=c, **kwargs))
  add('clean')
  add('source_deleted', deleted=True)
  add(f'deny_L{source}_mlp', [source], ['mlp'])
  for layer in range(source + 1, min(source + 5, 24)):
    for field in ROW_CONSUMER_NAMES[:9]:
      add(f'deny_L{layer}_{field}', [layer], [field])
  for field in ['v_cross', 'v_self', 'q', 'k', 'local_qk', 'mix', 'read', 'write', 'mlp']:
    layers = list(range(source + 1, min(source + 5, 24)))
    if field == 'mlp':
      layers = [source] + layers
    add(f'joint_L{layers[0]}-{layers[-1]}_{field}', layers, [field])
  add('joint_all_direct_consumers', range(source + 1, min(source + 5, 24)),
      ROW_CONSUMER_NAMES[:9])
  result[-1]['control'][source, ROW_CONSUMER_NAMES.index('mlp')] = 1
  add(f'cut_L{source}_attention', [source], ['cut_attention'])
  for layer in range(source, 23):
    add(f'cut_L{layer}_mlp', [layer], ['cut_mlp'])
  # All switches exercised with z=0; checks projection/edge/cut bookkeeping.
  add('null_all_consumers', range(source + 1, min(source + 5, 24)),
      ROW_CONSUMER_NAMES, null=True)
  return result


BINS = [('origin', 0, 0), ('future_1_8', 1, 8), ('future_9_32', 9, 32),
        ('future_33_128', 33, 128), ('future_129_512', 129, 512),
        ('future_513_plus', 513, 100000), ('past', -100000, -1)]


def position_effects(token_loss, valid, positions):
  delta = token_loss.astype(np.float64) - token_loss[:, :1].astype(np.float64)
  distance = np.arange(valid.shape[1])[None, :] - positions[:, None]
  bins = []
  for _, low, high in BINS:
    mask = valid & (distance >= low) & (distance <= high)
    bins.append(np.sum(delta * mask[:, None, :], axis=-1))
  return np.stack(bins, axis=-1)


def aggregate(output, metadata):
  effects = []
  for path in sorted(output.glob('batch_*.npz')):
    with np.load(path) as x:
      effects.append(x['position_effects'])
  e = np.concatenate(effects)
  rows = []
  for i, arm in enumerate(metadata['arms']):
    totals = e[:, i, :6].sum(-1)
    rows.append(dict(arm=arm['name'], sum_delta_per_origin=float(totals.mean()),
        ci95=float(1.96 * totals.std(ddof=1) / len(e)**.5) if len(e)>1 else None,
        positive=int((totals > 0).sum()), bins_mean=e[:, i].mean(0).tolist()))
  metadata.update(completed_sequences=len(e), results=rows)
  (output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')


def run(config):
  assert not config.dense_conn and not config.fused_qkv
  assert not getattr(config, 'bam_mlp_write', False)
  source = int(os.environ.get('BAM_MEDIATION_SOURCE', '11'))
  component = os.environ.get('BAM_MEDIATION_COMPONENT', 'cross')
  if component not in ('self', 'cross'):
    raise ValueError(component)
  matrix = arms(source)
  output = Path(os.environ['BAM_MEDIATION_OUTPUT'])
  output.mkdir(parents=True, exist_ok=True)
  cohort_path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  n = int(os.environ.get('BAM_MEDIATION_N', '128'))
  bs = int(os.environ.get('BAM_RESIDUAL_ATTR_BATCH_SIZE', '1'))
  with np.load(cohort_path) as data:
    cohort = {k: np.asarray(data[k])[:n] for k in ['inputs', 'targets', 'inputs_position',
        'inputs_segmentation', 'targets_segmentation', 'sequence_hashes']}
  positions = origin_positions(cohort)
  start = time.perf_counter()
  rng, writer, manager, mesh, model, _, tx = base.train.setup_mesh_and_model(config)
  iterator, _ = base.create_data_iterator(config, mesh)
  state, _, _, _ = base.max_utils.setup_training_state(model, iterator, tx, config, rng, mesh, manager)
  infer = jax.jit(lambda p,b,r,s,m,c,z: forward(model,p,b,r,config,s,m,c,z,source))
  clean_s = jnp.ones((24,3), jnp.float32)
  deleted_s = (clean_s.at[source,2].set(0) if component == 'self'
               else clean_s.at[source,:2].set(0))
  c0 = jnp.zeros((24,len(ROW_CONSUMER_NAMES)), jnp.float32)
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'], trainer_commit=base._TRAINER_COMMIT,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      source_layer=source, source_component=component, scan_layers=config.scan_layers,
      positions=positions.tolist(), position_rule='row-origin-v1: hash, [64,T-256)',
      prediction_semantics='loss at position s predicts token s+1',
      batch_size=bs, requested_sequences=n, bins=BINS, controls=ROW_CONSUMER_NAMES,
      arms=[dict(a,control=a['control'].tolist()) for a in matrix],
      setup_seconds=time.perf_counter()-start)
  print('CONSUMERS_RESTORED ' + json.dumps({k:v for k,v in meta.items() if k!='arms'}), flush=True)
  for offset in range(0,n,bs):
    target = output / f'batch_{offset:03d}.npz'
    if target.exists():
      continue
    batch = {k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    pos = positions[offset:offset+bs]
    mask = jnp.arange(batch['inputs'].shape[1])[None,:] == jnp.asarray(pos)[:,None]
    z0 = jnp.zeros(batch['inputs'].shape + (config.emb_dim,), jnp.float32)
    batch_start = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      clean = infer(state.params,batch,rng,clean_s,mask,c0,z0)
      deleted = infer(state.params,batch,rng,deleted_s,mask,c0,z0)
      z = clean[2].astype(jnp.float32) - deleted[2].astype(jnp.float32)
      outside = float(jnp.max(jnp.where(mask[...,None],0,abs(z))))
      source_m_error = float(jnp.max(abs(clean[3]-deleted[3])))
      if outside != 0 or source_m_error != 0:
        raise ValueError(f'source scope error: outside={outside}, M={source_m_error}')
      null_result = infer(state.params,batch,rng,clean_s,mask,
                          jnp.asarray(matrix[-1]['control']),z0)
      null_error = float(jnp.max(abs(null_result[1]-clean[1])))
      if null_error != 0:
        raise ValueError(f'zero-increment control failed before arm sweep: {null_error}')
      losses, tokens = [], []
      for i,a in enumerate(matrix):
        if i < 2:
          result = (clean,deleted)[i]
        elif a.get('null'):
          result = null_result
        else:
          result = infer(state.params,batch,rng,clean_s,mask,jnp.asarray(a['control']),
                         z0 if a.get('null') else z)
        loss,tok = jax.device_get(result[:2])
        losses.append(loss); tokens.append(tok)
        if offset == 0:
          print(f'CONSUMER_ARM {i+1}/{len(matrix)} {a["name"]}', flush=True)
      token = np.stack(tokens,axis=1)
      valid = np.asarray(batch['targets_segmentation']) != 0
      effects = position_effects(token,valid,pos)
      null_error = float(np.max(abs(token[:,-1]-token[:,0])))
      past_mask = np.arange(valid.shape[1])[None,:] < pos[:,None]
      past_error = float(np.max(np.where(past_mask[:,None,:],
          abs(token-token[:,:1]),0)))
      if null_error != 0 or past_error != 0:
        raise ValueError(f'negative control failed: null={null_error}, past={past_error}')
    if not np.isfinite(token).all():
      raise ValueError('nonfinite loss')
    np.savez_compressed(target,loss=np.stack(losses,axis=1),token_loss=token,valid=valid,
        positions=pos,position_effects=effects,source_scope_error=np.asarray([outside,source_m_error]),
        null_max_error=null_error,past_max_error=past_error,
        source_z_norm=np.asarray(jnp.linalg.norm(z,axis=-1))[np.asarray(mask)],
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    meta['elapsed_seconds'] = time.perf_counter()-start
    aggregate(output,meta)
    print(f'CONSUMERS_BATCH {offset+bs}/{n} seconds={time.perf_counter()-batch_start:.2f} '
          f'delete_sum={effects[:,1,:6].sum(-1).mean():.6f} null={null_error}',flush=True)
  if writer:
    writer.flush()
  print('CONSUMERS_COMPLETE',flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv)
  base.train.validate_train_config(config)
  run(config)


if __name__ == '__main__':
  app.run(main)

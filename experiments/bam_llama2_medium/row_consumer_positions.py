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
    if os.environ.get('BAM_CONSUMER_BARRIER') == '1':
      tree[attn[:-1] + ('row_consumer_barrier',)] = (
          jnp.zeros(24) if scanned else jnp.asarray(0))
    if os.environ.get('BAM_CONSUMER_MLP_EXPORT') == '1':
      tree[attn[:-1] + ('row_export_capture',)] = (
          jnp.zeros(24) if scanned else jnp.asarray(0))
    if os.environ.get('BAM_V_EXPORT_RECIPIENT_SET') == 'expanded':
      tree[attn + ('row_export_std_boundary',)] = (
          jnp.zeros(24) if scanned else jnp.asarray(0))
  if not paths:
    raise ValueError('no BAM attention paths found')
  return traverse_util.unflatten_dict(tree)


def forward(model, params, batch, rng, config, scales, mask, controls, z, source,
            patch_refs=None, patch_controls=None, return_references=False):
  variables = dict(params)
  variables['causal_ablation'] = intervention_tree(
      params, scales, mask, controls, z, config.scan_layers)
  if patch_refs is not None:
    if patch_controls is None:
      raise ValueError('patch references require explicit recipient controls')
    tree = traverse_util.flatten_dict(variables['causal_ablation'])
    patch = traverse_util.flatten_dict(med.patch_tree(
        params, scales, patch_refs, patch_controls, jnp.zeros_like(z),
        config.scan_layers))
    # Only instantiate the recipients under test. Unused QK/V routing and
    # residual-cancellation machinery changes the compiled graph unnecessarily.
    names = {'med_full', 'med_full_scale', 'med_M', 'med_M_scale',
             'med_mlp', 'med_mlp_scale'}
    if os.environ.get('BAM_V_EXPORT_RECIPIENT_SET') == 'expanded':
      names.update({'med_std', 'med_std_scale'})
    tree.update({p: value for p, value in patch.items() if p[-1] in names})
    variables['causal_ablation'] = traverse_util.unflatten_dict(tree)
  r1, r2 = jax.random.split(rng)
  (token_loss, _, _), captured = model.apply(
      variables, batch['inputs'], batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
      enable_dropout=False, rngs={'dropout': r1, 'params': r2},
      mutable=(['mediation_capture','row_cross_probe'] if
               os.environ.get('BAM_CONSUMER_ALPHA_GEOMETRY')=='1' else ['mediation_capture']))
  refs = med.stack_capture(captured)
  if return_references:
    return (base._sequence_mean(token_loss, batch['targets_segmentation'] != 0),
            token_loss, refs)
  audit_refs = []
  if os.environ.get('BAM_CONSUMER_ALPHA_GEOMETRY')=='1':
    audit_refs.append(med.sign.stacked(captured,'alpha_stats')[source])
  if os.environ.get('BAM_CONSUMER_MLP_EXPORT') == '1':
    flat=traverse_util.flatten_dict(captured['mediation_capture'])
    name='trace_consumer_post_mlp'
    if config.scan_layers:
      values=[base._unwrap(v) for p,v in flat.items() if p[-1]==name]
      if len(values)!=1:raise ValueError(name)
      audit_refs.append(base._layer_axis_first(values[0],name)[source])
    else:
      audit_refs.append(next(base._unwrap(v) for p,v in flat.items()
                            if p[-1]==name and base._layer_from_path(p)==source))
  if os.environ.get('BAM_CONSUMER_AUDIT') == '1':
    flat = traverse_util.flatten_dict(captured['mediation_capture'])
    for name in ('trace_consumer_post_cut','trace_consumer_mlp_input'):
      if config.scan_layers:
        values = [base._unwrap(v) for p,v in flat.items() if p[-1]==name]
        if len(values)!=1:raise ValueError(name)
        audit_refs.append(base._layer_axis_first(values[0],name)[source])
      else:
        audit_refs.append(next(base._unwrap(v) for p,v in flat.items()
                              if p[-1]==name and base._layer_from_path(p)==source))
    audit_refs.append(refs['mlp'][source])
  # Always return the same outputs, including for null and intervention arms:
  # source capture and measurement therefore use ONE compiled executable.
  return (base._sequence_mean(token_loss, batch['targets_segmentation'] != 0),
          token_loss, refs['post_attention'][source], refs['M'][source], tuple(audit_refs))


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
  if os.environ.get('BAM_CONSUMER_ARM_SET') == 'interactions':
    # Retain the validated screen arms and add joint tests. Interactions are
    # paired contrasts, not sums of isolated consumer importance percentages.
    downstream = list(range(source + 1, min(source + 5, 24)))
    add('joint_source_mlp_cross_v', downstream, ['v_cross'])
    result[-1]['control'][source, ROW_CONSUMER_NAMES.index('mlp')] = 1
    add('joint_source_and_downstream_mlp_cross_v', downstream, ['mlp','v_cross'])
    result[-1]['control'][source, ROW_CONSUMER_NAMES.index('mlp')] = 1
    add('joint_downstream_all_v', downstream, ['v_self','v_cross'])
    for end in downstream[1:]:
      add(f'cumulative_L{source+1}-{end}_cross_v', range(source+1,end+1), ['v_cross'])
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
  source_mode = os.environ.get('BAM_CONSUMER_SOURCE_MODE', 'all')
  if source_mode != 'all':
    raise ValueError('Consumer probes cover all valid source positions; sparse point sampling is retired.')
  if component not in ('self', 'cross', 'both'):
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
  if component == 'both':deleted_s=clean_s.at[source].set(0)
  c0 = jnp.zeros((24,len(ROW_CONSUMER_NAMES)), jnp.float32)
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'], trainer_commit=base._TRAINER_COMMIT,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      source_layer=source, source_component=component, scan_layers=config.scan_layers,
      source_mode=source_mode, calculation_barrier=os.environ.get('BAM_CONSUMER_BARRIER')=='1',
      arm_set=os.environ.get('BAM_CONSUMER_ARM_SET','screen'),
      positions=positions.tolist(), position_rule='row-origin-v1: hash, [64,T-256)',
      prediction_semantics='loss at position s predicts token s+1',
      batch_size=bs, requested_sequences=n, bins=BINS, controls=ROW_CONSUMER_NAMES,
      arms=[dict(a,control=a['control'].tolist()) for a in matrix],
      setup_seconds=time.perf_counter()-start)
  geometry_names=['self_norm','cross_norm','both_norm','self_cross_cosine',
      'cosine_negative_fraction','both_over_self_plus_cross_norm','additive_closure_relative']
  if component=='both':meta['source_geometry_columns']=geometry_names
  if source_mode=='all':
    meta['positions']=None
    meta['position_rule']='all valid origins; no origin/future loss partition'
    meta['bins']=[('all_predictions',0,100000)]+[(f'unused_{i}',0,0) for i in range(1,7)]
  print('CONSUMERS_RESTORED ' + json.dumps({k:v for k,v in meta.items() if k!='arms'}), flush=True)
  for offset in range(0,n,bs):
    target = output / f'batch_{offset:03d}.npz'
    if target.exists():
      continue
    batch = {k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    pos = positions[offset:offset+bs]
    mask = jnp.arange(batch['inputs'].shape[1])[None,:] == jnp.asarray(pos)[:,None]
    if source_mode=='all':
      mask = batch['targets_segmentation'] != 0
    z0 = jnp.zeros(batch['inputs'].shape + (config.emb_dim,), jnp.float32)
    batch_start = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      clean = infer(state.params,batch,rng,clean_s,mask,c0,z0)
      deleted = infer(state.params,batch,rng,deleted_s,mask,c0,z0)
      z = clean[2].astype(jnp.float32) - deleted[2].astype(jnp.float32)
      geometry={}
      if component=='both':
        self_ref=infer(state.params,batch,rng,clean_s.at[source,2].set(0),mask,c0,z0)
        cross_ref=infer(state.params,batch,rng,clean_s.at[source,:2].set(0),mask,c0,z0)
        zs=clean[2].astype(jnp.float32)-self_ref[2].astype(jnp.float32)
        zc=clean[2].astype(jnp.float32)-cross_ref[2].astype(jnp.float32)
        ns,nc,nb=[jnp.linalg.norm(x,axis=-1) for x in (zs,zc,z)]
        cosine=jnp.sum(zs*zc,-1)/jnp.maximum(ns*nc,1e-12)
        terms=jnp.stack([ns,nc,nb,cosine,(cosine<0).astype(jnp.float32),
            nb/jnp.maximum(ns+nc,1e-12),
            jnp.linalg.norm(z-zs-zc,axis=-1)/jnp.maximum(ns+nc,1e-12)],axis=-1)
        geometry['source_geometry']=np.asarray(base._sequence_mean(terms,mask))
        if os.environ.get('BAM_CONSUMER_ALPHA_GEOMETRY')=='1':
          # Scalar per-token data, not residual/read vectors. These distinguish
          # true per-token centering from cancellation only after averaging.
          alpha_stats=clean[4][0].astype(jnp.float32)
          geometry['alpha_coefficient_sum']=np.asarray(1+alpha_stats[...,3]-alpha_stats[...,4])
          geometry['source_geometry_token']=np.asarray(terms)
      outside = float(jnp.max(jnp.where(mask[...,None],0,abs(z))))
      source_m_error = float(jnp.max(abs(clean[3]-deleted[3])))
      if outside != 0 or source_m_error != 0:
        raise ValueError(f'source scope error: outside={outside}, M={source_m_error}')
      if os.environ.get('BAM_CONSUMER_AUDIT') == '1':
        cut_control = c0.at[source,ROW_CONSUMER_NAMES.index('cut_attention')].set(1)
        unchanged = infer(state.params,batch,rng,clean_s,mask,c0,z)
        cut = infer(state.params,batch,rng,clean_s,mask,cut_control,z)
        reconstructed = (clean[2].astype(jnp.float32)-z).astype(clean[2].dtype)
        audit = dict(position=pos.tolist(),residual_dtype=str(clean[2].dtype),
            barrier=os.environ.get('BAM_CONSUMER_BARRIER')=='1',
            zero_control_nonzero_z_token_error=float(jnp.max(abs(unchanged[1]-clean[1]))),
            zero_control_nonzero_z_source_error=float(jnp.max(abs(unchanged[2]-clean[2]))),
            cut_source_pre_input_error=float(jnp.max(abs(cut[2]-clean[2]))),
            cut_source_M_error=float(jnp.max(abs(cut[3]-clean[3]))),
            reconstructed_source_max_error=float(jnp.max(abs(reconstructed-deleted[2]))),
            reconstructed_source_different_coordinates=int(jnp.count_nonzero(reconstructed!=deleted[2])),
            source_cut_delete_token_error=float(jnp.max(abs(cut[1]-deleted[1]))),
            source_cut_delete_mean_error=float(jnp.mean(cut[1]-deleted[1])))
        for i,name in enumerate(('post_cut','mlp_input','mlp_output')):
          audit[f'cut_delete_{name}_max_error']=float(jnp.max(abs(cut[4][i]-deleted[4][i])))
          audit[f'cut_delete_{name}_different_coordinates']=int(jnp.count_nonzero(cut[4][i]!=deleted[4][i]))
        (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('CONSUMER_AUDIT '+json.dumps(audit),flush=True)
        return
      null_result = infer(state.params,batch,rng,clean_s,mask,
                          jnp.asarray(matrix[-1]['control']),z0)
      null_error = float(jnp.max(abs(null_result[1]-clean[1])))
      if null_error != 0:
        raise ValueError(f'zero-increment control failed before arm sweep: {null_error}')
      unused_reference = infer(state.params,batch,rng,clean_s,mask,c0,z)
      unused_error = float(jnp.max(abs(unused_reference[1]-clean[1])))
      cut_control = c0.at[source,ROW_CONSUMER_NAMES.index('cut_attention')].set(1)
      immediate_cut = infer(state.params,batch,rng,clean_s,mask,cut_control,z)
      cut_error = float(jnp.max(abs(immediate_cut[1]-deleted[1])))
      if unused_error!=0 or cut_error!=0:
        raise ValueError(f'boundary controls failed: unused={unused_error}, cut={cut_error}')
      losses, tokens = [], []
      for i,a in enumerate(matrix):
        if i < 2:
          result = (clean,deleted)[i]
        elif a.get('null'):
          result = null_result
        elif a['name']==f'cut_L{source}_attention':
          result = immediate_cut
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
      if source_mode=='all':
        effects = np.zeros_like(effects)
        effects[...,0] = np.sum((token.astype(np.float64)-token[:,:1].astype(np.float64))*valid[:,None],-1)
        past_error = 0  # No unaffected prefix exists under all-position denial.
      if null_error != 0 or past_error != 0:
        raise ValueError(f'negative control failed: null={null_error}, past={past_error}')
    if not np.isfinite(token).all():
      raise ValueError('nonfinite loss')
    np.savez_compressed(target,loss=np.stack(losses,axis=1),token_loss=token,valid=valid,
        positions=pos,position_effects=effects,source_scope_error=np.asarray([outside,source_m_error]),
        null_max_error=null_error,past_max_error=past_error,
        unused_reference_error=unused_error,immediate_cut_error=cut_error,
        source_z_norm=np.asarray(jnp.sqrt(jnp.sum(z*z,axis=(1,2)))),
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs],**geometry)
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

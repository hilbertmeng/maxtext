"""All-token causal counterfactuals: own-origin versus earlier-origin row effects.

For query t, foreign K/V/M come from a donor prefix world while the diagonal
comes from t's own world. Causality makes every earlier position independent
of t's source intervention. Thus all T own-origin counterfactuals can run in
parallel, without sampling origins. Arithmetic endpoint controls remain required.
"""
import hashlib
import json
import os
from pathlib import Path

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

import row_mediation as med
from row_probe_resume import resume_batches, save_batch, save_summary
from analyze_row_mediation import stats

base = med.base


def variables_for(params, scales, refs, enabled, scanned):
  tree = traverse_util.flatten_dict(med.source_controls(params, scales, scanned))
  for attn in [p[:-1] for p in tree if p[-1] == 'row_sign_scales']:
    layer = base._layer_from_path(attn)
    take = lambda x: x if scanned else x[layer]
    for name in ('key', 'value', 'M'):
      tree[attn+(f'row_foreign_{name}',)] = take(refs[name])
    tree[attn+('row_foreign_enabled',)] = take(enabled)
    tree[attn[:-1]+('row_consumer_barrier',)] = take(jnp.zeros(24))
  return dict(params, causal_ablation=traverse_util.unflatten_dict(tree))


def forward(model, params, batch, rng, config, scales, refs, enabled):
  variables = variables_for(params, scales, refs, enabled, config.scan_layers)
  r1, r2 = jax.random.split(rng)
  (tokens, _, _), captured = model.apply(variables, batch['inputs'],
      batch['inputs_position'], decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
      enable_dropout=False, rngs={'dropout':r1, 'params':r2}, mutable=['mediation_capture'])
  captured = med.stack_capture(captured)
  # The cache read at l is M_out[l-1]. Layer zero is disabled in all interventions.
  refs = dict(key=captured['key'], value=captured['value'],
      M=jnp.concatenate((jnp.zeros_like(captured['M'][:1]),captured['M'][:-1]),0))
  return base._sequence_mean(tokens,batch['targets_segmentation']!=0), tokens, refs


def run(config):
  source = int(os.environ.get('BAM_MEDIATION_SOURCE','11'))
  component = os.environ.get('BAM_MEDIATION_COMPONENT','both')
  columns = {'cross':[0,1], 'self':[2], 'both':[0,1,2]}[component]
  output = Path(os.environ['BAM_MEDIATION_OUTPUT']); output.mkdir(parents=True,exist_ok=True)
  path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(path) as data:
    cohort = {k:np.asarray(data[k]) for k in ('inputs','targets','inputs_position',
        'inputs_segmentation','targets_segmentation','sequence_hashes')}
  bs = int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  names = ['clean','all_origins_deleted','own_origin_only_deleted','earlier_origins_only_deleted']
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT, diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_layer=source,
      source_component=component, batch_size=bs, requested_sequences=len(cohort['inputs']),
      arms=names, source_positions='all valid; exact causal diagonal construction in real arithmetic',
      checks=['clean_self_reference','deleted_self_reference','clean_disabled_reference','deleted_disabled_reference'],
      limitation='foreign/own effects interact; report both conditional differences and interaction')
  records=resume_batches(output,meta,cohort,os.environ.get('BAM_MEDIATION_RESUME_COMMIT'))
  rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
  iterator,_=base.create_data_iterator(config,mesh)
  state,_,_,_=base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  print('TOKEN_WORLDS_RESTORED '+json.dumps(meta),flush=True)
  infer=jax.jit(lambda p,b,s,r,e:forward(model,p,b,rng,config,s,r,e))
  scales=jnp.ones((24,3),jnp.float32)
  deleted=scales.at[source,jnp.asarray(columns)].set(0)
  inactive=jnp.zeros(24,jnp.bool_)
  active=jnp.arange(24)>source
  for offset in range(0,len(cohort['inputs']),bs):
    if offset in records:continue
    batch={k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    b,t=batch['inputs'].shape
    zero=dict(key=jnp.zeros((24,b,t,config.num_query_heads,config.head_dim),config.dtype),
        value=jnp.zeros((24,b,t,config.num_query_heads,config.head_dim),config.dtype),
        M=jnp.zeros((24,b,t,config.bam_k,config.bam_v),config.dtype))
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      clean=infer(state.params,batch,scales,zero,inactive)
      removed=infer(state.params,batch,deleted,zero,inactive)
      checks=[]
      for s,r,enable,expected in [(scales,clean[2],active,clean),
          (deleted,removed[2],active,removed),(scales,removed[2],inactive,clean),
          (deleted,clean[2],inactive,removed)]:
        actual=infer(state.params,batch,s,r,enable)
        checks.append(float(jnp.max(abs(actual[1].astype(jnp.float32)-expected[1].astype(jnp.float32)))))
      if any(checks):
        save_summary(output/'failed_checks.json',dict(offset=offset,checks=dict(zip(meta['checks'],checks))))
        raise ValueError(dict(zip(meta['checks'],checks)))
      own=infer(state.params,batch,deleted,clean[2],active)
      earlier=infer(state.params,batch,scales,removed[2],active)
      values=[jax.device_get(x[:2]) for x in (clean,removed,own,earlier)]
    loss=np.stack([v[0] for v in values],1);tokens=np.stack([v[1] for v in values],1)
    if not np.isfinite(tokens).all():raise ValueError('nonfinite token loss')
    save_batch(output/f'batch_{offset:03d}.npz',loss=loss,token_loss=tokens,
        checks=np.asarray(checks),valid=np.asarray(batch['targets_segmentation'])!=0,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records[offset]=loss
    a=np.concatenate([records[k] for k in sorted(records)]).astype(float)
    contrasts={'all_deleted':a[:,1]-a[:,0], 'own_only':a[:,2]-a[:,0],
        'earlier_only':a[:,3]-a[:,0], 'earlier_given_own_deleted':a[:,1]-a[:,2],
        'own_given_earlier_deleted':a[:,1]-a[:,3], 'interaction':a[:,1]-a[:,2]-a[:,3]+a[:,0]}
    meta.update(completed_sequences=len(a),results={k:stats(v) for k,v in contrasts.items()})
    save_summary(output/'summary.json',meta)
    print(f'TOKEN_WORLDS_BATCH {len(a)}/{len(cohort["inputs"])}',flush=True)
  if writer:writer.flush()
  print('TOKEN_WORLDS_COMPLETE',flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv);base.train.validate_train_config(config);run(config)


if __name__=='__main__':app.run(main)

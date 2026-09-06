"""All-origin consumer denial with clean foreign prefixes and own-token loss.

No origin sampling. Every t evolves in its own counterfactual while earlier
K/V/M remain clean. Store every valid token loss, not hidden-state vectors.
"""
import hashlib
import json
import os
from pathlib import Path

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

import row_token_worlds as worlds
from row_consumer_positions import arms as consumer_arms
from row_probe_resume import resume_batches, save_batch, save_summary
from analyze_row_mediation import stats
from layers.attentions import _split_row_difference, ROW_CONSUMER_NAMES

base = worlds.base


def selected_arms(source):
  """Joint screen plus lifetime, then optional per-layer localization."""
  all_arms = consumer_arms(source)
  arm_set = os.environ.get('BAM_OWN_CONSUMER_SET', 'joint')
  if arm_set == 'terminal':
    # Close the lifetime curve through the final layer, with shared anchors.
    result = [a for a in all_arms if a['name'] in (
        f'cut_L{source}_attention', 'cut_L22_mlp')]
    for field in ('mlp', 'cut_attention', 'cut_mlp'):
      control = np.zeros_like(all_arms[0]['control'])
      control[23, ROW_CONSUMER_NAMES.index(field)] = 1
      name = 'deny_L23_mlp' if field == 'mlp' else 'cut_L23_' + field.removeprefix('cut_')
      result.append(dict(name=name, control=control))
    return result
  if arm_set == 'individual':
    return [a for a in all_arms if a['name'].startswith('deny_')]
  return [a for a in all_arms if a['name'].startswith(('joint_', 'cut_'))
          or a['name'] == f'deny_L{source}_mlp']


def run(config):
  source = int(os.environ.get('BAM_MEDIATION_SOURCE', '11'))
  component = os.environ.get('BAM_MEDIATION_COMPONENT', 'cross')
  columns = {'cross':[0,1], 'self':[2], 'both':[0,1,2]}[component]
  path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(path) as data:
    cohort = {k:np.asarray(data[k]) for k in ('inputs','targets','inputs_position',
        'inputs_segmentation','targets_segmentation','sequence_hashes')}
  output = Path(os.environ['BAM_MEDIATION_OUTPUT'])
  output.mkdir(parents=True, exist_ok=True)
  bs = int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  arms = selected_arms(source)
  names = ['clean', 'all_origins_deleted', 'own_origin_only_deleted'] + [a['name'] for a in arms]
  checks_names = ['clean_self_reference', 'deleted_self_reference',
      'clean_disabled_reference', 'deleted_disabled_reference',
      'zero_controls_nonzero_z', 'null_all_consumers', 'own_v_cross_null']
  meta = dict(base_config_class=base._BASE_CONFIG_CLASS, checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT, diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_layer=source,
      source_component=component, batch_size=bs, requested_sequences=len(cohort['inputs']),
      arms=names, checks=checks_names, controls={a['name']:a['control'].tolist() for a in arms},
      source_positions='all valid origins; own-token loss; foreign prefixes remain clean',
      interpretation='consumer necessity at original position; not evidence of useful export',
      arm_set=os.environ.get('BAM_OWN_CONSUMER_SET','joint'))
  records = resume_batches(output,meta,cohort,os.environ.get('BAM_MEDIATION_RESUME_COMMIT'))
  rng,writer,manager,mesh,model,_,tx = base.train.setup_mesh_and_model(config)
  iterator,_ = base.create_data_iterator(config,mesh)
  state,_,_,_ = base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  print('OWN_CONSUMERS_RESTORED '+json.dumps(meta),flush=True)
  infer = jax.jit(lambda p,b,s,r,e,c,z: worlds.forward(model,p,b,rng,config,s,r,e,
      (b['targets_segmentation'] != 0,c,z)))
  scales = jnp.ones((24,3),jnp.float32)
  deleted = scales.at[source,jnp.asarray(columns)].set(0)
  inactive = jnp.zeros((24,3),jnp.bool_)
  active = jnp.broadcast_to((jnp.arange(24)>source)[:,None],(24,3))
  empty_c = jnp.zeros_like(jnp.asarray(arms[0]['control']))
  all_c = empty_c.at[source+1:source+5].set(1)
  v_cross_c = empty_c.at[source+1:source+5,3].set(1)
  for offset in range(0,len(cohort['inputs']),bs):
    if offset in records:
      continue
    batch = {k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    b,t = batch['inputs'].shape
    zero = dict(key=jnp.zeros((24,b,t,config.num_query_heads,config.head_dim),config.dtype),
        value=jnp.zeros((24,b,t,config.num_query_heads,config.head_dim),config.dtype),
        M=jnp.zeros((24,b,t,config.bam_k,config.bam_v),config.dtype))
    z0 = (jnp.zeros((b,t,config.emb_dim),jnp.float32),)*2
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      clean = infer(state.params,batch,scales,zero,inactive,empty_c,z0)
      removed = infer(state.params,batch,deleted,zero,inactive,empty_c,z0)
      z = _split_row_difference(clean[3]['post_attention'][source],removed[3]['post_attention'][source])
      checks = []
      for s,r,e,c,vector,expected in [
          (scales,clean[2],active,empty_c,z0,clean),
          (deleted,removed[2],active,empty_c,z0,removed),
          (scales,removed[2],inactive,empty_c,z0,clean),
          (deleted,clean[2],inactive,empty_c,z0,removed),
          (scales,clean[2],active,empty_c,z,clean),
          (scales,clean[2],active,all_c,z0,clean),
          (scales,clean[2],active,v_cross_c,z,clean)]:
        actual = infer(state.params,batch,s,r,e,c,vector)
        checks.append(float(jnp.max(abs(actual[1].astype(jnp.float32)-expected[1].astype(jnp.float32)))))
      if any(checks):
        save_summary(output/'failed_checks.json',dict(offset=offset,checks=dict(zip(checks_names,checks))))
        raise ValueError(dict(zip(checks_names,checks)))
      own = infer(state.params,batch,deleted,clean[2],active,empty_c,z0)
      values = [jax.device_get(x[:2]) for x in (clean,removed,own)]
      for arm in arms:
        result = infer(state.params,batch,scales,clean[2],active,jnp.asarray(arm['control']),z)
        values.append(jax.device_get(result[:2]))
    loss = np.stack([v[0] for v in values],1)
    tokens = np.stack([v[1] for v in values],1)
    if not np.isfinite(tokens).all():
      raise ValueError('nonfinite token loss')
    save_batch(output/f'batch_{offset:03d}.npz',loss=loss,token_loss=tokens,
        checks=np.asarray(checks),valid=np.asarray(batch['targets_segmentation'])!=0,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records[offset] = loss
    a = np.concatenate([records[k] for k in sorted(records)]).astype(float)
    meta.update(completed_sequences=len(a),results={name:stats(a[:,i]-a[:,0])
        for i,name in enumerate(names) if i})
    save_summary(output/'summary.json',meta)
    print(f'OWN_CONSUMERS_BATCH {len(a)}/{len(cohort["inputs"])}',flush=True)
  if writer:
    writer.flush()
  print('OWN_CONSUMERS_COMPLETE',flush=True)


def main(argv):
  config = base.pyconfig.initialize(argv)
  base.train.validate_train_config(config)
  run(config)


if __name__ == '__main__':
  app.run(main)

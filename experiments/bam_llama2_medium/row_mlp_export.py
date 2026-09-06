"""All-origin source-MLP response, denied to local versus cross-token consumers.

The reference is the effective post-MLP residual change when source row input is
denied only to the source MLP. It is not the whole-row deletion trajectory.
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

base=c.base


def export_arms(source):
  result=[]
  def add(name,layers=(),fields=()):
    controls=np.zeros((24,len(c.ROW_CONSUMER_NAMES)),np.float32)
    for l in layers:
      for f in fields:controls[l,c.ROW_CONSUMER_NAMES.index(f)]=1
    result.append(dict(name=name,control=controls))
  add('clean');add('source_mlp_input_denied')
  fields=['v_self','v_cross','mlp','local_qk','read','write']
  for l in range(source+1,source+5):
    for f in fields:add(f'deny_L{l}_{f}',[l],[f])
  for f in fields:add(f'joint_L{source+1}-{source+4}_{f}',range(source+1,source+5),[f])
  add('joint_vcross_mlp',range(source+1,source+5),['v_cross','mlp'])
  add('immediate_cut',[source],['cut_mlp'])
  add('null',range(source+1,source+5),fields)
  return result


def run(config):
  if os.environ.get('BAM_CONSUMER_BARRIER')!='1' or os.environ.get('BAM_CONSUMER_MLP_EXPORT')!='1':
    raise ValueError('MLP export requires explicit source and post-MLP boundaries')
  source=int(os.environ.get('BAM_MEDIATION_SOURCE','11'))
  component=os.environ.get('BAM_MEDIATION_COMPONENT','cross')
  if component not in ('self','cross','both'):raise ValueError(component)
  output=Path(os.environ['BAM_MEDIATION_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(path) as d:
    cohort={k:np.asarray(d[k]) for k in ['inputs','targets','inputs_position',
        'inputs_segmentation','targets_segmentation','sequence_hashes']}
  bs=int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE']);matrix=export_arms(source)
  start=time.perf_counter()
  rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
  iterator,_=base.create_data_iterator(config,mesh)
  state,_,_,_=base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  infer=jax.jit(lambda p,b,s,controls,z:c.forward(model,p,b,rng,config,s,
      b['targets_segmentation']!=0,controls,z,source))
  scales=jnp.ones((24,3),jnp.float32)
  deleted=scales.at[source,{'cross':jnp.array([0,1]),'self':jnp.array([2]),
                           'both':jnp.array([0,1,2])}[component]].set(0)
  empty=jnp.zeros((24,len(c.ROW_CONSUMER_NAMES)),jnp.float32)
  mlp=empty.at[source,c.ROW_CONSUMER_NAMES.index('mlp')].set(1)
  meta=dict(base_config_class=base._BASE_CONFIG_CLASS,checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT,diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_layer=source,
      source_component=component,source_positions='all valid',batch_size=bs,
      requested_sequences=len(cohort['inputs']),
      reference='effective source post-MLP difference from denying row input to that MLP only',
      arms=[dict(a,control=a['control'].tolist()) for a in matrix])
  records=[]
  for offset in range(0,len(cohort['inputs']),bs):
    batch={k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    z0=jnp.zeros(batch['inputs'].shape+(config.emb_dim,),jnp.float32)
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      clean=infer(state.params,batch,scales,empty,z0)
      row_deleted=infer(state.params,batch,deleted,empty,z0)
      zrow=clean[2].astype(jnp.float32)-row_deleted[2].astype(jnp.float32)
      denied=infer(state.params,batch,scales,mlp,zrow)
      response=clean[4][0].astype(jnp.float32)-denied[4][0].astype(jnp.float32)
      cut=infer(state.params,batch,scales,jnp.asarray(matrix[-2]['control']),response)
      null=infer(state.params,batch,scales,jnp.asarray(matrix[-1]['control']),z0)
      unused=infer(state.params,batch,scales,empty,response)
      errors=np.array([float(jnp.max(abs(cut[1]-denied[1]))),
          float(jnp.max(abs(null[1]-clean[1]))),float(jnp.max(abs(unused[1]-clean[1]))),
          float(jnp.max(abs(denied[2]-clean[2]))),float(jnp.max(abs(denied[3]-clean[3])))])
      if np.any(errors!=0):raise ValueError(f'MLP response boundary/null/scope: {errors}')
      losses=[];tokens=[]
      for i,a in enumerate(matrix):
        if i<2:result=(clean,denied)[i]
        elif a['name']=='immediate_cut':result=cut
        elif a['name']=='null':result=null
        else:result=infer(state.params,batch,scales,jnp.asarray(a['control']),response)
        loss,token=jax.device_get(result[:2]);losses.append(loss);tokens.append(token)
    loss=np.stack(losses,1);token=np.stack(tokens,1)
    if not np.isfinite(token).all():raise ValueError('nonfinite loss')
    np.savez_compressed(output/f'batch_{offset:03d}.npz',loss=loss,token_loss=token,
        valid=np.asarray(batch['targets_segmentation'])!=0,checks=errors,
        response_norm=np.asarray(jnp.linalg.norm(response,axis=(1,2))),
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records.append(loss);a=np.concatenate(records).astype(float)
    meta.update(completed_sequences=len(a),elapsed_seconds=time.perf_counter()-start,
        results=[dict(arm=m['name'],**stats(a[:,i]-a[:,0])) for i,m in enumerate(matrix)])
    (output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'MLP_EXPORT_BATCH {len(a)}/{len(cohort["inputs"])}',flush=True)
  if writer:writer.flush()
  print('MLP_EXPORT_COMPLETE',flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv);base.train.validate_train_config(config);run(config)


if __name__=='__main__':app.run(main)

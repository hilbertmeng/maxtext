"""All-position self/cross/joint row deletion across neighboring layers."""
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

import row_consumer_positions as consumer
from analyze_row_mediation import stats

base = consumer.base


def neighbor_arms(layers):
  names = ['clean']; scales = [np.ones((24,3),np.float32)]
  for layer in layers:
    for component, columns in [('cross',[0,1]),('self',[2]),('both',[0,1,2])]:
      s=np.ones((24,3),np.float32); s[layer,columns]=0
      names.append(f'L{layer}_{component}');scales.append(s)
  return names,scales


def run(config):
  output=Path(os.environ['BAM_MEDIATION_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  layers=[int(x) for x in os.environ.get('BAM_NEIGHBOR_LAYERS','8,9,10,11,12,13,14').split(',')]
  names,scales=neighbor_arms(layers)
  bs=int(os.environ['BAM_RESIDUAL_ATTR_BATCH_SIZE'])
  with np.load(path) as d:
    cohort={k:np.asarray(d[k]) for k in ['inputs','targets','inputs_position',
        'inputs_segmentation','targets_segmentation','sequence_hashes']}
  start=time.perf_counter()
  rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
  iterator,_=base.create_data_iterator(config,mesh)
  state,_,_,_=base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  control=jnp.zeros((24,len(consumer.ROW_CONSUMER_NAMES)),jnp.float32)
  # Keep exactly the consumer executable's capture outputs and explicit rounding
  # boundary, including for clean/deletion arms; unused captures stay on device.
  infer=jax.jit(lambda p,b,s:consumer.forward(model,p,b,rng,config,s,
      b['targets_segmentation']!=0,control,
      jnp.zeros(b['inputs'].shape+(config.emb_dim,),jnp.float32),11))
  meta=dict(base_config_class=base._BASE_CONFIG_CLASS,checkpoint=config.load_parameters_path,
      trainer_commit=base._TRAINER_COMMIT,diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      arms=names,layers=layers,batch_size=bs,source_positions='all valid',
      calculation_barrier=os.environ.get('BAM_CONSUMER_BARRIER')=='1',
      scan_layers=config.scan_layers,requested_sequences=len(cohort['inputs']))
  records=[]
  for offset in range(0,len(cohort['inputs']),bs):
    batch={k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    losses=[];tokens=[]
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      for scale in scales:
        result=infer(state.params,batch,jnp.asarray(scale))
        loss,token=jax.device_get(result[:2]);losses.append(loss);tokens.append(token)
    loss=np.stack(losses,1);token=np.stack(tokens,1)
    if not np.isfinite(token).all():raise ValueError('nonfinite loss')
    np.savez_compressed(output/f'batch_{offset:03d}.npz',loss=loss,token_loss=token,
        valid=np.asarray(batch['targets_segmentation'])!=0,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    records.append(loss);all_loss=np.concatenate(records).astype(np.float64)
    meta['completed_sequences']=len(all_loss);meta['elapsed_seconds']=time.perf_counter()-start
    meta['results']=[dict(arm=name,**stats(all_loss[:,i]-all_loss[:,0])) for i,name in enumerate(names)]
    meta['interactions']=[]
    for layer in layers:
      x,s,b=[names.index(f'L{layer}_{c}') for c in ['cross','self','both']]
      meta['interactions'].append(dict(layer=layer,
          both_minus_self_minus_cross=stats(all_loss[:,b]-all_loss[:,s]-all_loss[:,x]+all_loss[:,0])))
    (output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'NEIGHBORS_BATCH {len(all_loss)}/{len(cohort["inputs"])}',flush=True)
  if writer:writer.flush()
  print('NEIGHBORS_COMPLETE',flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv);base.train.validate_train_config(config);run(config)


if __name__=='__main__':app.run(main)

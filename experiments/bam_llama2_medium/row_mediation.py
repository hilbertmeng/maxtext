"""Read-only row-cross lifetime cuts and bidirectional mediator patching.

Reference activations stay on device for one microbatch only. All saved outputs
are sequence/token loss and scalar validation diagnostics, never activation vectors.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_attribution as base
import row_cross_sign as sign


class BamRowMediation(base.BamResidualAttribution):
  bam_readout_attribution = False


base.exp.BamRowMediation = BamRowMediation
CONTROL_NAMES = ['cancel_attention', 'cancel_mlp', 'std', 'full', 'M', 'mlp', 'v_self', 'v_cross']
REF_NAMES = ['value', 'std', 'full', 'M', 'mlp', 'post_attention']


def stack_capture(captured):
  grouped, scanned = {}, {}
  for path, value in traverse_util.flatten_dict(captured['mediation_capture']).items():
    name=path[-1].removeprefix('trace_')
    if name not in REF_NAMES: continue
    layer=base._layer_from_path(path)
    if layer is None:
      scanned.setdefault(name, []).append(base._unwrap(value))
    else: grouped.setdefault(layer, {})[name]=base._unwrap(value)
  if grouped:
    return {n:jnp.stack([grouped[l][n] for l in range(24)]) for n in REF_NAMES}
  if set(scanned)!=set(REF_NAMES) or any(len(v)!=1 for v in scanned.values()):
    raise ValueError(f'bad mediation captures: {[(k,len(v)) for k,v in scanned.items()]}')
  return {n:base._layer_axis_first(v[0],n) for n,v in scanned.items()}


def patch_tree(params, scales, refs, control, z, scanned):
  tree=traverse_util.flatten_dict(sign.controls(params, scales, scanned))
  paths=[p[:-1] for p in tree if p[-1]=='row_sign_scales']
  for attn in paths:
    layer=base._layer_from_path(attn)
    take=lambda x: x if scanned else x[layer]
    c=take(control)
    parent=attn[:-1]
    for name in ('value','std','full','M'):
      tree[attn+(f'med_{name}',)]=take(refs[name])
    for name,idx in [('std',2),('full',3),('M',4)]:
      tree[attn+(f'med_{name}_scale',)]=c[...,idx]
    tree[attn+('med_v_edges',)]=c[...,6:8]
    tree[parent+('med_mlp',)]=take(refs['mlp'])
    tree[parent+('med_mlp_scale',)]=c[...,5]
    tree[parent+('med_cancel',)]=c[...,:2]
    tree[parent+('med_z',)]=jnp.broadcast_to(z,(24,)+z.shape) if scanned else z
  return traverse_util.unflatten_dict(tree)


def apply(model, params, batch, rng, config, scales, refs=None, control=None, z=None, capture=False):
  variables=dict(params)
  variables['causal_ablation']=(sign.controls(params,scales,config.scan_layers) if refs is None
      else patch_tree(params,scales,refs,control,z,config.scan_layers))
  rng1,rng2=jax.random.split(rng)
  result=model.apply(variables,batch['inputs'],batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'],decoder_target_tokens=batch['targets'],
      enable_dropout=False,rngs={'dropout':rng1,'params':rng2},
      mutable=['residual_attribution','mediation_capture'] if capture else False)
  if capture:
    (token_loss,_,_),captured=result
    references=stack_capture(captured)
  else: token_loss,_,_=result
  loss=base._sequence_mean(token_loss,batch['targets_segmentation']!=0)
  return (loss,token_loss,references) if capture else (loss,token_loss)


def arms(source, phase, chosen=None):
  result=[]
  def add(name, corrupted=False, donor_corrupted=False, targets=(), fields=()):
    c=np.zeros((24,8),np.float32)
    for l in targets:
      for f in fields: c[l,CONTROL_NAMES.index(f)]=1
    result.append(dict(name=name,corrupted=corrupted,donor_corrupted=donor_corrupted,control=c))
  add('clean'); add('ablated',True)
  if phase=='coarse':
    # No final-output subtraction arm: each lifetime cut precedes a real consumer.
    add(f'cut_L{source}_attention',targets=[source],fields=['cancel_attention'])
    for l in range(source,23):
      add(f'cut_L{l}_mlp',targets=[l],fields=['cancel_mlp'])
    bands=[list(range(source,min(source+3,24))),list(range(source+3,min(source+7,24))),
           list(range(source+7,23)),[23]]
    groups=[x for x in bands if x]
  elif phase=='joint':
    # Exclude the source: restoring its deleted read would trivially undo the ablation.
    groups=[list(range(source+1,end+1)) for end in (source+1,source+2,source+4,23) if end<24]
  else:
    groups=[[x] for x in chosen]
  for group in groups:
    label=f'L{group[0]}' if len(group)==1 else f'L{group[0]}-{group[-1]}'
    field_sets=['std','mlp','full','M','v_self','v_cross','v_both']
    if phase=='joint':
      field_sets+=['v_cross+mlp','std+mlp','std+full','std+M','std+mlp+full+M']
    for field in field_sets:
      fields=['v_self','v_cross'] if field=='v_both' else field.split('+')
      add(f'rescue_{label}_{field}',True,False,group,fields)
      add(f'block_{label}_{field}',False,True,group,fields)
  return result


def aggregate(output, meta):
  files=sorted(output.glob('batch_*.npz'))
  if not files:return
  arrays=[]
  for p in files:
    with np.load(p) as x: arrays.append(x['loss'])
  loss=np.concatenate(arrays).astype(float)
  meta['completed_sequences']=len(loss); meta['results']=[]
  for i,arm in enumerate(meta['arms']):
    d=loss[:,i]-loss[:,0]
    effect=loss[:,i]-loss[:,1] if arm['corrupted'] else d
    meta['results'].append(dict(arm=arm['name'],delta=float(d.mean()),
        ci95=float(1.96*d.std(ddof=1)/len(d)**.5),harmed=int((d>0).sum()),
        effect_vs_recipient=float(effect.mean())))
  (output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')


def run(config):
  assert not getattr(config,'bam_mlp_write',False)
  source=int(os.environ.get('BAM_MEDIATION_SOURCE','11'))
  phase=os.environ.get('BAM_MEDIATION_PHASE','coarse')
  reference_mode=os.environ.get('BAM_MEDIATION_REFERENCE','opposite')
  if reference_mode not in ('opposite','self'):
    raise ValueError(f'unknown reference mode: {reference_mode}')
  chosen=[int(x) for x in os.environ.get('BAM_MEDIATION_LAYERS','11,12,13,14,15,16,17').split(',')]
  matrix=arms(source,phase,chosen)
  selector=os.environ.get('BAM_MEDIATION_ARM_FILTER')
  if selector:
    matrix=[a for a in matrix if a['name'] in ('clean','ablated') or re.search(selector,a['name'])]
  output=Path(os.environ['BAM_MEDIATION_OUTPUT']); output.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  n=int(os.environ.get('BAM_MEDIATION_N','128'))
  bs=int(os.environ.get('BAM_RESIDUAL_ATTR_BATCH_SIZE','1'))
  with np.load(path) as data:
    cohort={k:np.asarray(data[k])[:n] for k in ['inputs','targets','inputs_position',
        'inputs_segmentation','targets_segmentation','sequence_hashes']}
  start=time.perf_counter()
  rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
  iterator,_=base.create_data_iterator(config,mesh)
  state,_,_,_=base.max_utils.setup_training_state(model,iterator,tx,config,rng,mesh,manager)
  capture=jax.jit(lambda p,b,r,s:apply(model,p,b,r,config,s,capture=True))
  infer=jax.jit(lambda p,b,r,s,ref,c,z:apply(model,p,b,r,config,s,ref,c,z))
  clean_s=jnp.ones((24,2),jnp.float32); corrupt_s=clean_s.at[source].set(0)
  meta=dict(base_config_class=base._BASE_CONFIG_CLASS,checkpoint=config.load_parameters_path,
      diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],trainer_commit=base._TRAINER_COMMIT,
      cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_layer=source,
      phase=phase,reference_mode=reference_mode,scan_layers=config.scan_layers,batch_size=bs,requested_sequences=n,
      controls=CONTROL_NAMES,arms=[dict(a,control=a['control'].tolist()) for a in matrix],
      setup_seconds=time.perf_counter()-start)
  print('MEDIATION_RESTORED '+json.dumps({k:v for k,v in meta.items() if k!='arms'}),flush=True)
  for offset in range(0,n,bs):
    target=output/f'batch_{offset:03d}.npz'
    if target.exists():continue
    batch={k:jnp.asarray(v[offset:offset+bs]) for k,v in cohort.items() if k!='sequence_hashes'}
    with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
      clean_loss,_,clean_ref=capture(state.params,batch,rng,clean_s)
      corrupt_loss,_,corrupt_ref=capture(state.params,batch,rng,corrupt_s)
      z=(clean_ref['post_attention'][source].astype(jnp.float32)-
         corrupt_ref['post_attention'][source].astype(jnp.float32))
      # In this model row head coordinates do not feed the same-layer M write.
      source_m_error=float(jnp.max(abs(clean_ref['M'][source]-corrupt_ref['M'][source])))
      if source_m_error!=0:raise ValueError(f'source M unexpectedly changed: {source_m_error}')
      losses=[]; tokens=[]
      for a in matrix:
        donor_corrupted=a['corrupted'] if reference_mode=='self' else a['donor_corrupted']
        loss,tok=infer(state.params,batch,rng,corrupt_s if a['corrupted'] else clean_s,
            corrupt_ref if donor_corrupted else clean_ref,jnp.asarray(a['control']),z)
        loss,tok=jax.device_get((loss,tok)); losses.append(loss); tokens.append(tok)
      loss=np.stack(losses,axis=1)
      validation=np.stack([loss[:,0]-np.asarray(clean_loss),loss[:,1]-np.asarray(corrupt_loss)],axis=1)
    if not np.isfinite(loss).all():raise ValueError('nonfinite mediation loss')
    np.savez_compressed(target,loss=loss,token_loss=np.stack(tokens,axis=1),
        validation_loss=validation,valid=np.asarray(batch['targets_segmentation'])!=0,
        sequence_hashes=cohort['sequence_hashes'][offset:offset+bs])
    meta['elapsed_seconds']=time.perf_counter()-start; aggregate(output,meta)
    print(f'MEDIATION_BATCH {offset+bs}/{n} null_drift={validation.mean(0).tolist()} '
          f'ablation={float((loss[:,1]-loss[:,0]).mean()):.6f}',flush=True)
  if writer:writer.flush()
  print('MEDIATION_COMPLETE',flush=True)


def main(argv):
  config=base.pyconfig.initialize(argv); base.train.validate_train_config(config); run(config)


if __name__=='__main__':app.run(main)

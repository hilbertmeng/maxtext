"""Paired Q/K/V/O row-read knockout, dose and within-LLF interventions.

Linen interception at row-read outputs, no production-model edits. Positive loss
change means the native row path is useful in the intervened context. Four-path
factorial experiments permit exact Shapley allocation, not additive layer claims.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText'))
from absl import app
from flax import linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
import train
from layers import attentions

BASE='BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow'
KEYS=('inputs','targets','inputs_position','inputs_segmentation','targets_segmentation')
PATHS=('Q','K','V','O')


class RowContributionProbe(getattr(exp,BASE)):
  only_eval=True
  per_device_batch_size=1
  eval_per_device_batch_size=1
  load_parameters_path=f'gs://newproject-1-llm_projects_us-east5/log/{BASE}/checkpoints/13500/items'
  record_internal_nn_metrics=False
  record_training_health_metrics=False
  bam_record_fetched_read_amplitude_metrics=False
  bam_record_fetched_read_health_metrics=False
  bam_record_fetch_route_metrics=False
  bam_record_local_qk_amplitude_metrics=False
  bam_record_local_routing_metrics=False


exp.RowContributionProbe=RowContributionProbe


def scale_row(read,scale,bam_k=32):
  assert read.shape[-1]==64
  return jnp.concatenate((read[...,:bam_k], read[...,bam_k:]*scale.astype(read.dtype)),axis=-1)


@contextlib.contextmanager
def row_interventions(scales):
  """scales[layer,path], at post-gate/post-mix row output, before QK RoPE or injection."""
  layers=[]
  def intercept(next_fun,args,kw,ctx):
    if isinstance(ctx.module,attentions.BamAttention) and ctx.method_name=='__call__':
      assert kw.get('layer_index') is not None
      layers.append(kw['layer_index'])
      try:
        return next_fun(*args,**kw)
      finally:
        layers.pop()
    if isinstance(ctx.module,attentions.BamAttention) and ctx.method_name=='_read_local':
      assert layers
      name=args[0] if args else kw['name']
      path={'q':0,'k':1,'v':2}[name]
      return scale_row(next_fun(*args,**kw),scales[layers[-1],path],ctx.module.bam_k)
    if isinstance(ctx.module,attentions.BamAttention) and ctx.method_name=='_read_fetched_m':
      assert layers and not kw.get('ungated',False), 'Independent BAlignedRow only'
      read,gate=next_fun(*args,**kw)
      # Independent BAlignedRow has 16 fetched heads == 16 MHA heads. Packed tail
      # begins at32: eight compressed row coordinates followed by24 zeros.
      assert ctx.module._fetched_read_num_heads==ctx.module.num_query_heads
      return scale_row(read,scales[layers[-1],3],ctx.module.bam_k),gate
    return next_fun(*args,**kw)
  with nn.intercept_methods(intercept):
    yield


def forward(model,params,batch,rng,scales=None):
  cm=row_interventions(scales) if scales is not None else contextlib.nullcontext()
  with cm:
    output,_=model.apply(
        params,batch['inputs'],batch['inputs_position'],
        decoder_segment_ids=batch['inputs_segmentation'],
        decoder_target_mask=batch['targets_segmentation'],decoder_target_tokens=batch['targets'],
        enable_dropout=False,rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
  mask=batch['targets_segmentation']!=0
  return jnp.sum(output[0]*mask,-1)/jnp.maximum(mask.sum(-1),1)


def scenarios():
  result=[]
  def add(name,kind,selected,scale=0.,**meta):
    a=np.ones((24,4),np.float32)
    for layer,path in selected:a[layer,path]=scale
    result.append(dict(name=name,kind=kind,scale=scale,**meta,scales=a.tolist()))
  for bits in range(16):
    add(f'coalition_{bits:02d}','coalition',[(l,p) for l in range(24) for p in range(4) if bits&(1<<p)],coalition=bits)
  for p,name in enumerate(PATHS):
    add(f'half_{name}','dose',[(l,p) for l in range(24)],.5,path=name)
  add('half_all','dose',[(l,p) for l in range(24) for p in range(4)],.5,path='all')
  for l in range(24):
    for p,name in enumerate(PATHS):
      if p==2 and l%3==2:continue
      add(f'layer_{l:02d}_{name}','layer',[(l,p)],layer=l,path=name)
  for unit in range(8):
    for p,name in enumerate(PATHS):
      add(f'unit_{unit}_{name}','unit',[(l,p) for l in range(3*unit,3*unit+3)],unit=unit,path=name)
  assert len(result)==141
  return result


def digest_cohort(cohort):
  return [{k:hashlib.sha256(cohort[k][i].tobytes()).hexdigest() for k in KEYS} for i in range(len(cohort['inputs']))]


def run(config):
  out=Path(os.environ.get('ROW_OUTPUT','/tmp/llf-row-contribution'));out.mkdir(parents=True,exist_ok=True)
  cohort_path=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
  with np.load(cohort_path) as source: cohort={k:np.asarray(source[k]) for k in KEYS}
  hashes=digest_cohort(cohort)
  all_scenarios=scenarios()
  stage=os.environ.get('ROW_STAGE','all')
  use=[s for s in all_scenarios if stage=='all' or (stage=='groups' and s['kind'] in ('coalition','dose')) or (stage=='layers' and s['kind'] in ('layer','unit'))]
  assert use and stage in ('all','groups','layers')
  # Groups are cheap enough to expand independently after the first full32.
  start=int(os.environ.get('ROW_START','0'));stop=int(os.environ.get('ROW_STOP','32'))
  assert 0<=start<stop<=min(64,len(hashes))
  batch_size=int(os.environ.get('ROW_VARIANT_BATCH','8'))
  scales=np.asarray([s['scales'] for s in use],np.float32)
  metadata=dict(model=BASE,checkpoint=config.load_parameters_path,
      training_commit='77401da6f83a5aa6ddd61994e028c3c694221518',
      runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      cohort_file_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      sequence_hashes=hashes,stage=stage,start=start,stop=stop,
      overrides={k:getattr(config,k) for k in ('only_eval','record_internal_nn_metrics','record_training_health_metrics','bam_record_fetched_read_health_metrics','bam_record_fetched_read_amplitude_metrics','bam_record_fetch_route_metrics','bam_record_local_routing_metrics','bam_record_local_qk_amplitude_metrics')},
      intervention='Post-normalization/gate/mixing row output only; all token positions; downstream forward runs normally',
      interpretation='Frozen-network causal necessity, not retraining benefit or deployable speed measurement')
  tag=f'{stage}_{start:03d}_{stop:03d}'
  (out/f'{tag}_metadata.json').write_text(json.dumps(metadata,indent=2))
  (out/f'{stage}_scenarios.json').write_text(json.dumps(use,indent=2))
  rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
  cursor=SimpleNamespace(meta_dict={'checkpoint_step':None})
  state,_,_,_=max_utils.setup_training_state(model,cursor,tx,config,rng,mesh,manager)
  one=jax.jit(lambda p,b,s:forward(model,p,b,rng,s))
  many=jax.jit(jax.vmap(lambda p,b,s:forward(model,p,b,rng,s),in_axes=(None,None,0)))
  ordinary=jax.jit(lambda p,b:forward(model,p,b,rng))
  first=True
  for index in range(start,stop):
    file=out/f'{stage}_{index:03d}.npz'
    if file.exists():
      with np.load(file) as previous:
        assert str(previous['sequence_hash'])==hashes[index]['inputs']
      continue
    begun=time.perf_counter();batch={k:jnp.asarray(cohort[k][index:index+1]) for k in KEYS}
    with mesh,partitioning.axis_rules(config.logical_axis_rules):
      native=np.asarray(one(state.params,batch,jnp.ones((24,4),jnp.float32)))
      if first:
        reference=np.asarray(ordinary(state.params,batch))
        np.testing.assert_allclose(native,reference,atol=1e-6,rtol=0)
        control=np.ones((24,4),np.float32);control[0,:]=0.;control[2::3,2]=0.
        np.testing.assert_allclose(np.asarray(one(state.params,batch,jnp.asarray(control))),native,atol=1e-6,rtol=0)
        if batch_size>1:
          stacked=np.asarray(many(state.params,batch,jnp.ones((batch_size,24,4),jnp.float32)))
          if not np.allclose(stacked,native,atol=1e-6,rtol=0):
            print('VMAP_BASELINE_MISMATCH fallback variant_batch=1',float(np.max(np.abs(stacked-native))),flush=True)
            batch_size=1
        # Verify batching on nontrivial knockouts as well as the all-on control.
        if batch_size>1:
          check=scales[:min(batch_size,len(scales))]
          check=np.concatenate((check,np.ones((batch_size-len(check),24,4),np.float32)))
          batched=np.asarray(many(state.params,batch,jnp.asarray(check)))
          for i in range(min(3,len(check))):
            serial=np.asarray(one(state.params,batch,jnp.asarray(check[i])))
            np.testing.assert_allclose(batched[i],serial,atol=1e-6,rtol=0)
          begun_benchmark=time.perf_counter()
          for _ in range(3):np.asarray(many(state.params,batch,jnp.asarray(check)))
          metadata['benchmark_variants_per_second']=3*batch_size/(time.perf_counter()-begun_benchmark)
        print(f'FIRST_STEP ROW_NOOP_OK sample={index} loss={native} variant_batch={batch_size} variants_per_second={metadata.get("benchmark_variants_per_second")}',flush=True)
        metadata['effective_variant_batch']=batch_size
        (out/f'{tag}_metadata.json').write_text(json.dumps(metadata,indent=2))
        first=False
      values=[]
      for begin in range(0,len(use),batch_size):
        part=scales[begin:begin+batch_size]
        if batch_size==1: prediction=np.asarray(one(state.params,batch,jnp.asarray(part[0])))[None]
        else:
          padded=np.concatenate((part,np.ones((batch_size-len(part),24,4),np.float32)))
          prediction=np.asarray(many(state.params,batch,jnp.asarray(padded)))[:len(part)]
        values.extend(prediction)
      losses=np.stack(values);assert np.isfinite(losses).all()
    temporary=out/f'.pending_{stage}_{index:03d}.npz'
    np.savez_compressed(temporary,loss=losses,baseline=native,gap=losses-native,
        sequence_hash=hashes[index]['inputs'],tokens=int(np.sum(cohort['targets_segmentation'][index]!=0)))
    temporary.replace(file)
    print(f'ROW_DONE stage={stage} sample={index} scenarios={len(use)} seconds={time.perf_counter()-begun:.2f}',flush=True)
  if writer:writer.flush()
  print(f'ROW_STAGE_DONE {tag}',flush=True)


if __name__=='__main__':app.run(lambda argv:run(pyconfig.initialize(argv)))

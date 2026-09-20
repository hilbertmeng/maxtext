"""Raw standard V half scaling before BAM injection; immutable training source."""
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

BASE='BamMediumIndependentLLFMLPPerLayerColOnly'
KEYS=('inputs','targets','inputs_position','inputs_segmentation','targets_segmentation')
PATHS=('Q','K','V','O')


class StdVProbe(getattr(exp,BASE)):
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


exp.StdVProbe=StdVProbe


def forward(model,params,batch,rng,scales=None):
  cm=interventions(scales) if scales is not None else contextlib.nullcontext()
  with cm:
    output,_=model.apply(
        params,batch['inputs'],batch['inputs_position'],
        decoder_segment_ids=batch['inputs_segmentation'],
        decoder_target_mask=batch['targets_segmentation'],decoder_target_tokens=batch['targets'],
        enable_dropout=False,rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
  mask=batch['targets_segmentation']!=0
  return jnp.sum(output[0]*mask,-1)/jnp.maximum(mask.sum(-1),1)


SHAPE=(24,3) # standard front, standard tail, BAM V
LEVELS=(0.,.2,.5,.8,1.)
TRAINING='2ca927c1a76011a247303efccbb3afd5b868ffd2'

def scale_raw(v,s):
  assert v.shape[-1]==64
  return jnp.concatenate((v[...,:32]*s[0].astype(v.dtype),v[...,32:]*s[1].astype(v.dtype)),axis=-1)

@contextlib.contextmanager
def interventions(scales):
  layers=[]
  def intercept(next_fun,args,kw,ctx):
    module=ctx.module
    if isinstance(module,attentions.BamAttention) and ctx.method_name=='__call__':
      assert kw.get('layer_index') is not None
      layers.append((kw['layer_index'],bool(module._local_o)))
      try:return next_fun(*args,**kw)
      finally:layers.pop()
    if isinstance(module,attentions.BamAttention) and layers and layers[-1][1]:
      l=layers[-1][0]
      if ctx.method_name=='kv_projection':
        name=kw.get('proj_name',args[1] if len(args)>1 else None)
        v=next_fun(*args,**kw)
        return scale_raw(v,scales[l]) if name=='value' else v
      if ctx.method_name=='qkv_projection':
        q,k,v=next_fun(*args,**kw);return q,k,scale_raw(v,scales[l])
      if ctx.method_name=='_read_local':
        name=args[0] if args else kw['name'];v=next_fun(*args,**kw)
        return v*scales[l,2].astype(v.dtype) if name=='v' else v
    return next_fun(*args,**kw)
  with nn.intercept_methods(intercept):yield

def scenarios():
  local=[l for l in range(24) if l%3!=2]
  scopes={f'L{l:02d}':[l] for l in local}
  scopes.update(all_L=local,ordinary_L=[l for l in local if l>1],early_ordinary_L=[l for l in local if 1<l<12],late_L=[l for l in local if l>=12])
  out=[]
  for scope,ls in scopes.items():
    for bam in [1.,0.]:
      for a in LEVELS:
        for b in LEVELS:
          mask=np.ones(SHAPE,np.float32);mask[ls]=[a,b,bam]
          out.append(dict(name=f'{scope}/bam{bam:g}/a{a:g}_b{b:g}',scope=scope,layers=ls,alpha=a,beta=b,bam=bam,scales=mask.tolist()))
  return out

def digest_cohort(cohort):
  return [{k:hashlib.sha256(np.ascontiguousarray(v[i]).tobytes()).hexdigest() for k,v in cohort.items()} for i in range(len(cohort['inputs']))]

def run(config):
  out=Path(os.environ['STD_V_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
  with np.load(path) as f:cohort={k:np.asarray(f[k]) for k in KEYS}
  hashes=digest_cohort(cohort);ss=scenarios();unique=[];mapping=[];seen={}
  for s in ss:
    a=np.asarray(s['scales'],np.float32);a[0,2]=1. # M at L0 is zero; raw V at L0 is NOT zero
    key=a.tobytes()
    if key not in seen:seen[key]=len(unique);unique.append(a)
    mapping.append(seen[key])
  start=int(os.environ.get('STD_V_START','0'));stop=int(os.environ.get('STD_V_STOP','64'))
  assert 0<=start<stop<=64
  assert config.bam_k==32 and config.head_dim==64 and config.num_decoder_layers==24
  assert config.only_eval and config.bam_prune_all_row_reads and not config.bam_local_v_share_output_coordinates
  meta=dict(model=BASE,checkpoint=config.load_parameters_path,training_commit=TRAINING,runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sequence_hashes=hashes[:64],start=start,stop=stop,scenarios=len(ss),unique_forwards=len(unique),mapping=mapping,shape_config={k:getattr(config,k) for k in ['emb_dim','num_query_heads','head_dim','bam_k','bam_local_v_rank','fused_qkv','scan_layers']},method='raw V pre-injection output scaling; BAM V separate; full downstream recompute')
  (out/'scenarios.json').write_text(json.dumps(ss));save=lambda:(out/'metadata.json').write_text(json.dumps(meta,indent=2));save()
  rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
  state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
  one=jax.jit(lambda p,b,s:forward(model,p,b,rng,s));ordinary=jax.jit(lambda p,b:forward(model,p,b,rng));scales=jnp.asarray(np.stack(unique));mode=None
  def dispatch(batch,part,mode):
    if mode=='async':return np.stack(jax.device_get([one(state.params,batch,s) for s in part]))
    return np.stack([np.asarray(one(state.params,batch,s)) for s in part])
  with mesh,partitioning.axis_rules(config.logical_axis_rules):
    for i in range(start,stop):
      dest=out/f'dose_{i:03d}.npz'
      if dest.exists():continue
      begun=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
      native=np.asarray(one(state.params,batch,jnp.ones(SHAPE,np.float32)))
      orig=np.asarray(ordinary(state.params,batch));np.testing.assert_allclose(native,orig,atol=1e-6,rtol=0)
      if mode is None:
        noop=np.ones(SHAPE,np.float32);noop[2::3]=0.;noop[0,2]=0.
        np.testing.assert_array_equal(np.asarray(one(state.params,batch,jnp.asarray(noop))),native)
        check=scales[:16];serial=dispatch(batch,check,'scalar');timings={}
        for m in ['scalar','async']:
          vals=dispatch(batch,check,m);np.testing.assert_array_equal(vals,serial)
          t=time.perf_counter();dispatch(batch,check,m);timings[m]=len(check)/(time.perf_counter()-t)
        mode=max(timings,key=timings.get);meta.update(dispatch=mode,benchmarks=timings,native_max_error=float(np.max(np.abs(native-orig))));save()
        print('FIRST_STEP STD_V_NATIVE_OK',native,len(unique),mode,timings,flush=True)
      vals=[]
      for j in range(0,len(scales),16):vals.extend(dispatch(batch,scales[j:j+16],mode))
      loss=np.stack(vals)[mapping];assert np.isfinite(loss).all()
      np.testing.assert_array_equal(loss[0],np.asarray(one(state.params,batch,jnp.asarray(ss[0]['scales'],np.float32))))
      temp=out/f'.pending_dose_{i:03d}.npz';np.savez_compressed(temp,loss=loss,baseline=native,ordinary=orig,sequence_hash=hashes[i]['inputs'],tokens=int(np.sum(cohort['targets_segmentation'][i]!=0)));temp.replace(dest)
      print(f'STD_V_DONE sample={i} seconds={time.perf_counter()-begun:.2f}',flush=True)
  if writer:writer.flush()
  print('STD_V_STAGE_DONE',start,stop,flush=True)

def run_grad(config):
  out=Path(os.environ['STD_V_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
  with np.load(path) as f:cohort={k:np.asarray(f[k]) for k in KEYS}
  hashes=digest_cohort(cohort);start=int(os.environ.get('STD_V_START','0'));stop=int(os.environ.get('STD_V_STOP','32'))
  assert config.bam_k==32 and config.head_dim==64 and config.num_decoder_layers==24
  assert config.only_eval and config.bam_prune_all_row_reads and not config.bam_local_v_share_output_coordinates
  meta=dict(model=BASE,checkpoint=config.load_parameters_path,training_commit=TRAINING,runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sequence_hashes=hashes[:stop],start=start,stop=stop,shape_config={k:getattr(config,k) for k in ['emb_dim','num_query_heads','head_dim','bam_k','bam_local_v_rank','fused_qkv','scan_layers']},method='native alpha=beta=bam=1; per-sequence loss gradient w.r.t. per-layer scaling of raw standard V halves; BAM derivative auxiliary',channels=['raw_front','raw_tail','bam_v'],unit='nats/token per unit scale')
  (out/'metadata.json').write_text(json.dumps(meta,indent=2))
  rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
  state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
  f=lambda p,b,s:forward(model,p,b,rng,s).mean()
  grad=jax.jit(jax.value_and_grad(f,argnums=2));one=jax.jit(f);ordinary=jax.jit(lambda p,b:forward(model,p,b,rng).mean());ones=jnp.ones(SHAPE,jnp.float32)
  with mesh,partitioning.axis_rules(config.logical_axis_rules):
    for i in range(start,stop):
      begun=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
      value,g=grad(state.params,batch,ones);value,g=jax.device_get((value,g));native=float(ordinary(state.params,batch))
      np.testing.assert_allclose(value,native,atol=1e-6,rtol=0);assert np.isfinite(g).all();np.testing.assert_array_equal(g[2::3],0.)
      assert np.any(g[:,0]!=0) and np.any(g[:,1]!=0), 'raw V hook must be reached'
      # Small finite-difference probes are sanity checks, not an integrated-gradient analysis.
      checks=[]
      if i<start+4:
        for channel in [0,1]:
          mask=np.zeros(SHAPE,np.float32);mask[[l for l in range(3,24) if l%3!=2],channel]=1.
          plus=float(one(state.params,batch,ones+jnp.asarray(mask)*.05));minus=float(one(state.params,batch,ones-jnp.asarray(mask)*.05))
          checks.append([channel,float((g*mask).sum()),(plus-minus)/.1,plus,minus])
      np.savez_compressed(out/f'grad_{i:03d}.npz',loss=value,ordinary=native,gradient=g,sequence_hash=hashes[i]['inputs'],finite_difference=np.asarray(checks),tokens=int(np.sum(cohort['targets_segmentation'][i]!=0)))
      if i==start:print('FIRST_STEP STD_V_GRAD_NATIVE_OK',value,'gradient_norm',np.linalg.norm(g),flush=True)
      print('STD_V_GRAD_DONE',i,'seconds',time.perf_counter()-begun,flush=True)
  if writer:writer.flush()
  print('STD_V_GRAD_STAGE_DONE',start,stop,flush=True)

if __name__=='__main__':app.run(lambda argv:(run_grad if os.environ.get('STD_V_MODE')=='grad' else run)(pyconfig.initialize(argv)))

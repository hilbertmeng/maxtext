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


def forward(model,params,batch,rng,scales=None,native_gradient=False):
  cm=interventions(scales,native_gradient=native_gradient) if scales is not None else contextlib.nullcontext()
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

@jax.custom_jvp
def native_raw_scale(v,s):
  # Native-point identity; tangent is that of actual raw-V scaling at scale=1.
  return v

@native_raw_scale.defjvp
def native_raw_scale_jvp(primals,tangents):
  v,s=primals;dv,ds=tangents
  return v,dv+scale_raw(v,ds)

@jax.custom_jvp
def native_bam_scale(v,s):return v

@native_bam_scale.defjvp
def native_bam_scale_jvp(primals,tangents):
  v,s=primals;dv,ds=tangents
  return v,dv+v*ds.astype(v.dtype)

@contextlib.contextmanager
def interventions(scales,native_gradient=False):
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
        return (native_raw_scale(v,scales[l]) if native_gradient else scale_raw(v,scales[l])) if name=='value' else v
      if ctx.method_name=='qkv_projection':
        q,k,v=next_fun(*args,**kw);return q,k,(native_raw_scale(v,scales[l]) if native_gradient else scale_raw(v,scales[l]))
      if ctx.method_name=='_read_local':
        name=args[0] if args else kw['name'];v=next_fun(*args,**kw)
        return (native_bam_scale(v,scales[l,2]) if native_gradient else v*scales[l,2].astype(v.dtype)) if name=='v' else v
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
  import re
  from flax.traverse_util import flatten_dict,unflatten_dict
  assert not config.fused_qkv and not config.qkv_bias
  layout_info={}
  def capture(next_fun,args,kw,ctx):
    m=ctx.module
    if isinstance(m,attentions.BamAttention) and ctx.method_name=='_local_inputs' and 'v' in m._local_arms:
      arms=list(m._local_arms.values());layout,width=attentions._packed_local_layout(arms,m._share_qk_basis)
      arm=m._local_arms['v'];basis=layout[[x.name for x in arms].index('v')][0]
      info=dict(start=basis.start,stop=basis.stop,width=width,bias=arm.prefix+'_bias',packed=getattr(config,'bam_local_packed_parameter_name','W_local_packed'))
      assert not layout_info or layout_info==info
      layout_info.update(info)
    return next_fun(*args,**kw)
  # Capture static arm layout only. No tensor operation is intercepted or changed.
  def f(p,b):
    with nn.intercept_methods(capture):return forward(model,p,b,rng)
  ordinary=jax.jit(f);flat=flatten_dict(state.params);mode=None
  with mesh,partitioning.axis_rules(config.logical_axis_rules):
    initial_batch={k:jnp.asarray(v[start:start+1]) for k,v in cohort.items()}
    np.asarray(ordinary(state.params,initial_batch));assert layout_info
    raw=[];packed=[];bias=[]
    for key,v in flat.items():
      m=re.search(r'(?:sub|local|fetch)_([012])','/'.join(key))
      if m is None or int(m[1])==2:continue
      block=int(m[1])
      if key[-2:]==('value','kernel'):
        assert v.shape==(1024,8,16,64);raw.append((key,block))
      elif key[-2:]==(layout_info['packed'],'kernel'):
        assert v.shape==(1024,8,layout_info['width']);packed.append((key,block))
      elif key[-1]==layout_info['bias']:
        assert v.shape[1]==8;bias.append((key,block))
    assert len(raw)==len(packed)==len(bias)==2,(raw,packed,bias)
    selected=raw+packed+bias
    base=tuple(flat[k] for k,_ in selected)
    def modify(values,scales):
      result=[]
      for n,((key,block),v) in enumerate(zip(selected,values)):
        ls=jnp.arange(8)*3+block
        if n<2:
          factors=jnp.repeat(scales[ls,:2],32,axis=-1)[None,:,None,:]
        elif n<4:
          columns=jnp.arange(v.shape[-1]);hit=(columns>=layout_info['start'])&(columns<layout_info['stop'])
          factors=jnp.where(hit[None,None,:],scales[ls,2][None,:,None],1.)
        else:
          shape=[1]*v.ndim;shape[1]=8;factors=scales[ls,2].reshape(shape)
        result.append(v*factors.astype(v.dtype))
      return tuple(result)
    mutate=jax.jit(modify)
    def params_for(scale):
      changed=dict(flat);changed.update({k:v for (k,_),v in zip(selected,mutate(base,scale))})
      return unflatten_dict(changed)
    # Multiplication by one must preserve every modified parameter exactly.
    for old,new in zip(base,mutate(base,jnp.ones(SHAPE))):np.testing.assert_array_equal(np.asarray(old),np.asarray(new))
    meta.update(method='external W_V half scaling and BAM V key projection/bias zeroing; one unmodified ordinary compiled forward; full downstream recompute',bam_v_layout=layout_info,parameter_paths=['/'.join(k) for k,_ in selected])
    scales=jnp.asarray(np.stack(unique))
    def dispatch(batch,part,mode):
      if mode=='async':return np.stack(jax.device_get([ordinary(params_for(s),batch) for s in part]))
      return np.stack([np.asarray(ordinary(params_for(s),batch)) for s in part])
    for i in range(start,stop):
      dest=out/f'dose_{i:03d}.npz'
      if dest.exists():continue
      begun=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
      orig=np.asarray(ordinary(state.params,batch));native=np.asarray(ordinary(params_for(jnp.ones(SHAPE)),batch))
      np.testing.assert_array_equal(native,orig)
      if mode is None:
        noop=np.ones(SHAPE,np.float32);noop[2::3]=0.;noop[0,2]=0.
        np.testing.assert_array_equal(np.asarray(ordinary(params_for(jnp.asarray(noop)),batch)),native)
        check=scales[:8];serial=dispatch(batch,check,'scalar');timings={}
        for m in ['scalar','async']:
          vals=dispatch(batch,check,m);np.testing.assert_array_equal(vals,serial)
          t=time.perf_counter();dispatch(batch,check,m);timings[m]=len(check)/(time.perf_counter()-t)
        mode=max(timings,key=timings.get);meta.update(dispatch=mode,benchmarks=timings,native_max_error=float(np.max(np.abs(native-orig))));save()
        print('FIRST_STEP STD_V_NATIVE_OK',native,len(unique),mode,timings,flush=True)
      vals=[]
      for j in range(0,len(scales),8):vals.extend(dispatch(batch,scales[j:j+8],mode))
      loss=np.stack(vals)[mapping];assert np.isfinite(loss).all()
      temp=out/f'.pending_dose_{i:03d}.npz';np.savez_compressed(temp,loss=loss,baseline=native,ordinary=orig,sequence_hash=hashes[i]['inputs'],tokens=int(np.sum(cohort['targets_segmentation'][i]!=0)));temp.replace(dest)
      print(f'STD_V_DONE sample={i} seconds={time.perf_counter()-begun:.2f}',flush=True)
  if writer:writer.flush()
  print('STD_V_STAGE_DONE',start,stop,flush=True)

def run_grad(config):
  import re
  from flax.traverse_util import flatten_dict,unflatten_dict
  out=Path(os.environ['STD_V_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
  path=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
  with np.load(path) as f:cohort={k:np.asarray(f[k]) for k in KEYS}
  hashes=digest_cohort(cohort);start=int(os.environ.get('STD_V_START','0'));stop=int(os.environ.get('STD_V_STOP','32'))
  assert config.bam_k==32 and config.head_dim==64 and config.num_decoder_layers==24
  assert config.only_eval and config.bam_prune_all_row_reads and not config.fused_qkv and not config.qkv_bias
  rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
  state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
  flat=flatten_dict(state.params);selected=[]
  for key,v in flat.items():
    if key[-2:]==('value','kernel'):
      name='/'.join(key);m=re.search(r'(?:sub|local|fetch)_([012])',name)
      print('RAW_V_PARAMETER',name,v.shape,flush=True)
      assert m is not None and v.shape==(1024,8,16,64),(name,v.shape)
      if int(m[1])!=2:selected.append((key,int(m[1])))
  assert len(selected)==2,selected
  meta=dict(model=BASE,checkpoint=config.load_parameters_path,training_commit=TRAINING,runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sequence_hashes=hashes[:stop],start=start,stop=stop,shape_config={k:getattr(config,k) for k in ['emb_dim','num_query_heads','head_dim','bam_k','bam_local_v_rank','fused_qkv','scan_layers']},method='unmodified model parameter gradient; sum W_V*dL/dW_V per original V half; no activation scaling or identity JVP in gradient forward',channels=['raw_front','raw_tail','unused_zero'],parameter_paths=['/'.join(k) for k,_ in selected],unit='nats/token per unit scale')
  (out/'metadata.json').write_text(json.dumps(meta,indent=2))
  f=lambda p,b:forward(model,p,b,rng).mean()
  def pull(p,b):
    value,dp=jax.value_and_grad(f)(p,b);fp=flatten_dict(p);fg=flatten_dict(dp);g=jnp.zeros(SHAPE,jnp.float32)
    for key,block in selected:
      prod=fp[key].astype(jnp.float32)*fg[key].astype(jnp.float32)
      for half in [0,1]:g=g.at[jnp.arange(8)*3+block,half].set(prod[...,half*32:(half+1)*32].sum(axis=(0,2,3)))
    return value,g
  grad=jax.jit(pull);ordinary=jax.jit(f)
  def perturbed(half,factor):
    copy=dict(flat)
    for key,block in selected:
      mask=np.ones((1,8,1,64),np.float32)
      for u in range(8):
        if 3*u+block>1:mask[:,u,:,half*32:(half+1)*32]=factor
      copy[key]=flat[key]*jnp.asarray(mask,dtype=flat[key].dtype)
    return unflatten_dict(copy)
  with mesh,partitioning.axis_rules(config.logical_axis_rules):
    variants=[perturbed(c,1+sign*.0625) for c in [0,1] for sign in [1,-1]]
    for i in range(start,stop):
      begun=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
      value,g=jax.device_get(grad(state.params,batch));native=float(ordinary(state.params,batch));assert np.isfinite(g).all();np.testing.assert_array_equal(g[2::3],0.)
      print('PARAM_GRAD_FIDELITY',i,'ordinary',native,'ad',value,'difference',float(value)-native,flush=True)
      checks=[]
      for c in [0,1]:
        plus=float(ordinary(variants[2*c],batch));minus=float(ordinary(variants[2*c+1],batch));ad=float(g[[l for l in range(3,24) if l%3!=2],c].sum());checks.append([c,ad,(plus-minus)/.125,plus,minus])
      np.savez_compressed(out/f'grad_{i:03d}.npz',loss=value,ordinary=native,gradient=g,sequence_hash=hashes[i]['inputs'],finite_difference=np.asarray(checks),tokens=int(np.sum(cohort['targets_segmentation'][i]!=0)))
      if i==start:print('FIRST_STEP STD_V_GRAD_COMPUTED',value,'gradient_norm',np.linalg.norm(g),flush=True)
      print('STD_V_GRAD_DONE',i,'seconds',time.perf_counter()-begun,flush=True)
  if writer:writer.flush()
  print('STD_V_GRAD_STAGE_DONE',start,stop,flush=True)

if __name__=='__main__':app.run(lambda argv:(run_grad if os.environ.get('STD_V_MODE')=='grad' else run)(pyconfig.initialize(argv)))

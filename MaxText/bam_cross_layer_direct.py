"""Direct cross-layer Down sharing with calibration-only native-GELU selection.

Arguments: REPO OUT PREVIOUS_FUNCTIONAL_DIR [MaxText config...].
First 32 sequences are the existing held-out cohort; next 96 fit selection/up
heads. All paired loss variants use one padded, capture-free forward executable.
No checkpoint writes and no optimizer/MLP-budget adaptation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo,out,previous=map(Path,sys.argv[1:4]);sys.argv=[sys.argv[0],*sys.argv[4:]]
os.chdir(repo);sys.path.insert(0,str(repo/'MaxText'))
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,pyconfig,train
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import linears
from bam_global_projection_geometry import PARTITIONS

STRIDE=64
GROUPS=('local_0','local_1','fetch_2')
READS={'W_lq_c8_up':'q','W_lk_c8_up':'k','W_R_up':'vo'}
UPS=set(READS)|{'P_loc_up','mlp_address_up'}
original_dense=linears.DenseGeneral.__call__
def dense(self,inputs):
  kernel=self.get_variable('params','direct_kernel')
  if kernel is None:y=original_dense(self,inputs)
  else:
    x=jnp.asarray(inputs,self.dtype);w=jnp.asarray(kernel,self.dtype)
    y=jax.lax.dot_general(x,w,(((x.ndim-1,),(0,)),((),())),precision=jax.lax.Precision(self.matmul_precision))
    if self.use_bias:y=y+jnp.asarray(self.get_variable('params','direct_bias'),self.dtype)
  if self.is_mutable_collection('intermediates') and not self.is_initializing():
    if self.name=='full_m_read_down':
      self.sow('intermediates','direct_x',inputs[:,::STRIDE].astype(jnp.float32))
      self.sow('intermediates','direct_h',nn.gelu(y)[:,::STRIDE].astype(jnp.float32))
    elif self.name in UPS:
      self.sow('intermediates','direct_y',y[:,::STRIDE].astype(jnp.float32))
  return y
linears.DenseGeneral.__call__=dense

def forward(model,p,b,rng,capture=False):
  r1,r2=jax.random.split(rng)
  y=model.apply(p,b['inputs'],b['inputs_position'],decoder_segment_ids=b['inputs_segmentation'],
      decoder_target_mask=b['targets_segmentation'],decoder_target_tokens=b['targets'],
      enable_dropout=False,rngs={'params':r2,'dropout':r1},mutable=['intermediates'] if capture else False)
  values={}
  if capture:
    y,c=y
    for path,v in flatten_dict(c['intermediates']).items():
      if not path[-1].startswith('direct_'):continue
      if not isinstance(v,(tuple,list)):v=(v,)
      for i,a in enumerate(v):values['/'.join(path)+f'/call{i}']=a
  mask=b['targets_segmentation']!=0;count=mask.sum(-1)
  return jnp.sum(y[0]*mask,-1)/jnp.maximum(count,1),values

def select_units(features,score,rank):
  """Truncated pivoted Cholesky = greedy residual feature selection on Gram.

  Avoid full QR of a many-thousand-column bank. Double precision Gram and
  residuals, unique pivots; caller uses only calibration inputs and sensitivities.
  """
  assert features.shape[0]>rank, (features.shape,rank)
  f=features.astype(np.float64);f-=f.mean(0)
  f*=np.sqrt(np.maximum(score,1e-20))
  gram=f.T@f;diag=np.diag(gram).copy();total=float(diag.sum())
  rows=np.zeros((rank,len(diag)),np.float64);chosen=[]
  for j in range(rank):
    pivot=int(np.argmax(diag));energy=float(diag[pivot])
    if energy<=max(total,1.)*1e-14:raise ValueError(('degenerate selection',j,energy))
    residual=gram[pivot].copy()
    if j:residual-=rows[:j,pivot]@rows[:j]
    rows[j]=residual/np.sqrt(energy)
    diag=np.maximum(diag-rows[j]**2,0);chosen.append(pivot);diag[chosen]=-np.inf
  assert len(set(chosen))==rank
  return np.sort(chosen),dict(selection_residual_fraction=float(np.maximum(diag,0).sum()/total))

def fit_affine(h,y):
  a=h.astype(np.float64);b=y.reshape(len(y),-1).astype(np.float64)
  ma,mb=a.mean(0),b.mean(0);a-=ma;b-=mb
  gram=a.T@a;ridge=1e-4*np.trace(gram)/len(gram)
  k=np.linalg.solve(gram+ridge*np.eye(len(gram)),a.T@b);bias=mb-ma@k
  return k.astype(np.float32).reshape((h.shape[-1],)+y.shape[1:]),bias.astype(np.float32).reshape(y.shape[1:])

def run(cfg):
  assert cfg.only_eval and not cfg.bam_diagnostics
  cfg.get_keys()['enable_checkpointing']=False
  out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
  subprocess.run(['git','diff','--quiet'],check=True)
  rng,_,manager,mesh,model,_,tx=train.setup_mesh_and_model(cfg)
  tf.random.set_seed(int(cfg.data_shuffle_seed));data,evaluation=create_data_iterator(cfg,mesh)
  state,_,_,_=max_utils.setup_training_state(model,data,tx,cfg,rng,mesh,manager)
  params=state.params;flat=flatten_dict(params)
  downpaths={g:next(p for p in flat if g in p and p[-2:]==('full_m_read_down','kernel')) for g in GROUPS}
  r=int(flat[downpaths[GROUPS[0]]].shape[-1]);rank=384 if r==256 else 512
  assert r in (256,384)
  shared_mlp=bool(cfg.bam_share_mlp_write_down)
  targets={p:v for p,v in flat.items() if p[-1]=='kernel' and
      (p[-2]=='full_m_read_down' or p[-2] in UPS-{'mlp_address_up'} or (p[-2]=='mlp_address_up' and shared_mlp))}
  padded=dict(flat)
  for p,v in targets.items():
    axis=-1 if p[-2]=='full_m_read_down' else 0
    pad=[(0,0)]*v.ndim;pad[axis]=(0,rank-v.shape[axis])
    padded[p[:-1]+('direct_kernel',)]=jnp.pad(v,pad)
    bp=p[:-1]+('bias',)
    if bp in flat:padded[p[:-1]+('direct_bias',)]=flat[bp]
  batches=[];hashes=[];chunks={};masks=[]
  capture=jax.jit(lambda p,b:forward(model,p,b,rng,True))
  for i in range(32):
    b=next(evaluation);batches.append(b)
    hashes.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in np.asarray(b['inputs']))
    masks.append(np.asarray(b['targets_segmentation'][:,::STRIDE])!=0)
    with mesh,nn_partitioning.axis_rules(cfg.logical_axis_rules):_,values=jax.device_get(capture(params,b))
    for n,v in values.items():chunks.setdefault(n,[]).append(np.asarray(v))
    if i%4==3:print('DIRECT_CAPTURE',i+1,flush=True)
  assert len(set(hashes))==128
  prior=json.loads((previous/'activation-report.json').read_text())
  assert hashes[:32]==prior['sequence_hashes']
  captures={n:np.concatenate(v,axis=1) for n,v in chunks.items()};del chunks
  valid=np.concatenate(masks);fit=valid.copy();fit[:32]=False;hold=valid.copy();hold[32:]=False
  layers=[]
  for i in range(18):
    g=GROUPS[i%3];scan=i//3;dp=downpaths[g];prefix='/'.join(dp[1:-1])
    xs=[captures[prefix+'/direct_x/call0'][scan]]
    hs=[captures[prefix+'/direct_h/call0'][scan]]
    if shared_mlp and i%3==1:
      xs.append(captures[prefix+'/direct_x/call1'][scan]);hs.append(captures[prefix+'/direct_h/call1'][scan])
    heads=[];score=np.zeros(r)
    for p,v in targets.items():
      if p[-2]=='full_m_read_down' or g not in p:continue
      # Final carried M has no consumer: do not favor its unused writer.
      if i==17 and p[-2]=='P_loc_up':continue
      call=1 if p[-2]=='mlp_address_up' else 0
      yp='/'.join(p[1:-1])+'/direct_y/call0';y=captures[yp][scan]
      w=np.asarray(v[:,scan],np.float64)
      energy=np.mean(y[fit].astype(np.float64)**2,axis=(0,-1))
      sensitivity=np.sum(w*w,axis=-1)/np.maximum(energy,1e-8)
      z=sensitivity.reshape(r,-1).sum(1);score+=z/max(z.sum(),1e-20)
      heads.append((p,call,y))
    score/=len(heads)
    layers.append(dict(index=i,scan=scan,dp=dp,w=np.asarray(flat[dp][:,scan],np.float32),xs=xs,hs=hs,heads=heads,score=score))
  report=dict(checkpoint=cfg.load_parameters_path,diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      rank_before=r,rank_shared=rank,fit_sequences=96,heldout_sequences=32,sequence_hashes=hashes,
      fit_indices=[32,128],heldout_indices=[0,32],selection_rows_per_input=256,sampling_stride=STRIDE,
      method='calibration native-GELU unit selection + ridge affine Up refit',variants={},banks=[],reconstruction=[])
  plain=jax.jit(lambda p,b:forward(model,p,b,rng,False)[0])
  def evaluate(p):
    losses=[]
    for b in batches[:8]:
      with mesh,nn_partitioning.axis_rules(cfg.logical_axis_rules):losses.extend(np.asarray(plain(unflatten_dict(p),b)).tolist())
    return np.asarray(losses)
  original=evaluate(flat);baseline=evaluate(padded)
  report['baseline_losses']=baseline.tolist()
  report['padding_minus_original']=(baseline-original).tolist()
  report['original_minus_previous']=(original-np.asarray(prior['baseline_sequence_losses'])).tolist()
  assert np.max(np.abs(baseline-original))<.01
  report_path=out/'direct-report.json'
  def save():report['elapsed_seconds']=time.monotonic()-start;report_path.write_text(json.dumps(report,indent=2)+'\n')
  hidden=jax.jit(lambda x,w:nn.gelu(jnp.asarray(x,cfg.dtype)@jnp.asarray(w,cfg.dtype)).astype(jnp.float32))
  for name in ('native_refit','blocks','thirds','halves','global'):
    variant=dict(padded);bank_for={}
    if name=='native_refit':bank_for={i:l['w'] for i,l in enumerate(layers)}
    else:
      for gi,ids in enumerate(PARTITIONS[name]):
        bank=np.concatenate([layers[i]['w'] for i in ids],axis=1)
        score=np.concatenate([layers[i]['score'] for i in ids]);inputs=[]
        for i in ids:
          for x in layers[i]['xs']:
            xf=x[fit];indices=np.random.default_rng(20261013+i).choice(len(xf),256,replace=False);inputs.append(xf[indices])
        inputs=np.concatenate(inputs)
        features=np.asarray(hidden(inputs,bank))
        chosen,details=select_units(features,score,rank);joint=bank[:,chosen]
        for i in ids:bank_for[i]=joint
        report['banks'].append(dict(variant=name,group=gi,layers=ids,selected_columns=chosen.tolist(),**details))
        print('DIRECT_BANK',name,gi,flush=True)
    # Work in host arrays and transfer once per leaf, avoiding repeated TPU dispatch.
    host={p:np.asarray(v).copy() for p,v in variant.items() if p[-1].startswith('direct_')}
    biases={}
    native_width=r if name=='native_refit' else rank
    for i,l in enumerate(layers):
      scan=l['scan'];joint=bank_for[i]
      if name!='native_refit':host[l['dp'][:-1]+('direct_kernel',)][:,scan]=joint
      for p,call,y in l['heads']:
        h=l['hs'][call] if name=='native_refit' else np.asarray(hidden(l['xs'][call],joint))
        k,b=fit_affine(h[fit],y[fit]);kp=p[:-1]+('direct_kernel',)
        host[kp][:,scan]=0;host[kp][:native_width,scan]=k
        if p[-2] in READS:
          bp=p[:-2]+('general_'+READS[p[-2]]+'_pre_bias',)
          if bp not in biases:biases[bp]=np.asarray(flat[bp]).copy()
          biases[bp][:,scan]+=b
        else:host[p[:-1]+('direct_bias',)][:,scan]=b
        pred=h[hold]@k.reshape(native_width,-1)+b.reshape(-1)
        target=y[hold].reshape(len(pred),-1)
        report['reconstruction'].append(dict(variant=name,layer=i,head=p[-2],
            relative_squared_error=float(np.sum((pred-target)**2)/max(np.sum(target*target),1e-20))))
      print('DIRECT_REFIT',name,i,flush=True)
    for p,v in {**host,**biases}.items():variant[p]=jnp.asarray(v,dtype=variant[p].dtype)
    loss=evaluate(variant);delta=loss-baseline
    report['variants'][name]=dict(losses=loss.tolist(),paired_deltas=delta.tolist(),mean_delta=float(delta.mean()),
        standard_error=float(delta.std(ddof=1)/np.sqrt(32)))
    if name!='native_refit':
      down_before=sum(int(v.size) for p,v in targets.items() if p[-2]=='full_m_read_down')
      up_before=sum(int(v.size) for p,v in targets.items() if p[-2]!='full_m_read_down')
      down_after=len(PARTITIONS[name])*1200*rank;up_after=up_before//r*rank
      report['variants'][name]['parameter_savings']=down_before+up_before-down_after-up_after
    np.savez_compressed(out/(name+'-parameters.npz'),**{'/'.join(p):v for p,v in {**host,**biases}.items()})
    print('DIRECT_LOSS',name,report['variants'][name]['mean_delta'],flush=True);save()
  # Calibrated banks and per-sequence metrics suffice to reproduce interventions;
  # retain captures for audit, never route them through the orchestration host.
  np.savez_compressed(out/'direct-captures.npz',valid=valid,**captures)
  save();print('DIRECT_DONE',out,flush=True)

def main(argv):run(pyconfig.initialize(argv))
if __name__=='__main__':app.run(main)

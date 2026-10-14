"""Read-only held-out evaluation of blockwise common input subspaces.

Usage: script SEALED_REPO OUTPUT_DIR GEOMETRY_DIR [MaxText arguments].
Fixed32-sequence Pile cohort. No fitted activation statistics or training.
Project each original W onto the joint weight-SVD basis, keeping each layer's
own coefficients before GELU. This is a compression intervention, not direct
cross-layer tying of nonlinear features. Checkpoint arrays are never modified.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo,out,geometry=map(Path,sys.argv[1:4]);sys.argv=[sys.argv[0],*sys.argv[4:]]
os.chdir(repo);sys.path.insert(0,str(repo/'MaxText'))
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app
from flax import linen as nn
from flax.core import FrozenDict,freeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import linears

STRIDE=64
GROUPS=('local_0','local_1','fetch_2')
original_dense=linears.DenseGeneral.__call__
def capture_dense(self,inputs):
  y=original_dense(self,inputs)
  if self.name=='full_m_read_down' and self.is_mutable_collection('intermediates') and not self.is_initializing():
    self.sow('intermediates','cross_x',inputs[:,::STRIDE].astype(jnp.float32))
    self.sow('intermediates','cross_h',nn.gelu(y)[:,::STRIDE].astype(jnp.float32))
  return y
linears.DenseGeneral.__call__=capture_dense

def forward(model,params,batch,rng,capture):
  r1,r2=jax.random.split(rng)
  result=model.apply(params,batch['inputs'],batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],decoder_target_mask=batch['targets_segmentation'],
      decoder_target_tokens=batch['targets'],enable_dropout=False,rngs={'dropout':r1,'params':r2},
      mutable=['intermediates'] if capture else False)
  sampled={}
  if capture:
    result,collections=result
    for path,values in flatten_dict(collections['intermediates']).items():
      if path[-1] not in ('cross_x','cross_h'):continue
      if not isinstance(values,(tuple,list)):values=(values,)
      for call,value in enumerate(values):sampled['/'.join(path)+f'/call{call}']=value
  mask=batch['targets_segmentation']!=0
  counts=jnp.sum(mask,axis=-1)
  return {'loss':jnp.sum(result[0]*mask,axis=-1)/jnp.maximum(counts,1),'weights':counts},sampled

def cka(a,b):
  a=a-a.mean(0);b=b-b.mean(0)
  return float(np.sum((a.T@b)**2)/max(float(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)),1e-30))

def run(cfg):
  assert cfg.only_eval and not cfg.bam_diagnostics
  cfg.get_keys()['enable_checkpointing']=False
  out.mkdir(parents=True,exist_ok=True)
  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  subprocess.run(['git','diff','--quiet'],check=True)
  start=time.monotonic()
  rng,_,manager,mesh,model,_,tx=train.setup_mesh_and_model(cfg)
  tf.random.set_seed(int(cfg.data_shuffle_seed))
  data,evaluation=create_data_iterator(cfg,mesh)
  state,_,_,_=max_utils.setup_training_state(model,data,tx,cfg,rng,mesh,manager)
  params=state.params;flat=flatten_dict(params)
  capture=jax.jit(lambda p,b,r:forward(model,p,b,r,True))
  plain=jax.jit(lambda p,b,r:forward(model,p,b,r,False)[0])
  batches=[];hashes=[];parts={};masks=[];losses=[]
  for index in range(8):
    batch=next(evaluation);batches.append(batch)
    hashes.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in np.asarray(batch['inputs']))
    masks.append(np.asarray(batch['targets_segmentation'][:,::STRIDE])!=0)
    with mesh,nn_partitioning.axis_rules(cfg.logical_axis_rules):
      metrics,values=jax.device_get(capture(params,batch,rng))
      if index==0:
        check=jax.device_get(plain(params,batch,rng))
        np.testing.assert_allclose(metrics['loss'],check['loss'],rtol=1e-6,atol=1e-6)
    losses.extend(metrics['loss'].tolist())
    for name,value in values.items():parts.setdefault(name,[]).append(np.asarray(value))
    print('CAPTURE',index,len(hashes),time.monotonic()-start,flush=True)
  assert len(hashes)==32 and len(set(hashes))==32
  values={n:np.concatenate(v,axis=1) for n,v in parts.items()}
  valid=np.concatenate(masks,axis=0)
  assert all(np.isfinite(a).all() for a in values.values())
  np.savez_compressed(out/'captures.npz',valid=valid,**values)
  bases=np.load(geometry/'bases.npz')
  report=dict(checkpoint=cfg.load_parameters_path,diagnostic_commit=commit,
      cohort_seed=cfg.data_shuffle_seed,sequence_hashes=hashes,stride=STRIDE,
      fit_sequences=0,heldout_sequences=32,baseline_sequence_losses=losses,
      activation_errors=[],activation_cka=[],loss_interventions={},
      scope='All six three-layer blocks; common pre-GELU input basis with per-layer coefficients.')
  selected={p:v for p,v in flat.items() if p[-2:]==('full_m_read_down','kernel')}
  assert len(selected)==3
  act=jax.jit(lambda x,w:nn.gelu(jnp.asarray(x,cfg.dtype)@jnp.asarray(w,cfg.dtype)).astype(jnp.float32))
  for block in range(6):
    hs=[]
    for offset,group in enumerate(GROUPS):
      key=next(p for p in selected if group in p)
      w=np.asarray(selected[key][:,block],dtype=np.float32)
      prefix='decoder/layers/'+group+'/block/self_attention/full_m_read_down/'
      hs.append(values[prefix+'cross_h/call0'][block][valid])
      calls=sorted(n for n in values if n.startswith(prefix+'cross_x/'))
      for name in calls:
        x=values[name][block];native=values[name.replace('cross_x','cross_h')][block]
        original=np.asarray(act(x,w))
        for rank in (256,384):
          u=bases[f'block{block}_rank{rank}'];wp=u@(u.T@w)
          projected=np.asarray(act(x,wp))
          error=np.sum((projected-original)**2,axis=-1);energy=np.sum(original**2,axis=-1)
          se=np.sum(error*valid,axis=1)/np.maximum(np.sum(energy*valid,axis=1),1e-30)
          xvalid=x[valid];y=xvalid@w;yp=xvalid@wp
          report['activation_errors'].append(dict(layer=3*block+offset,call=name.rsplit('/',1)[-1],rank=rank,
              pre_gelu_relative_squared_error=float(np.sum((yp-y)**2)/np.sum(y*y)),
              post_gelu_relative_squared_error=float(np.sum(error[valid])/np.sum(energy[valid])),
              per_sequence_relative_squared_error=se.tolist(),
              native_recompute_relative_squared_error=float(np.sum((original[valid]-native[valid])**2)/np.sum(native[valid]**2))))
    for i,j in ((0,1),(1,2),(0,2)):
      perm=np.random.default_rng(42).permutation(len(hs[j]))
      report['activation_cka'].append(dict(layers=[3*block+i,3*block+j],
          own_input_gelu_cka=cka(hs[i],hs[j]),token_permuted_cka=cka(hs[i],hs[j][perm])))
  (out/'activation-report.json').write_text(json.dumps(report,indent=2)+'\n')
  for rank in (256,384):
    modified=dict(flat)
    for key,w0 in selected.items():
      w=np.asarray(w0,dtype=np.float32).copy()
      for block in range(6):
        u=bases[f'block{block}_rank{rank}'];w[:,block]=u@(u.T@w[:,block])
      modified[key]=jnp.asarray(w,dtype=w0.dtype)
    tree=unflatten_dict(modified)
    if isinstance(params,FrozenDict):tree=freeze(tree)
    measured=[]
    for batch in batches:
      with mesh,nn_partitioning.axis_rules(cfg.logical_axis_rules):metrics=jax.device_get(plain(tree,batch,rng))
      measured.extend(metrics['loss'].tolist())
    delta=np.asarray(measured)-np.asarray(losses)
    report['loss_interventions'][str(rank)]=dict(sequence_losses=measured,paired_deltas=delta.tolist(),
         mean_delta=float(delta.mean()),standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))))
    print('LOSS_RANK',rank,report['loss_interventions'][str(rank)]['mean_delta'],flush=True)
    report['elapsed_seconds']=time.monotonic()-start
    (out/'activation-report.json').write_text(json.dumps(report,indent=2)+'\n')
  print('FUNCTIONAL_DONE',out,flush=True)

def main(argv):
  pyconfig.initialize(argv);run(pyconfig.config)
if __name__=='__main__':app.run(main)

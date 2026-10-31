"""Native-JAX FP32 diagnostic of low/mid/high MHA residual and M-write contributions.

Checkpoint parameter values stay unchanged. FP32 forward arithmetic deliberately
separates functional diagnosis from TPU BF16 fusion differences. No Pallas or
Splash operator is used, and no training process or checkpoint is modified.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


@contextlib.contextmanager
def intervene_mha(modes,record=False):
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions

  def intercept(next_fun,args,kwargs,ctx):
    module=ctx.module
    if ctx.method_name!='__call__' or not isinstance(module,attentions.BamAttention):
      return next_fun(*args,**kwargs)
    layer=kwargs.get('layer_index')
    if layer is None:layer=module.layer_inx
    attention=attentions._attention_op
    local=attentions.BamAttention._regularize_local_o_heads
    projection=attentions.BamAttention.out_projection
    state={}

    def attention_call(*args,**kwargs):
      context,weights=attention(*args,**kwargs)
      output=jnp.where(modes[layer]==2,jnp.zeros_like(context),context)
      return output,weights

    def local_call(self,value,deterministic):
      result=local(self,value,deterministic);state['local']=result
      return result

    def projection_call(self,output_dim,value):
      selected=jnp.where(modes[layer]==1,state['local'],value)
      if record and not self.is_initializing():
        self.sow('intermediates','native_mha_head_probe',jnp.stack((
            jnp.mean(value.astype(jnp.float32)**2),
            jnp.mean(selected.astype(jnp.float32)**2),
            jnp.mean(state['local'].astype(jnp.float32)**2))))
      return projection(self,output_dim,selected)

    attentions._attention_op=attention_call
    attentions.BamAttention._regularize_local_o_heads=local_call
    attentions.BamAttention.out_projection=projection_call
    try:
      result=next_fun(*args,**kwargs)
      if record and not module.is_initializing() and len(result)==3 and result[2] is not None:
        module.sow('intermediates','native_mha_write_probe',jnp.mean(result[2][0].astype(jnp.float32)**2))
      return result
    finally:
      attentions._attention_op=attention
      attentions.BamAttention._regularize_local_o_heads=local
      attentions.BamAttention.out_projection=projection

  with nn.intercept_methods(intercept):yield


def extract(collection,name,final_layer):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  rows=[]
  for path,value in flatten_dict(collection).items():
    if path[-1]!=name:continue
    while isinstance(value,(tuple,list)):value=value[0]
    if 'layers' in path:rows.append((0,value))
    elif 'final_local_layer' in path:rows.append((final_layer,value[None]))
    else:
      match=next((re.fullmatch(r'layers_(\d+)',p) for p in path if re.fullmatch(r'layers_(\d+)',p)),None)
      if match:rows.append((int(match[1]),value[None]))
  if not rows:raise ValueError(name)
  return jnp.concatenate([v for _,v in sorted(rows)],axis=0)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir',type=Path,required=True);p.add_argument('--exp',required=True)
  p.add_argument('--checkpoint',required=True);p.add_argument('--cohort',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--length',type=int,default=2048)
  args=p.parse_args();os.chdir(args.runtime_dir);sys.path.insert(0,str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer
  from h_write_balance_probe import restore_params
  assert jax.process_count()==1
  (args.output/'configuration'/'native-mha').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=args.exp,run_name='native-mha',
      base_output_directory=str(args.output/'configuration')+'/',enable_checkpointing=False,
      only_eval=True,dataset_type='synthetic',target_train_batch_size=None,per_device_batch_size=1.,
      max_target_length=args.length,max_prefill_predict_length=args.length,
      record_internal_nn_metrics=False,record_training_health_metrics=False,jax_cache_dir='',log_config=False)
  cfg.get_keys().update(dtype=jnp.float32,bam_pallas_core=False,bam_pallas_write=False,
                        bam_splash_attention=False,matmul_precision='highest')
  assert cfg.num_decoder_layers==32
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
  cohort=np.load(args.cohort)['tokens'][:32,:args.length+1];assert cohort.shape==(32,args.length+1)
  positions=jnp.arange(args.length,dtype=jnp.int32)[None];segments=jnp.ones((1,args.length),jnp.int32)
  regions={'lower':list(range(8)),'middle':list(range(12,20)),'upper':list(range(24,32))}
  names=['learned'];controls=[np.zeros(32,np.int32)]
  for region,layers in regions.items():
    for label,mode in [('residual_off',1),('residual_and_write_off',2)]:
      value=np.zeros(32,np.int32);value[layers]=mode;controls.append(value);names.append(region+'_'+label)
  controls=np.stack(controls);traces=0

  def apply(params,tokens):
    return model.apply(params,tokens[:,:-1],positions,decoder_segment_ids=segments,
        decoder_target_mask=segments,decoder_target_tokens=tokens[:,1:],enable_dropout=False)

  def forward(params,tokens,modes):
    nonlocal traces
    traces+=1;print('NATIVE_FORWARD_TRACE',traces,flush=True)
    with intervene_mha(modes):xent,_,_=apply(params,tokens)
    return jnp.mean(xent.astype(jnp.float32))

  start=time.monotonic()
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),jax.default_matmul_precision('highest'):
    print('RESTORE_BEGIN',args.checkpoint,flush=True)
    params=jax.device_put(unfreeze(nn.unbox(restore_params(args.checkpoint))),jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    jax.block_until_ready(params);print('RESTORE_OK',round(time.monotonic()-start,2),flush=True)
    compiled=jax.jit(forward);reference=jax.jit(lambda params,tokens:jnp.mean(apply(params,tokens)[0].astype(jnp.float32)))
    losses=np.empty((7,32),np.float64);noop=[]
    for variant,name in enumerate(names):
      before=time.monotonic()
      for index,tokens in enumerate(cohort):
        losses[variant,index]=float(compiled(params,jnp.asarray(tokens[None]),jnp.asarray(controls[variant])))
        if index in (0,7,31):print('FORWARD_OK',name,index+1,round(time.monotonic()-before,2),flush=True)
        if variant==0 and index<2:
          delta=float(reference(params,jnp.asarray(tokens[None])))-losses[0,index]
          noop.append(delta);assert abs(delta)<1e-5,(index,delta)
          print('NATIVE_NOOP_CHECK_OK',index,delta,flush=True)
      assert np.isfinite(losses[variant]).all()
      print('VARIANT_DONE',name,float(losses[variant].mean()),'seconds',round(time.monotonic()-before,2),flush=True)
    repeated=float(compiled(params,jnp.asarray(cohort[:1]),jnp.asarray(controls[0])))
    assert repeated==losses[0,0]
  np.savez_compressed(args.output/'paired-losses.npz',losses=losses,controls=controls)
  result=dict(exp=args.exp,checkpoint=args.checkpoint,names=names,regions=regions,
      mean_loss=losses.mean(1).tolist(),mean_delta=(losses-losses[0]).mean(1).tolist(),
      sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in cohort],
      sequences=32,length=args.length,noop_deltas=noop,baseline_repeat_exact=True,forward_traces=traces,
      runtime_commit=__import__('subprocess').check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,
      backend='native JAX FP32 activations/highest matmul precision; checkpoint parameter values/dtypes unchanged; Pallas/Splash OFF',
      semantics='Each8-layer region: mode1 removes MHA context from W_O input only, preserving attention M-write content; mode2 zeroes context before local O addition, removing both paths. Local O and all parameters preserved. Paired same-graph CE; native FP32 baseline differs numerically from formal BF16 training.')
  (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('NATIVE_MHA_COMPLETE',result['mean_delta'],flush=True)


if __name__=='__main__':main()

"""Freeze upper Q/K dynamic addresses using disjoint calibration/evaluation sequences."""
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


def normalized_address(key, epsilon):
  import jax
  import jax.numpy as jnp
  x = key.astype(jnp.float32)
  return x * jax.lax.rsqrt(jnp.mean(x*x, axis=-1, keepdims=True)+epsilon)


def frozen_inputs(key, logit, mean, epsilon):
  """Realize g(t)*mean(normalized key) through the original normalized-key API.

  The mean is NOT renormalized to unit RMS. Correct the transported gate for
  the kernel's unavoidable key normalization, preserving the original gate's
  multiplicative role. Compute in FP32; no checkpoint parameters are changed.
  """
  import jax
  import jax.numpy as jnp
  # Preserve the original Pallas input dtype and its rounding barrier. Promoting
  # even an unmodified key to FP32 changes upstream fused BF16 rounding on TPU.
  proxy = jnp.broadcast_to(mean.astype(key.dtype), key.shape)
  rounded = proxy.astype(jnp.float32)
  rms = jnp.sqrt(jnp.mean(rounded*rounded, axis=-1)+epsilon)
  gate = jax.nn.sigmoid(logit.astype(jnp.float32))*rms
  gate = jnp.clip(gate, 1e-30, 1-jnp.finfo(jnp.float32).eps)
  return proxy, (jnp.log(gate)-jnp.log1p(-gate)).astype(logit.dtype)


@contextlib.contextmanager
def address_intervention(controls=None, addresses=None, record=False,
                         zero_mha=None, record_context=False, dtype_debug=False):
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions, bam_pallas

  def intercept(next_fun, args, kwargs, ctx):
    module = ctx.module
    if ctx.method_name != '__call__' or not isinstance(module, attentions.BamAttention):
      return next_fun(*args, **kwargs)
    layer = kwargs.get('layer_index')
    if layer is None:
      layer = module.layer_inx
    original = bam_pallas.read
    original_splash = attentions._bam_splash_attention
    original_attention = attentions._attention_op
    original_local = attentions.BamAttention._regularize_local_o_heads
    original_projection = attentions.BamAttention.out_projection
    state = {}

    def modified_context(output):
      if zero_mha is not None and not module.is_initializing():
        # Mode1 removes context only at W_O input; mode2 removes it before M write too.
        changed = jnp.where(zero_mha[layer] == 2,jnp.zeros_like(output),output)
        if record_context:
          module.sow('intermediates','mha_context_probe',jnp.stack((
              jnp.mean(output.astype(jnp.float32)**2),
              jnp.mean(changed.astype(jnp.float32)**2))))
        return changed
      return output

    def modified_splash(*args,**kwargs):
      return modified_context(original_splash(*args,**kwargs))

    def modified_attention(*args,**kwargs):
      context,weights=original_attention(*args,**kwargs)
      return modified_context(context),weights

    def modified_local(self, local, deterministic):
      output = original_local(self,local,deterministic)
      state['local_o'] = output
      return output

    def modified_projection(self, output_dim, output):
      if dtype_debug:
        print('PROJECTION_DTYPES',output.dtype,state['local_o'].dtype,flush=True)
      if zero_mha is not None and not self.is_initializing():
        local = state['local_o']
        changed = jnp.where(zero_mha[layer] == 1,local,output)
        if record_context:
          ms = lambda x:jnp.mean(x.astype(jnp.float32)**2)
          self.sow('intermediates','mha_output_probe',jnp.stack((
              ms(output),ms(changed),ms(changed-local),ms(local))))
        output = changed
      return original_projection(self,output_dim,output)

    def modified(matrix, weight, q_key, q_logit, k_key, k_logit, vo_key,
                 o_logit, v_logit, q_standard, k_standard, **kw):
      if dtype_debug:
        print('READ_DTYPES',q_key.dtype,q_logit.dtype,k_key.dtype,k_logit.dtype,flush=True)
      if not module.is_initializing():
        if record:
          means = jnp.stack([normalized_address(x,kw['read_epsilon']).mean((0,1))
                             for x in (q_key,k_key)])
          module.sow('intermediates','read_address_probe',means)
        if controls is not None:
          changed=[]
          for arm,(key,logit) in enumerate(((q_key,q_logit),(k_key,k_logit))):
            proxy,proxy_logit = frozen_inputs(key,logit,addresses[layer,arm],kw['read_epsilon'])
            changed.append((jnp.where(controls[layer,arm],proxy,key),
                            jnp.where(controls[layer,arm],proxy_logit,logit)))
          (q_key,q_logit),(k_key,k_logit)=changed
      return original(matrix,weight,q_key,q_logit,k_key,k_logit,vo_key,
                      o_logit,v_logit,q_standard,k_standard,**kw)

    bam_pallas.read=modified
    attentions._bam_splash_attention=modified_splash
    attentions._attention_op=modified_attention
    attentions.BamAttention._regularize_local_o_heads=modified_local
    attentions.BamAttention.out_projection=modified_projection
    try:
      result=next_fun(*args,**kwargs)
      if (record_context and not module.is_initializing() and len(result)==3
          and result[2] is not None):
        module.sow('intermediates','mha_write_content_probe',
                   jnp.mean(result[2][0].astype(jnp.float32)**2))
      return result
    finally:
      bam_pallas.read=original
      attentions._bam_splash_attention=original_splash
      attentions._attention_op=original_attention
      attentions.BamAttention._regularize_local_o_heads=original_local
      attentions.BamAttention.out_projection=original_projection

  with nn.intercept_methods(intercept):
    yield


def extract_addresses(collection, final_layer, name='read_address_probe'):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  found=[]
  for path,value in flatten_dict(collection).items():
    if path[-1]!=name:continue
    while isinstance(value,(tuple,list)):value=value[0]
    if 'layers' in path:found.append((0,value))
    elif 'final_local_layer' in path:found.append((final_layer,value[None]))
    else:
      match=next((re.fullmatch(r'layers_(\d+)',part) for part in path
                  if re.fullmatch(r'layers_(\d+)',part)),None)
      if match:found.append((int(match[1]),value[None]))
  if not found:raise ValueError('No read addresses captured')
  return jnp.concatenate([v for _,v in sorted(found)],axis=0)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir',type=Path,required=True)
  p.add_argument('--exp',required=True)
  p.add_argument('--checkpoint',required=True)
  p.add_argument('--cohort',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--high-start',type=int,default=24)
  p.add_argument('--length',type=int,default=2048)
  p.add_argument('--vmem-mib',type=int,default=40)
  p.add_argument('--fidelity-only',action='store_true')
  args=p.parse_args();os.chdir(args.runtime_dir);sys.path.insert(0,str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer
  from layers import attentions
  from h_write_balance_probe import restore_params
  assert jax.process_count()==1
  (args.output/'configuration'/'address-freeze').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=args.exp,
      run_name='address-freeze',base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False,only_eval=True,dataset_type='synthetic',
      target_train_batch_size=None,per_device_batch_size=1.,max_target_length=args.length,
      max_prefill_predict_length=args.length,record_internal_nn_metrics=False,
      record_training_health_metrics=False,jax_cache_dir='',log_config=False)
  cfg.get_keys().update(bam_pallas_read_tile=128,bam_pallas_write_tile=128,
                        bam_pallas_vmem_mib=args.vmem_mib)
  assert cfg.bam_pallas_core and cfg.bam_pallas_body=='v8t'
  assert all(hasattr(attentions,name) for name in ('_bam_splash_attention','_attention_op'))
  assert all(hasattr(attentions.BamAttention,name) for name in ('_regularize_local_o_heads','out_projection'))
  print('HISTORICAL_INTERVENTION_HOOKS_VERIFIED',flush=True)
  assert 0<=args.high_start<cfg.num_decoder_layers
  cohort=np.load(args.cohort)['tokens'][:32,:args.length+1];assert cohort.shape==(32,args.length+1)
  calibration,evaluation=cohort[:16],cohort[16:]
  names=['learned','upper_Q_constant_address','upper_K_constant_address','upper_QK_constant_address']
  regions=[('lower',0,8),('middle',12,20),('upper',24,32)]
  for region,_,_ in regions:
    names.extend([region+'_MHA_residual_off',region+'_MHA_residual_and_write_off'])
  controls=np.zeros((len(names),cfg.num_decoder_layers,2),bool)
  controls[1,args.high_start:,0]=True;controls[2,args.high_start:,1]=True
  controls[3,args.high_start:]=True
  mha_controls=np.zeros((len(names),cfg.num_decoder_layers),np.int32)
  for index,(_,begin,end) in enumerate(regions):
    mha_controls[4+index*2,begin:end]=1
    mha_controls[5+index*2,begin:end]=2
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  model=Transformer(cfg,mesh,quant=None);positions=jnp.arange(args.length,dtype=jnp.int32)[None]
  segments=jnp.ones((1,args.length),jnp.int32);traces={'calibration':0,'evaluation':0}

  def apply(params,tokens):
    return model.apply(params,tokens[:,:-1],positions,decoder_segment_ids=segments,
        decoder_target_mask=segments,decoder_target_tokens=tokens[:,1:],enable_dropout=False)

  def calibrate(params,tokens):
    traces['calibration']+=1;print('CALIBRATION_TRACE',traces['calibration'],flush=True)
    with address_intervention(record=True):
      _,aux=model.apply(params,tokens[:,:-1],positions,decoder_segment_ids=segments,
          decoder_target_mask=segments,decoder_target_tokens=tokens[:,1:],enable_dropout=False,
          mutable=['intermediates'])
    return extract_addresses(aux['intermediates'],cfg.num_decoder_layers-1)

  def evaluate(params,tokens,control,means,mha_control):
    traces['evaluation']+=1;print('EVALUATION_TRACE',traces['evaluation'],flush=True)
    with address_intervention(control,means,zero_mha=mha_control):xent,_,_=apply(params,tokens)
    return jnp.mean(xent.astype(jnp.float32))

  start=time.monotonic()
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    print('RESTORE_BEGIN',args.checkpoint,flush=True)
    params=jax.device_put(unfreeze(nn.unbox(restore_params(args.checkpoint))),
                         jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    jax.block_until_ready(params);print('RESTORE_OK',round(time.monotonic()-start,2),flush=True)
    if args.fidelity_only:
      dummy=jnp.zeros((cfg.num_decoder_layers,2,cfg.num_query_heads,cfg.bam_v),jnp.float32)
      check={}
      for case in ['original','wrappers','addresses_only','mha_only','combined']:
        def check_forward(params,tokens,control,means,mha_control):
          if case=='original':xent,_,_=apply(params,tokens)
          else:
            with address_intervention(
                control if case in ('addresses_only','combined') else None,
                means if case in ('addresses_only','combined') else None,
                zero_mha=mha_control if case in ('mha_only','combined') else None,
                dtype_debug=True):xent,_,_=apply(params,tokens)
          return jnp.mean(xent.astype(jnp.float32))
        check_compiled=jax.jit(check_forward)
        check[case]=[float(check_compiled(params,jnp.asarray(evaluation[i:i+1]),
            jnp.asarray(controls[0]),dummy,jnp.asarray(mha_controls[0]))) for i in range(2)]
        print('FIDELITY_COMPONENT',case,check[case],flush=True)
      (args.output/'fidelity-components.json').write_text(json.dumps(check,indent=2)+'\n')
      return
    capture=jax.jit(calibrate);total=None
    for index,tokens in enumerate(calibration):
      value=np.asarray(capture(params,jnp.asarray(tokens[None])))
      assert value.shape==(cfg.num_decoder_layers,2,cfg.num_query_heads,cfg.bam_v)
      total=value.astype(np.float64) if total is None else total+value
    means=(total/len(calibration)).astype(np.float32)
    assert np.isfinite(means).all();assert np.max(np.sqrt(np.mean(means**2,axis=-1)))<=1.00001
    print('CALIBRATION_DONE',round(time.monotonic()-start,2),flush=True)
    forward=jax.jit(evaluate);losses=np.empty((len(names),16),np.float64)
    for variant,name in enumerate(names):
      before=time.monotonic()
      for index,tokens in enumerate(evaluation):
        losses[variant,index]=float(forward(params,jnp.asarray(tokens[None]),
            jnp.asarray(controls[variant]),jnp.asarray(means),jnp.asarray(mha_controls[variant])))
      assert np.isfinite(losses[variant]).all()
      print('VARIANT_DONE',name,float(losses[variant].mean()),'seconds',round(time.monotonic()-before,2),flush=True)
      if variant==0:
        reference=jax.jit(lambda params,tokens:jnp.mean(apply(params,tokens)[0].astype(jnp.float32)))
        no_op_deltas=[float(reference(params,jnp.asarray(evaluation[i:i+1])))-losses[0,i] for i in range(2)]
        (args.output/'noop-check.json').write_text(json.dumps(dict(deltas=no_op_deltas))+'\n')
        assert max(map(abs,no_op_deltas))<1e-5,no_op_deltas
        print('EXACT_FORWARD_CHECK_OK',no_op_deltas,flush=True)
    repeated=float(forward(params,jnp.asarray(evaluation[:1]),jnp.asarray(controls[0]),
                          jnp.asarray(means),jnp.asarray(mha_controls[0])))
    assert repeated==losses[0,0]
  np.savez_compressed(args.output/'paired-losses.npz',losses=losses,controls=controls,
                      mha_controls=mha_controls,mean_normalized_addresses=means)
  meta=dict(exp=args.exp,checkpoint=args.checkpoint,names=names,
      mean_loss=losses.mean(1).tolist(),mean_delta=(losses-losses[0]).mean(1).tolist(),
      calibration_sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in calibration],
      evaluation_sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in evaluation],
      high_start=args.high_start,upper_layers=list(range(args.high_start,cfg.num_decoder_layers)),
      original_vs_noop_deltas=no_op_deltas,baseline_repeat_exact=True,traces=traces,
      mean_address_rms=np.sqrt(np.mean(means**2,axis=-1)).tolist(),
      sequences_calibration=16,sequences_evaluation=16,length=args.length,
      runtime_commit=__import__('subprocess').check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,
      semantics='First16 sequences calibrate mean RMS-normalized pre-gate Q/K addresses; disjoint last16 evaluate. Frozen mean is not renormalized. Original gates retain multiplicative role through kernel-input normalization compensation. MHA mode1 replaces W_O input with local O but keeps attention write content; mode2 zeroes context before addition to local O, removing residual and attention-M-write contributions. Local O remains in both. Other parameters preserved; downstream responds normally.')
  (args.output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
  print('ADDRESS_ABLATION_COMPLETE',json.dumps(meta['mean_delta']),flush=True)


if __name__=='__main__':main()

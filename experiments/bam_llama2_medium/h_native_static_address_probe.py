"""Calibrated static upper Q/K addresses, native JAX FP32 with disjoint evaluation."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time


@contextlib.contextmanager
def address_probe(controls=None,means=None,record=False):
  import jax
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions
  keys={'W_lq_c8':('q',0),'W_lk_c8':('k',1)}
  gates={'W_lq_gate':0,'W_lk_gate':1}

  def intercept(next_fun,args,kwargs,ctx):
    module=ctx.module
    if ctx.method_name!='__call__' or not isinstance(module,attentions.BamAttention):
      return next_fun(*args,**kwargs)
    layer=kwargs.get('layer_index')
    if layer is None:layer=module.layer_inx
    project=attentions.BamAttention._project_column_read_key
    gate=attentions.BamAttention._project_read_gate_logits

    def project_call(self,name,*args,**kwargs):
      value=project(self,name,*args,**kwargs)
      if record and name in keys and not self.is_initializing():
        x=value.astype(jnp.float32)
        x*=jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+self._read_key_epsilon)
        self.sow('intermediates','native_address_'+keys[name][0],x.mean((0,1)))
      return value

    def gate_call(self,name,*args,**kwargs):
      value=gate(self,name,*args,**kwargs)
      if controls is not None and name in gates and not self.is_initializing():
        arm=gates[name]
        # Match the fixed pre-bias's checkpoint storage dtype exactly.
        mean=jnp.asarray(means[layer,arm],self.weight_dtype).astype(jnp.float32)
        assert value.shape[-1]==1 and value.shape[-2]==mean.shape[-2]
        rms=jnp.sqrt(jnp.mean(mean*mean,axis=-1)+self._read_key_epsilon)[...,None]
        g=jnp.clip(jax.nn.sigmoid(value.astype(jnp.float32))*rms,1e-30,1-jnp.finfo(jnp.float32).eps)
        replacement=(jnp.log(g)-jnp.log1p(-g)).astype(value.dtype)
        value=jnp.where(controls[layer,arm],replacement,value)
      return value

    attentions.BamAttention._project_column_read_key=project_call
    attentions.BamAttention._project_read_gate_logits=gate_call
    try:return next_fun(*args,**kwargs)
    finally:
      attentions.BamAttention._project_column_read_key=project
      attentions.BamAttention._project_read_gate_logits=gate

  with nn.intercept_methods(intercept):yield


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
  from h_read_address_ablation import freeze_address_parameters
  from h_mha_native_probe import extract
  assert jax.process_count()==1
  (args.output/'configuration'/'native-address').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=args.exp,run_name='native-address',
      base_output_directory=str(args.output/'configuration')+'/',enable_checkpointing=False,
      only_eval=True,dataset_type='synthetic',target_train_batch_size=None,per_device_batch_size=1.,
      max_target_length=args.length,max_prefill_predict_length=args.length,
      record_internal_nn_metrics=False,record_training_health_metrics=False,jax_cache_dir='',log_config=False)
  cfg.get_keys().update(dtype=jnp.float32,matmul_precision='highest',bam_pallas_core=False,
                        bam_pallas_write=False,bam_splash_attention=False)
  assert cfg.num_decoder_layers==32 and cfg.bam_read_gate_activation=='sigmoid'
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
  cohort=np.load(args.cohort)['tokens'][:32,:args.length+1];assert cohort.shape==(32,args.length+1)
  calibration,evaluation=cohort[:16],cohort[16:]
  positions=jnp.arange(args.length,dtype=jnp.int32)[None];segments=jnp.ones((1,args.length),jnp.int32)
  names=['learned','upper_Q_constant_address','upper_K_constant_address',
         'upper_QK_constant_address','upper_QK_off']
  controls=np.zeros((5,32,2),bool);controls[1,24:,0]=True;controls[2,24:,1]=True;controls[3,24:]=True
  traces={'calibration':0,'evaluation':0}

  def apply(params,tokens,mutable=False):
    return model.apply(params,tokens[:,:-1],positions,decoder_segment_ids=segments,
        decoder_target_mask=segments,decoder_target_tokens=tokens[:,1:],enable_dropout=False,
        **({'mutable':['intermediates']} if mutable else {}))

  def calibrate(params,tokens):
    traces['calibration']+=1;print('CALIBRATION_TRACE',traces['calibration'],flush=True)
    with address_probe(record=True):_,aux=apply(params,tokens,True)
    return jnp.stack([extract(aux['intermediates'],'native_address_'+arm,31) for arm in ['q','k']],axis=1)

  def evaluate(params,tokens,control,means):
    traces['evaluation']+=1;print('EVALUATION_TRACE',traces['evaluation'],flush=True)
    with address_probe(control,means):xent,_,_=apply(params,tokens)
    return jnp.mean(xent.astype(jnp.float32))

  start=time.monotonic()
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),jax.default_matmul_precision('highest'):
    print('RESTORE_BEGIN',args.checkpoint,flush=True)
    params=jax.device_put(unfreeze(nn.unbox(restore_params(args.checkpoint))),jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    jax.block_until_ready(params);print('RESTORE_OK',round(time.monotonic()-start,2),flush=True)
    capture=jax.jit(calibrate);total=None
    for index,tokens in enumerate(calibration):
      value=np.asarray(capture(params,jnp.asarray(tokens[None])));assert value.shape==(32,2,cfg.num_query_heads,cfg.bam_v)
      total=value.astype(np.float64) if total is None else total+value
    means=(total/16).astype(np.float32);assert np.isfinite(means).all()
    print('CALIBRATION_DONE',round(time.monotonic()-start,2),flush=True)
    forward=jax.jit(evaluate);reference=jax.jit(lambda p,t:jnp.mean(apply(p,t)[0].astype(jnp.float32)))
    losses=np.empty((len(names),16),np.float64);noop=[]
    for variant,name in enumerate(names):
      before=time.monotonic();variant_params=freeze_address_parameters(params,means,controls[variant],cfg.param_scan_axis)
      if name=='upper_QK_off':
        # Norm(0)=0: zero up+pre-bias is forward-equivalent to zero dynamic gates.
        variant_params=freeze_address_parameters(params,np.zeros_like(means),controls[3],cfg.param_scan_axis)
      for index,tokens in enumerate(evaluation):
        losses[variant,index]=float(forward(variant_params,jnp.asarray(tokens[None]),jnp.asarray(controls[variant]),jnp.asarray(means)))
        if variant==0 and index<2:
          delta=float(reference(params,jnp.asarray(tokens[None])))-losses[0,index]
          noop.append(delta);assert abs(delta)<1e-5,(index,delta)
          print('NOOP_CHECK_OK',index,delta,flush=True)
      assert np.isfinite(losses[variant]).all()
      print('VARIANT_DONE',name,float(losses[variant].mean()),'seconds',round(time.monotonic()-before,2),flush=True)
    repeat=float(forward(params,jnp.asarray(evaluation[:1]),jnp.asarray(controls[0]),jnp.asarray(means)))
    assert repeat==losses[0,0]
  np.savez_compressed(args.output/'paired-losses.npz',losses=losses,controls=controls,mean_normalized_addresses=means)
  result=dict(exp=args.exp,checkpoint=args.checkpoint,names=names,
      mean_loss=losses.mean(1).tolist(),mean_delta=(losses-losses[0]).mean(1).tolist(),
      calibration_sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in calibration],
      evaluation_sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in evaluation],
      calibration_sequences=16,evaluation_sequences=16,length=args.length,high_layers=list(range(24,32)),
      noop_deltas=noop,baseline_repeat_exact=True,traces=traces,
      runtime_commit=__import__('subprocess').check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,
      semantics='Native JAX FP32/highest; parameters unchanged except selected Q/K up kernels zero and pre-biases replaced by disjoint-calibration mean normalized addresses. Gate compensation cancels fixed-key normalization: effective dynamic address is original sigmoid gate times stored mean address, not unit-normalized mean. Full MHA and original gate dependence retained.')
  (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('NATIVE_STATIC_ADDRESS_COMPLETE',result['mean_delta'],flush=True)


if __name__=='__main__':main()

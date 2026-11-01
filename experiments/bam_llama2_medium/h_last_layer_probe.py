"""Paired native-JAX interventions of final context/local O over training."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def intervention_grid(layers):
  import numpy as np
  names=['learned','last_context_off','last_local_o_off','last_both_off',
         'middle_context_off','penultimate_context_off','last_local_o_half',
         'last_local_o_half_context_off']
  controls=np.ones((len(names),layers,2),np.float32)
  controls[1,-1,0]=0;controls[2,-1,1]=0;controls[3,-1]=0
  controls[4,layers//2,0]=0;controls[5,-2,0]=0
  controls[6,-1,1]=.5;controls[7,-1]=[0,.5]
  return names,controls


def context_intervention_grid(layers):
  """One baseline and one residual-context cut for every layer."""
  import numpy as np
  controls=np.ones((layers+1,layers,2),np.float32)
  controls[1+np.arange(layers),np.arange(layers),0]=0
  return ['learned']+[f'layer_{i:02d}_context_off' for i in range(layers)],controls


@contextlib.contextmanager
def intervene(scales,record=False):
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
    def attention_call(*a,**kw):
      result=attention(*a,**kw);state['context']=result[0];return result
    def local_call(self,value,deterministic):
      result=local(self,value,deterministic);state['local']=result;return result
    def projection_call(self,output_dim,value):
      c=state['context'];o=state['local'];assert c.shape==o.shape==value.shape
      # Intervene only at W_O input. Original M-write content stays untouched.
      modified=scales[layer,0]*c+scales[layer,1]*o
      selected=jnp.where(jnp.all(scales[layer]==1),value,modified)
      if record and not self.is_initializing():
        mean=lambda x:jnp.mean(x.astype(jnp.float32)**2)
        self.sow('intermediates','last_path_head_probe',jnp.stack((mean(c),mean(o),mean(value),mean(selected),mean(value-c-o))))
      return projection(self,output_dim,selected)
    attentions._attention_op=attention_call
    attentions.BamAttention._regularize_local_o_heads=local_call
    attentions.BamAttention.out_projection=projection_call
    try:
      result=next_fun(*args,**kwargs)
      if record and not module.is_initializing() and len(result)==3 and result[2] is not None:
        module.sow('intermediates','last_path_write_probe',jnp.mean(result[2][0].astype(jnp.float32)**2))
      return result
    finally:
      attentions._attention_op=attention
      attentions.BamAttention._regularize_local_o_heads=local
      attentions.BamAttention.out_projection=projection
  with nn.intercept_methods(intercept):yield


def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ['runtime-dir','output']:p.add_argument('--'+name,type=Path,required=True)
  for name in ['exp','checkpoint-root','steps','cohort']:p.add_argument('--'+name,required=True)
  p.add_argument('--all-context-layers',action='store_true')
  p.add_argument('--reuse-result-dir',type=Path)
  args=p.parse_args();assert args.reuse_result_dir is None or args.all_context_layers
  os.chdir(args.runtime_dir);sys.path.insert(0,str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils,pyconfig
  from layers.models import Transformer
  from h_write_balance_probe import restore_params
  assert jax.process_count()==1
  (args.output/'configuration/last-path').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=args.exp,run_name='last-path',
      base_output_directory=str(args.output/'configuration')+'/',enable_checkpointing=False,
      only_eval=True,dataset_type='synthetic',target_train_batch_size=None,per_device_batch_size=1.,
      max_target_length=2048,max_prefill_predict_length=2048,
      record_internal_nn_metrics=False,record_training_health_metrics=False,jax_cache_dir='',log_config=False)
  cfg.get_keys().update(dtype=jnp.float32,bam_pallas_core=False,bam_pallas_write=False,
                       bam_splash_attention=False,matmul_precision='highest')
  assert cfg.num_decoder_layers==32 and 31 not in cfg.bam_attention_write_layers
  assert cfg.bam_skip_final_mlp_write
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
  cohort=np.load(args.cohort)['tokens'][:32,:2049];assert cohort.shape==(32,2049)
  positions=jnp.arange(2048,dtype=jnp.int32)[None];segments=jnp.ones((1,2048),jnp.int32)
  names,controls=(context_intervention_grid(32) if args.all_context_layers else intervention_grid(32));traces=0
  def apply(params,tokens):
    return model.apply(params,tokens[:,:-1],positions,decoder_segment_ids=segments,
        decoder_target_mask=segments,decoder_target_tokens=tokens[:,1:],enable_dropout=False)
  def forward(params,tokens,scales):
    nonlocal traces
    traces+=1;print('FORWARD_TRACE',traces,flush=True)
    with intervene(scales):xent,_,_=apply(params,tokens)
    return jnp.mean(xent.astype(jnp.float32))
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),jax.default_matmul_precision('highest'):
    compiled=jax.jit(forward);reference=jax.jit(lambda params,tokens:jnp.mean(apply(params,tokens)[0].astype(jnp.float32)))
    for step in map(int,args.steps.split(',')):
      start=time.monotonic();checkpoint=args.checkpoint_root.rstrip('/')+f'/{step}/items'
      print('RESTORE_BEGIN',step,checkpoint,flush=True)
      params=jax.device_put(unfreeze(nn.unbox(restore_params(checkpoint))),jax.sharding.SingleDeviceSharding(jax.devices()[0]));jax.block_until_ready(params)
      print('RESTORE_OK',step,round(time.monotonic()-start,2),flush=True)
      losses=np.empty((len(names),32),np.float64);noop=[];reused={};reuse_losses=None;reuse_meta=None
      if args.reuse_result_dir is not None:
        reuse_path=args.reuse_result_dir/f'step-{step}.npz'
        reuse_losses=np.load(reuse_path)['losses'];reuse_meta=json.loads(reuse_path.with_suffix('.json').read_text())
        assert reuse_losses.shape==(8,32) and np.isfinite(reuse_losses).all()
        assert reuse_meta['exp']==args.exp and reuse_meta['checkpoint']==checkpoint
        assert reuse_meta['runtime_commit']==subprocess.check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip()
        assert reuse_meta['sequence_hashes']==[hashlib.sha256(x.tobytes()).hexdigest() for x in cohort]
        reused={f'layer_{layer:02d}_context_off':reuse_meta['names'].index(label) for layer,label in [(16,'middle_context_off'),(30,'penultimate_context_off'),(31,'last_context_off')]}
      for variant,name in enumerate(names):
        if name in reused:
          losses[variant]=reuse_losses[reused[name]];print('VARIANT_REUSED',step,name,flush=True);continue
        before=time.monotonic()
        for index,tokens in enumerate(cohort):
          losses[variant,index]=float(compiled(params,jnp.asarray(tokens[None]),jnp.asarray(controls[variant])))
          if variant==0 and index<2:
            delta=float(reference(params,jnp.asarray(tokens[None])))-losses[0,index]
            noop.append(delta);assert abs(delta)<1e-6,(step,index,delta)
            print('NOOP_CHECK_OK',step,index,delta,flush=True)
        assert np.isfinite(losses[variant]).all()
        if variant==0 and reuse_losses is not None:
          np.testing.assert_allclose(losses[0],reuse_losses[0],rtol=0,atol=1e-6)
          print('PREVIOUS_BASELINE_MATCH_OK',step,float(np.max(np.abs(losses[0]-reuse_losses[0]))),flush=True)
        print('VARIANT_DONE',step,name,float(losses[variant].mean()),'seconds',round(time.monotonic()-before,2),flush=True)
      assert float(compiled(params,jnp.asarray(cohort[:1]),jnp.asarray(controls[0])))==losses[0,0]
      np.savez_compressed(args.output/f'step-{step}.npz',losses=losses,controls=controls)
      result=dict(exp=args.exp,checkpoint=checkpoint,step=step,names=names,mean_loss=losses.mean(1).tolist(),
          mean_delta=(losses-losses[0]).mean(1).tolist(),sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in cohort],
          sequences=32,length=2048,noop_deltas=noop,baseline_repeat_exact=True,forward_traces=traces,
          runtime_commit=subprocess.check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start,
          backend='Native JAX FP32/highest; unchanged checkpoint values, Pallas/Splash OFF.',
          reused_variants=list(reused),reuse_result_dir=str(args.reuse_result_dir) if args.reuse_result_dir else None,
          reuse_baseline_max_error=float(np.max(np.abs(losses[0]-reuse_losses[0]))) if reuse_losses is not None else None,
          semantics='Residual-path intervention at W_O input, bias preserved. General sweep cuts one layer at a time. Last31 has no M writes. Original attention M-write content is preserved at the cut layer; current private MLP and downstream paths react normally. Half-O includes complete static+dynamic localO. Frozen-checkpoint necessity/interaction, not retraining benefit.')
      (args.output/f'step-{step}.json').write_text(json.dumps(result,indent=2)+'\n');print('STEP_DONE',step,result['mean_delta'],flush=True)
if __name__=='__main__':main()

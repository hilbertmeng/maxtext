"""Read-only native-FP32 context/local-O amplitude probe on exact BAM runtimes."""
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

FIELDS=('context_ms','local_o_ms','sum_ms','context_local_cross')

@contextlib.contextmanager
def capture_context():
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions
  def intercept(next_fun,args,kwargs,ctx):
    module=ctx.module
    if ctx.method_name!='__call__' or not isinstance(module,attentions.BamAttention):
      return next_fun(*args,**kwargs)
    attention=attentions._attention_op
    local=attentions.BamAttention._regularize_local_o_heads
    projection=attentions.BamAttention.out_projection
    state={}
    def attention_call(*args,**kwargs):
      result=attention(*args,**kwargs);state['context']=result[0];return result
    def local_call(self,value,deterministic):
      result=local(self,value,deterministic);state['local']=result;return result
    def projection_call(self,output_dim,value):
      if not self.is_initializing():
        c=state['context'].astype(jnp.float32);o=state['local'].astype(jnp.float32);s=value.astype(jnp.float32)
        assert c.shape==o.shape==s.shape,(c.shape,o.shape,s.shape)
        mean=lambda x:jnp.mean(x,axis=tuple(range(1,x.ndim)))
        self.sow('intermediates','context_balance_probe',jnp.stack((mean(c*c),mean(o*o),mean(s*s),mean(c*o)),axis=-1))
      return projection(self,output_dim,value)
    attentions._attention_op=attention_call
    attentions.BamAttention._regularize_local_o_heads=local_call
    attentions.BamAttention.out_projection=projection_call
    try:return next_fun(*args,**kwargs)
    finally:
      attentions._attention_op=attention
      attentions.BamAttention._regularize_local_o_heads=local
      attentions.BamAttention.out_projection=projection
  with nn.intercept_methods(intercept):yield

def extract(collection,final_layer):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  rows=[]
  for path,value in flatten_dict(collection).items():
    if path[-1]!='context_balance_probe':continue
    while isinstance(value,(tuple,list)):value=value[0]
    if 'layers' in path:rows.append((0,value))
    elif 'final_local_layer' in path:rows.append((final_layer,value[None]))
    else:
      match=next((re.fullmatch(r'layers_(\d+)',p) for p in path if re.fullmatch(r'layers_(\d+)',p)),None)
      if match:rows.append((int(match[1]),value[None]))
  if not rows:raise ValueError('No context statistics captured')
  return jnp.concatenate([v for _,v in sorted(rows)],axis=0)

def main():
  p=argparse.ArgumentParser(description=__doc__)
  for name in ['runtime-dir','output']:p.add_argument('--'+name,type=Path,required=True)
  for name in ['exp','checkpoint-root','steps','cohort']:p.add_argument('--'+name,required=True)
  args=p.parse_args();os.chdir(args.runtime_dir);sys.path.insert(0,str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils,pyconfig
  from layers.models import Transformer
  from h_write_balance_probe import restore_params
  assert jax.process_count()==1
  (args.output/'configuration/context-balance').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=args.exp,run_name='context-balance',base_output_directory=str(args.output/'configuration')+'/',enable_checkpointing=False,only_eval=True,dataset_type='synthetic',target_train_batch_size=None,per_device_batch_size=1.,max_target_length=2048,max_prefill_predict_length=2048,record_internal_nn_metrics=False,record_training_health_metrics=False,jax_cache_dir='',log_config=False)
  cfg.get_keys().update(bam_pallas_read_tile=128,bam_pallas_write_tile=128,bam_pallas_vmem_mib=40)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);steps=list(map(int,args.steps.split(',')));initial=None
  if 0 in steps:
    assert steps[0]==0
    init_model=Transformer(cfg,mesh,quant=None)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):initial=nn.unbox(jax.jit(lambda rng:max_utils.init_initial_state(init_model,None,cfg,False,rng).params)(jax.random.PRNGKey(cfg.init_weights_seed)))
    jax.block_until_ready(initial)
  cfg.get_keys().update(dtype=jnp.float32,bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,matmul_precision='highest')
  model=Transformer(cfg,mesh,quant=None);cohort=np.load(args.cohort)['tokens'][:32,:2049];assert cohort.shape==(32,2049)
  pos=jnp.arange(2048,dtype=jnp.int32)[None];seg=jnp.ones((1,2048),jnp.int32);traces=0
  def apply(params,tokens,mutable=False):
    return model.apply(params,tokens[:,:-1],pos,decoder_segment_ids=seg,decoder_target_mask=seg,decoder_target_tokens=tokens[:,1:],enable_dropout=False,**({'mutable':['intermediates']} if mutable else {}))
  def forward(params,tokens):
    nonlocal traces
    traces+=1;print('FORWARD_TRACE',traces,flush=True)
    with capture_context():result,collections=apply(params,tokens,True)
    return jnp.mean(result[0].astype(jnp.float32)),extract(collections['intermediates'],31)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),jax.default_matmul_precision('highest'):
    compiled=jax.jit(forward);reference=jax.jit(lambda params,tokens:jnp.mean(apply(params,tokens)[0].astype(jnp.float32)))
    for step in steps:
      start=time.monotonic();checkpoint=None if step==0 else args.checkpoint_root.rstrip('/')+f'/{step}/items';print('RESTORE_BEGIN',step,checkpoint,flush=True)
      params=initial if step==0 else restore_params(checkpoint);params=jax.device_put(unfreeze(nn.unbox(params)),jax.sharding.SingleDeviceSharding(jax.devices()[0]));jax.block_until_ready(params)
      rows=[];losses=[];noop=[]
      for index,tokens in enumerate(cohort):
        loss,stats=compiled(params,jnp.asarray(tokens[None]));stats=np.asarray(stats);assert stats.shape==(32,1,4) and np.isfinite(stats).all();rows.append(stats[:,0]);losses.append(float(loss))
        if index<2:
          delta=float(reference(params,jnp.asarray(tokens[None])))-losses[-1];noop.append(delta);assert abs(delta)<1e-5,('No-op mismatch',delta)
        if index in (0,7,31):print('FORWARD_OK',step,index+1,round(time.monotonic()-start,2),flush=True)
      np.savez_compressed(args.output/f'step-{step}.npz',statistics=np.stack(rows),losses=np.array(losses))
      summary=dict(exp=args.exp,step=step,checkpoint=checkpoint,fields=FIELDS,sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in cohort],noop_deltas=noop,forward_traces=traces,seconds=time.monotonic()-start,backend='Native JAX FP32/highest, Pallas/Splash disabled; checkpoint parameter values retained.0k parameters initialized with original formal runtime config before forward overrides.',scope='Context and local O immediately before shared W_O. Amplitudes/correlation, not causal attribution.',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
      (args.output/f'step-{step}.json').write_text(json.dumps(summary,indent=2)+'\n');print('STEP_DONE',step,flush=True)
if __name__=='__main__':main()

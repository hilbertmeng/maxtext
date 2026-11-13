"""Matched early-attention routes and weak residual writes, immutable d54 model.

FP32 route reconstruction is diagnostic only; the real forward retains Splash.
Captured arrays cover two fixed sequences and twelve queries; not all attention
rows. Full-batch write amplitudes are reduced inside JIT. No optimizer writes.
"""
from pathlib import Path
import runpy

SEQUENCES = (9, 104)
QUERIES = (0, 1, 2, 8, 16, 128, 512, 1211, 1220, 2048, 3072, 4095)


def runtime_modules():
  # The immutable d54 runtime predates the bam_kernels rename.
  from layers import attentions, fusion, bam_pallas
  return attentions, fusion, bam_pallas


def parameter_plans():
  p = [dict(name='peak_control', families=[], direction='revert'),
       dict(name='old_control', families=['ALL'], direction='revert')]
  for label, families in [('QK', ['attention_standard_qk','attention_local_qk']),
                           ('V', ['attention_local_v']), ('WO', ['attention_output_projection']),
                           ('V_WO', ['attention_local_v','attention_output_projection']),
                           ('all', ['ATTENTION'])]:
    p.append(dict(name='revert_L00_05_'+label, families=families, direction='revert', layer=list(range(6))))
  return p


def routing_probabilities(q, k, segments, queries):
  import jax
  import jax.numpy as jnp
  positions=jnp.asarray(queries)
  logits=jnp.einsum('bqhd,bshd->bhqs',q.astype(jnp.float32),k.astype(jnp.float32))
  valid=(jnp.arange(k.shape[1])[None,:] < positions[:,None])[None,None]
  valid=valid & (segments[:,None,positions,None] == segments[:,None,None,:])
  valid=valid & (segments[:,None,positions,None] != 0)
  weights=jax.nn.softmax(jnp.where(valid,logits,-1e30),axis=-1)
  return jnp.where(valid,weights,0.)


def write_reference(content, logits, address, epsilon, divisor=None):
  import jax
  import jax.numpy as jnp
  c=content.astype(jnp.float32);a=address.astype(jnp.float32)
  c=c/jnp.sqrt(jnp.mean(c*c,axis=-1,keepdims=True)+epsilon) if divisor is None else c/divisor
  a=a/jnp.sqrt(jnp.mean(a*a,axis=-1,keepdims=True)+epsilon)
  c=(c*jax.nn.sigmoid(logits.astype(jnp.float32))[...,None]).astype(content.dtype).astype(jnp.float32)
  a=a.astype(address.dtype).astype(jnp.float32)
  return jnp.einsum('bqhv,bqhk->bqvk',a,c)


def run_probe(train,captured,reference,output,sequence,target_layer):
  import hashlib,json,os,subprocess,time
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn,serialization
  attentions,fusion,bam_kernels=runtime_modules()
  helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
  helper['run_probe'].__globals__['case_plans']=parameter_plans
  helper['run_probe'](train,captured,reference,output,sequence,target_layer)
  aot=json.loads((output/'loss-localization.json').read_text())
  cfg,mesh,state=captured['config'],captured['mesh'],captured['state']
  model,_,shardings=captured['functional_train'].args
  assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full' and cfg.bam_pallas_core
  assert jax.device_count()==16 and jax.process_count()==4 and cfg.global_batch_size_to_train_on==128
  source=os.environ['BAM_DIAG_REFERENCE_CHECKPOINT']
  abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
  cp=ocp.PyTreeCheckpointer();raw=cp.restore(source,item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
  old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
  classifier=runpy.run_path(str(Path(__file__).with_name('xlprop_parameter_family_probe.py')))['parameter_family']
  kernels=[(path,x) for path,x in jax.tree_util.tree_flatten_with_path(old)[0] if classifier(path,True,True)=='attention_output_projection' and x.ndim==4 and x.shape[1]==27]
  assert len(kernels)==1,[(str(p),x.shape) for p,x in kernels]
  old_wo=kernels[0][1]
  peak_wo=[x for path,x in jax.tree_util.tree_flatten_with_path(state.params)[0] if classifier(path,True,True)=='attention_output_projection' and x.ndim==4 and x.shape[1]==27][0]
  layer_stack=[];attention_stack=[];fusion_stack=[];residual_stack=[];active_routes=None;active_valid=None
  original_splash=attentions._bam_splash_attention;original_write=bam_kernels.write
  si=jnp.asarray(SEQUENCES);qi=jnp.asarray(QUERIES)
  def sample(x):return x[si[:,None],qi[None,:]]
  def rms(x,axes=-1):return jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2,axis=axes))
  def amplitude(x):
    vals=rms(x,tuple(range(2,x.ndim)))
    valid=active_valid.astype(jnp.float32);denom=jnp.maximum(valid.sum(),1.)
    return jnp.stack((jnp.sqrt(jnp.sum(vals**2*valid)/denom),jnp.sum(vals*valid)/denom,jnp.max(jnp.where(valid>0,vals,0.))))
  def splash(q,k,v,segments,config,mesh,window=None,seq_minor=False):
    assert window is None or window==k.shape[1], (window,k.shape)
    y=original_splash(q,k,v,segments,config,mesh,window=window,seq_minor=seq_minor)
    layer=jnp.asarray(layer_stack[-1]);module=attention_stack[-1]
    def calculate(_):
      qs=q[si[:,None],qi[None,:]];ks=k[si];vs=v[si]
      weights=routing_probabilities(qs,ks,segments[si],QUERIES)
      context=jnp.einsum('bhqs,bshd->bqhd',weights,vs.astype(jnp.float32))
      fixed=jnp.einsum('bhqs,bshd->bqhd',active_routes[jnp.minimum(layer,5)],vs.astype(jnp.float32))
      return weights,context,fixed
    weights,context,fixed=jax.lax.cond(layer<6,calculate,lambda _:(jnp.zeros((2,cfg.num_query_heads,len(QUERIES),4096),jnp.float32),jnp.zeros((2,len(QUERIES),cfg.num_query_heads,cfg.head_dim),jnp.float32),jnp.zeros((2,len(QUERIES),cfg.num_query_heads,cfg.head_dim),jnp.float32)),None)
    for name,x in [('route',weights),('context_reconstructed',context),('context_fixed_oldP',fixed),('context_native',sample(y)),('V_queries',sample(v)),('V_key_rms',rms(v[si],-1)),('Q_queries',sample(q))]:module.sow('intermediates','trace_'+name,x)
    module.sow('intermediates','trace_V_amplitude',amplitude(v))
    return y
  def write(m,groups,**kwargs):
    result=original_write(m,groups,**kwargs);module=fusion_stack[-1]
    assert len(groups)==2,('Expected separate attention and memory-MLP write groups',len(groups))
    divisors=kwargs.get('content_divisors',(None,None))
    for index,label in enumerate(('attention','memory_mlp')):
      content,logits,address=groups[index]
      dm=write_reference(sample(content),sample(logits),sample(address),kwargs['epsilon'],divisors[index])
      module.sow('intermediates','trace_'+label+'_dM_reference',dm)
    return result
  def intercept(next_fun,args,kwargs,context):
    module=context.module;method=context.method_name
    if isinstance(module,fusion.SubDecoderLayer) and method=='__call__':
      layer=kwargs.get('layer_index',module.layer_inx);layer=module.layer_inx if layer is None else layer
      layer_stack.append(layer);fusion_stack.append(module);residual_stack.append(args[0])
      try:
        module.sow('intermediates','trace_residual_in',sample(args[0]));module.sow('intermediates','trace_M_in',sample(kwargs['M_in']))
        result=next_fun(*args,**kwargs)
        module.sow('intermediates','trace_residual_out',sample(result[0]));module.sow('intermediates','trace_M_out',sample(result[1]))
        return result
      finally:layer_stack.pop();fusion_stack.pop();residual_stack.pop()
    if isinstance(module,attentions.BamAttention) and method=='__call__':
      attention_stack.append(module)
      try:
        result=next_fun(*args,**kwargs)
        x=residual_stack[-1];update=result[0];post=x+update
        xr=rms(x);ur=rms(update);ratio=ur/jnp.maximum(xr,1e-30)
        rounding=(post.astype(jnp.float32)-x.astype(jnp.float32))-update.astype(jnp.float32)
        valid=active_valid.astype(jnp.float32);denom=jnp.maximum(valid.sum(),1.)
        avg=lambda a:jnp.sum(a*valid)/denom
        stats=jnp.concatenate((amplitude(x),amplitude(update),jnp.stack((avg(ratio),jnp.max(jnp.where(valid>0,ratio,0.)),avg(ratio<.001),avg(ratio<.01),avg(ratio<.1),jnp.sqrt(jnp.sum(rounding**2*valid[...,None])/jnp.maximum(jnp.sum(update.astype(jnp.float32)**2*valid[...,None]),1e-30)),jnp.sum(((post==x)&(update!=0))*valid[...,None])/(denom*x.shape[-1])))))
        module.sow('intermediates','trace_residual_write_stats',stats);module.sow('intermediates','trace_residual_after_attention',sample(post))
        return result
      finally:attention_stack.pop()
    if isinstance(module,attentions.BamAttention) and method=='out_projection':
      head=args[1];result=next_fun(*args,**kwargs);h=sample(head).astype(jnp.float32);layer=jnp.minimum(jnp.asarray(layer_stack[-1]),26)
      for name,x in [('head',h),('projection_native',sample(result)),('projection_oldW',jnp.einsum('bqhd,hdo->bqo',h,jnp.take(old_wo,layer,axis=1).astype(jnp.float32))),('projection_peakW',jnp.einsum('bqhd,hdo->bqo',h,jnp.take(peak_wo,layer,axis=1).astype(jnp.float32)))]:module.sow('intermediates','trace_'+name,x)
      return result
    return next_fun(*args,**kwargs)
  # Reference W_O tensors are explicit JIT arguments, never embedded constants.
  def objective(params,batch,rng,ref_routes,ref_old_wo,ref_peak_wo):
    nonlocal active_routes,old_wo,peak_wo,active_valid
    active_routes=ref_routes;old_wo=ref_old_wo;peak_wo=ref_peak_wo;active_valid=batch["inputs_segmentation"]!=0
    with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
    body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
    values={k.removeprefix('trace_'):v[0][:6] for k,v in body.items() if k.startswith('trace_')}
    values.update({k.removeprefix('trace_'):v[0][:6] for k,v in body['self_attention'].items() if k.startswith('trace_')})
    return loss,values
  replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
  data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  compiled=jax.jit(objective,in_shardings=(shardings.params,data_sharding,None,None,None,None),out_shardings=replicated)
  ref_routes=jnp.zeros((6,2,cfg.num_query_heads,len(QUERIES),4096),jnp.float32)
  # Preserve concrete arrays: objective's Python trace temporarily rebinds these names.
  wo_old_array,wo_peak_array=old_wo,peak_wo
  result=dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,old_checkpoint=source,peak_checkpoint_step=35675,sequences=SEQUENCES,queries=QUERIES,layers=list(range(6)),pattern_scope='Two fixed sequences, twelve queries, all source keys; FP32 strict-causal reconstruction, not exact Splash probabilities. Fullbatch residual/V amplitude stats separately.',write_reference_scope='Sampled attention/memory group outer products after FP32 normalization and model-dtype factor rounding; reference decomposition, compare with actual total M delta; not independent native Pallas output.',residual_write_stats=['residual_global_rms','residual_mean_token_rms','residual_max_token_rms','write_global_rms','write_mean_token_rms','write_max_token_rms','mean_write_over_residual','max_write_over_residual','fraction_ratio_lt_1e-3','fraction_ratio_lt_1e-2','fraction_ratio_lt_1e-1','bf16_addition_error_relative_energy_sqrt','nonzero_write_elements_rounded_away_fraction'],input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest()) for x in captured['data']['inputs'].addressable_shards],cases=[])
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  def publish():
    p=output/'pattern-write.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/pattern-write.json'],check=True,timeout=90)
  def params_for(plan):
    if plan['name']=='old_control':return old
    if plan['name']=='peak_control':return state.params
    wanted=set(plan['families'])
    def transplant(path,before,after):
      names=tuple(getattr(k,'key',None) for k in path);label=classifier(path,True,True)
      if (label.startswith('attention_') if 'ATTENTION' in wanted else label in wanted) and 'layers' in names:
        assert after.shape[1]==27
        index=(slice(None),jnp.arange(6))+(slice(None),)*(after.ndim-2)
        return after.at[index].set(before[index])
      return after
    return jax.tree.map_with_path(transplant,old,state.params)
  attentions._bam_splash_attention=splash;bam_kernels.write=write
  try:
    plans=parameter_plans();plans=[plans[1],plans[0]]+plans[2:]
    for plan in plans:
      start=time.monotonic();params=params_for(plan)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(params,captured['data'],captured['rng'],ref_routes,wo_old_array,wo_peak_array)
      loss,values=jax.device_get((loss,values));expected=next(c['metrics']['learning/loss'] for c in aot['cases'] if c['name']==plan['name'])
      error=abs(float(loss)-expected);assert error<1e-3,('Forward fidelity failed',plan,error)
      assert all(np.isfinite(np.asarray(v)).all() for v in values.values()),plan
      entry=dict(name=plan['name'],loss=float(loss),original_aot_loss=expected,loss_abs_error=error,fidelity_passed=True,seconds=time.monotonic()-start,residual_write_stats=np.asarray(values['residual_write_stats']).tolist(),V_amplitude=np.asarray(values['V_amplitude']).tolist())
      if jax.process_index()==0:
        dest=output/(plan['name']+'-pattern-write.npz');np.savez_compressed(dest,**{k:np.asarray(v,dtype=np.float32) for k,v in values.items()})
        subprocess.run(['gsutil','-q','cp',str(dest),upload+'/'+dest.name],check=True,timeout=180)
      if plan['name']=='old_control':ref_routes=jax.device_put(values['route'],replicated)
      result['cases'].append(entry);publish()
      if jax.process_index()==0:print('PATTERN_WRITE_CASE',plan['name'],float(loss),error,flush=True)
    result['complete']=True;publish()
  finally:attentions._bam_splash_attention=original_splash;bam_kernels.write=original_write

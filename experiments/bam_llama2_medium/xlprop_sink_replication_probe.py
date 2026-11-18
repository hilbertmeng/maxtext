"""Three input batches, two fixed checkpoints; retained-host sink replication.

Original batch native controls stay hard gated. New batches use fresh original
AOT controls. No optimizer update from either control or capture is retained.
"""
from pathlib import Path
import runpy
SEQUENCES=(9,40,73,104)
QUERIES=(0,1,2,8,16,128,512,1211,2048,3072,4095)

def select_batches(captured):
 samples=[dict(step=35676,data=captured['data'],rng=captured['rng'])]
 fresh=captured['frozen_batches'];assert [x['step']for x in fresh]==[35677,35678]
 samples.extend(dict(step=x['step'],data=x['data'],rng=x['rng'])for x in fresh)
 return samples

def run_probe(train,captured,reference,output,sequence,target_layer):
 import os,json,hashlib,subprocess
 import jax,jax.numpy as jnp,numpy as np
 import orbax.checkpoint as ocp
 from flax import linen as nn,serialization
 from layers import attentions,bam_pallas
 base=runpy.run_path(str(Path(__file__).with_name('xlprop_pattern_write_probe.py')))
 route=runpy.run_path(str(Path(__file__).with_name('xlprop_sink_route_probe.py')))
 helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
 upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
 cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
 assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full' and cfg.bam_pallas_body=='v8t'
 assert not cfg.qk_norm and cfg.num_decoder_layers==28 and cfg.num_query_heads==20 and jax.device_count()==16 and jax.process_count()==4
 abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
 cp=ocp.PyTreeCheckpointer();raw=cp.restore(os.environ['BAM_DIAG_REFERENCE_CHECKPOINT'],item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
 old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
 samples=select_batches(captured);captured.pop('frozen_batches')
 copy=lambda x:jax.tree.map(lambda y:y.copy(),x)
 si=jnp.array(SEQUENCES);qi=jnp.array(QUERIES);sample=lambda x:x[si[:,None],qi[None,:]]
 attns=[];original_read=bam_pallas.read;original_splash=attentions._bam_splash_attention
 def record(module,name,x):module.sow('intermediates','rep_'+name,x)
 def read(*args,**kw):
  result=original_read(*args,**kw);module=attns[-1]
  gates=jnp.stack([sample(x)for x in(args[3],args[5],args[8],args[7])],axis=2)
  static=jnp.stack([sample(x)for x in kw['static_gates']],axis=2)
  record(module,'read_gate_logits',gates);record(module,'read_static_gates',static)
  return result
 def splash(q,k,v,segments,config,mesh,window=None,seq_minor=False):
  y=original_splash(q,k,v,segments,config,mesh,window=window,seq_minor=seq_minor)
  module=attns[-1];qs=sample(q);ks=k[si];vs=v[si]
  cf=route['route_counterfactuals'](qs,ks,segments[si],QUERIES,base['routing_probabilities'],96)
  for name,p in cf.items():record(module,'cf_'+name+'_p_first',p[...,0])
  ctx=jnp.einsum('bhqs,bshd->bqhd',cf['full'],vs.astype(jnp.float32));native=sample(y).astype(jnp.float32)
  record(module,'context_error_sums',jnp.stack([jnp.sum((ctx-native)**2),jnp.sum(native**2)]))
  for name,x in [('Q',qs),('K',sample(k)),('V',sample(v))]:record(module,name,x)
  return y
 def intercept(next_fun,args,kwargs,context):
  module=context.module
  if isinstance(module,attentions.BamAttention) and context.method_name=='__call__':
   attns.append(module)
   try:return next_fun(*args,**kwargs)
   finally:attns.pop()
  return next_fun(*args,**kwargs)
 def objective(params,batch,rng):
  with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
  body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']['self_attention']
  vals={k[4:]:v[0]for k,v in body.items()if k.startswith('rep_')};vals['sample_tokens']=sample(batch['inputs']);vals['sample_segments']=sample(batch['inputs_segmentation'])
  return loss,vals
 replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
 compiled=jax.jit(objective,in_shardings=(shardings.params,data_sharding,None),out_shardings=replicated)
 result=dict(runtime='d54c4c5f1339b01e406932ec28be0e398722b918',fixed_peak_checkpoint=35675,fixed_old_checkpoint=35635,sequences=SEQUENCES,queries=QUERIES,scope='Same preselected heads and four sequence indices over three batches; freeze checkpoint parameters. No head reselection or updates retained.',batches=[])
 bam_pallas.read=read;attentions._bam_splash_attention=splash
 try:
  for item in samples:
   step=item['step'];batch=item['data'];rng=item['rng'];record_batch=dict(step=step,input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest())for x in batch['inputs'].addressable_shards],cases=[])
   for name,params in [('old_control',old),('peak_control',state.params)]:
    # Original serialized train step provides the fresh per-batch scientific control.
    updated,metrics=captured['compiled'](copy(state).replace(params=copy(params)),batch,rng);del updated
    native={k:float(np.asarray(x))for k,x in jax.device_get(metrics['scalar']).items()if k in ('learning/loss','learning/raw_grad_norm')}
    assert all(np.isfinite(x)for x in native.values())
    if step==35676:
     expected={'old_control':(1.9770115613937378,.21527354419231415),'peak_control':(2.595792531967163,809414.5625)}[name]
     assert all(abs(native[k]-v)<=1e-5*max(1,abs(v))for k,v in zip(('learning/loss','learning/raw_grad_norm'),expected)),native
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(params,batch,rng)
    loss,values=jax.device_get((loss,values));err=abs(float(loss)-native['learning/loss']);assert err<1e-3,(step,name,err)
    assert all(np.isfinite(x).all()for x in values.values())
    sums=np.asarray(values['context_error_sums']);rel=np.sqrt(sums[:,0]/sums[:,1]);assert np.max(rel)<.005,(step,name,rel)
    entry=dict(name=name,native=native,capture_loss=float(loss),forward_error=err,max_layer_context_error=float(rel.max()))
    if jax.process_index()==0:
     dest=output/f'batch{step}-{name}.npz';np.savez_compressed(dest,**{k:np.asarray(v)for k,v in values.items()});entry.update(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest());subprocess.run(['gsutil','-q','cp',str(dest),upload+'/'+dest.name],check=True,timeout=300)
    record_batch['cases'].append(entry)
   result['batches'].append(record_batch)
   (output/'replication-local.json').write_text(json.dumps(result,indent=2)+'\n')
   if jax.process_index()==0:(output/'replication.json').write_text(json.dumps(result,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(output/'replication.json'),upload+'/replication.json'],check=True,timeout=90)
  result['complete']=True;(output/'replication-local.json').write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:(output/'replication.json').write_text(json.dumps(result,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(output/'replication.json'),upload+'/replication.json'],check=True,timeout=90)
 finally:bam_pallas.read=original_read;attentions._bam_splash_attention=original_splash

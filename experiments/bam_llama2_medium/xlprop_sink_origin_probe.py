"""All scanned layers: frozen sink geometry and M-write provenance, d54 only.

Observational capture; no training updates are retained. Raw native Q/K/V and
M states are preserved alongside FP32 descriptive reconstructions. Source shares
are conditional fixed-read decompositions, never additive causal percentages.
"""
from pathlib import Path
import runpy
SEQUENCES=(9,40,73,104)
QUERIES=(0,1,2,8,16,128,512,1211,2048,3072,4095)

def split_scores(q,k,cols):
 import jax.numpy as jnp
 return (jnp.einsum('bqhd,bshd->bhqs',q[...,:cols].astype(jnp.float32),k[...,:cols].astype(jnp.float32)),
         jnp.einsum('bqhd,bshd->bhqs',q[...,cols:].astype(jnp.float32),k[...,cols:].astype(jnp.float32)))

def effective_keys(raw,logits,static,static_gates,eps,key_scale):
 import jax,jax.numpy as jnp
 dynamic=raw.astype(jnp.float32)*jax.lax.rsqrt(jnp.mean(raw.astype(jnp.float32)**2,-1,keepdims=True)+eps)
 dynamic=dynamic*(key_scale*jax.nn.sigmoid(logits.astype(jnp.float32)))[...,None]
 return dynamic+static[None,None]*static_gates[...,None]

def centered_gram(w):
 import jax.numpy as jnp
 w=w.astype(jnp.float32);s=jnp.sum(w,axis=1)
 return w@w.T-jnp.outer(s,s)/w.shape[1]

def run_probe(train,captured,reference,output,sequence,target_layer):
 import os,json,time,hashlib,subprocess
 import jax,jax.numpy as jnp,numpy as np
 import orbax.checkpoint as ocp
 from flax import linen as nn,serialization
 from layers import attentions,fusion,bam_pallas
 base=runpy.run_path(str(Path(__file__).with_name('xlprop_pattern_write_probe.py')))
 helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
 plans=lambda:[dict(name='peak_control',families=[],direction='revert'),dict(name='old_control',families=['ALL'],direction='revert')]
 helper['run_probe'].__globals__['case_plans']=plans
 helper['run_probe'](train,captured,reference,output,sequence,target_layer)
 native=json.loads((output/'loss-localization.json').read_text())
 cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
 assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full' and cfg.bam_pallas_body=='v8t'
 assert cfg.num_decoder_layers==28 and cfg.num_query_heads==20 and jax.device_count()==16 and jax.process_count()==4
 source=os.environ['BAM_DIAG_REFERENCE_CHECKPOINT']
 abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
 cp=ocp.PyTreeCheckpointer();raw=cp.restore(source,item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
 old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
 si=jnp.array(SEQUENCES);qi=jnp.array(QUERIES)
 sample=lambda x:x[si[:,None],qi[None,:]]
 layers=[];attns=[];blocks=[];current_v=[];active_seg=None
 original_read=bam_pallas.read;original_write=bam_pallas.write;original_splash=attentions._bam_splash_attention
 def record(module,name,value):module.sow('intermediates','sink_'+name,value)
 def read(*args,**kw):
  result=original_read(*args,**kw);module=attns[-1]
  m,sw,q,lq,k,lk,vo,lo,lv,qs,ks=args
  assert m.shape[1]==4096 and m.ndim==4 and kw['body']=='v8t'
  if kw.get('keys_cnt') is not None:
   keys=sample(jnp.transpose(kw['keys_cnt'],(0,3,2,1)));assert keys.shape[-2] in (60,80)
   keys=keys.reshape(keys.shape[:2]+(-1,20,keys.shape[-1]))
  else:
   vkey,okey=jnp.split(vo,2,axis=2) if vo.shape[2]==40 else (vo,vo)
   keys=jnp.stack([sample(x)for x in(q,k,vkey,okey)],axis=2)
  if keys.shape[2]==3:keys=jnp.concatenate((keys[:,:,:2],keys[:,:,2:3],keys[:,:,2:3]),axis=2)
  gates=jnp.stack([sample(x)for x in(lq,lk,lv,lo)],axis=2)
  sg=jnp.stack([sample(x)for x in kw['static_gates']],axis=2) if kw.get('static_gates') else jnp.ones(gates.shape,jnp.float32)
  # Native static_gates are already activation-scaled, not raw logits.
  for name,x in [('M_read_input',sample(m)),('read_raw_keys',keys),('read_gate_logits',gates),('read_static_gates',sg),('read_static_weights',sw),('read_outputs',jnp.stack([sample(x)for x in result[:4]],axis=2))]:record(module,name,x)
  for arm in ('q','k','v','o'):
   scale=kw.get(arm+'_dynamic_scale');record(module,'read_'+arm+'_scale',jnp.ones((20,m.shape[2]),jnp.float32) if scale is None else jnp.broadcast_to(scale[:,None] if scale.ndim==1 else scale,(20,m.shape[2])))
  record(module,'read_constants',jnp.array([kw['read_epsilon'],kw['key_scale'],kw['q_scale'],kw.get('m_norm') or 0,kw['qk_cols'],bool(kw.get('qk_norm'))]))
  if kw.get('shift') is not None:record(module,'shift_gate',sample(kw['shift']))
  return result
 def splash(q,k,v,segments,config,mesh,window=None,seq_minor=False):
  y=original_splash(q,k,v,segments,config,mesh,window=window,seq_minor=seq_minor)
  assert window is None or window==4096
  module=attns[-1];qs=sample(q);ks=k[si];vs=v[si]
  p=base['routing_probabilities'](qs,ks,segments[si],QUERIES)
  recon=jnp.einsum('bhqs,bshd->bqhd',p,vs.astype(jnp.float32));scores=split_scores(qs,ks,int(module._qk_col_width))
  # Retain each sampled query, with all20heads; source axis reduced on TPU.
  for name,x in [('Q',qs),('K',sample(k)),('V',sample(v)),('context_native',sample(y)),('context_reference',recon),('p_first',p[...,0]),('p_prefix8',p[...,:8].sum(-1)),('entropy',-jnp.sum(jnp.where(p>0,p*jnp.log(jnp.maximum(p,1e-30)),0),-1)),('max_source',jnp.argmax(p,-1)),('score_first_nope',scores[0][...,0]),('score_first_rope',scores[1][...,0]),('score_nope_rms',jnp.sqrt(jnp.mean(scores[0]**2,-1))),('score_rope_rms',jnp.sqrt(jnp.mean(scores[1]**2,-1))),('score_max_other',jnp.max(jnp.where((jnp.arange(4096)[None,:]<qi[:,None])[None,None] & (segments[si,None,None,:]==segments[si[:,None],qi[None,:]][:,None,:,None]) & (segments[si,None,None,:]!=0),scores[0]+scores[1],-1e30)[...,1:],-1))]:record(module,name,x)
  current_v[-1]=sample(v)
  return y
 def write(m,groups,**kw):
  result=original_write(m,groups,**kw)
  if blocks and len(groups)==2:
   assert not any(a is not None for a in kw.get('content_adapters',()))
   divs=kw.get('content_divisors',(None,None))
   for i,label in enumerate(('attention','memory_mlp')):
    content,logits,address=groups[i]
    delta=base['write_reference'](sample(content),sample(logits),sample(address),kw['epsilon'],divs[i])
    record(blocks[-1],label+'_dM',delta)
  return result
 def intercept(next_fun,args,kwargs,context):
  module=context.module;method=context.method_name
  if isinstance(module,fusion.SubDecoderLayer) and method=='__call__':
   layers.append(kwargs.get('layer_index',module.layer_inx));blocks.append(module)
   try:
    record(module,'M_in',sample(kwargs['M_in']));record(module,'residual_in',sample(args[0]))
    result=next_fun(*args,**kwargs);record(module,'M_out',sample(result[1]));record(module,'residual_out',sample(result[0]));return result
   finally:layers.pop();blocks.pop()
  if isinstance(module,attentions.BamAttention) and method=='__call__':
   attns.append(module);current_v.append(None)
   try:return next_fun(*args,**kwargs)
   finally:attns.pop();current_v.pop()
  if isinstance(module,attentions.BamAttention) and method=='out_projection':
   result=next_fun(*args,**kwargs)
   w=nn.unbox(module.variables['params']['out']['kernel']).astype(jnp.float32)
   record(module,'V_to_residual',jnp.einsum('bqhd,hdo->bqho',current_v[-1][:,jnp.array([0,4,6])],w))
   record(module,'head_to_residual',jnp.einsum('bqhd,hdo->bqho',sample(args[1])[:,jnp.array([0,4,6])].astype(jnp.float32),w))
   return result
  return next_fun(*args,**kwargs)
 def objective(params,batch,rng):
  with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
  body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
  vals={k[5:]:v[0]for k,v in body.items()if k.startswith('sink_')}
  vals.update({k[5:]:v[0]for k,v in body['self_attention'].items()if k.startswith('sink_')})
  leaves=[v for path,v in jax.tree_util.tree_flatten_with_path(params)[0]if getattr(path[-1],'key',None)=='logits_dense']
  assert len(leaves)==1
  vals['lm_head_centered_gram']=centered_gram(leaves[0])
  vals['sample_tokens']=sample(batch['inputs']);vals['sample_segments']=sample(batch['inputs_segmentation'])
  return loss,vals
 replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
 compiled=jax.jit(objective,in_shardings=(shardings.params,data_sharding,None),out_shardings=replicated)
 result=dict(projected_query_indices=[0,4,6],runtime='d54c4c5f1339b01e406932ec28be0e398722b918',layers=list(range(27)),sequences=SEQUENCES,queries=QUERIES,batch_step=35676,scope='Scanned layers0..26; final27 excluded. Four fixed sequences,11query positions,allsourcekeys. Rank sinks excludingqueries0/1. Fullbatch frozenloss gates. FP32 reference != nativeSplash.',input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest())for x in captured['data']['inputs'].addressable_shards],cases=[])
 upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
 (output/'input-shards.json').write_text(json.dumps(result['input_shards'],indent=2)+'\n')
 bam_pallas.read=read;bam_pallas.write=write;attentions._bam_splash_attention=splash
 try:
  for name,params in [('old_control',old),('peak_control',state.params)]:
   start=time.monotonic()
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(params,captured['data'],captured['rng'])
   loss,values=jax.device_get((loss,values));expected=next(c['metrics']['learning/loss']for c in native['cases']if c['name']==name)
   error=abs(float(loss)-expected);assert error<1e-3,(name,'forward fidelity',error)
   assert all(np.isfinite(np.asarray(x)).all()for x in values.values()),name
   entry=dict(name=name,loss=float(loss),original_aot_loss=expected,error=error,seconds=time.monotonic()-start)
   if jax.process_index()==0:
    dest=output/(name+'-sink-origin.npz');np.savez_compressed(dest,**{k:np.asarray(v)for k,v in values.items()});entry.update(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest());subprocess.run(['gsutil','-q','cp',str(dest),upload+'/'+dest.name],check=True,timeout=300)
   result['cases'].append(entry)
   (output/'sink-origin-local.json').write_text(json.dumps(result,indent=2)+'\n')
   if jax.process_index()==0:(output/'sink-origin.json').write_text(json.dumps(result,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(output/'sink-origin.json'),upload+'/sink-origin.json'],check=True,timeout=90)
  result['complete']=True
  (output/'sink-origin-local.json').write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:(output/'sink-origin.json').write_text(json.dumps(result,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(output/'sink-origin.json'),upload+'/sink-origin.json'],check=True,timeout=90)
 finally:bam_pallas.read=original_read;bam_pallas.write=original_write;attentions._bam_splash_attention=original_splash

"""Native-gated W_O gradient AV/O split with shared output cotangent, first six layers."""
import json,os,hashlib,subprocess
import numpy as np

def split_surrogate(A,O,weights,masks):
 import jax,jax.numpy as jnp
 a=jax.lax.stop_gradient(A.astype(jnp.float32));o=jax.lax.stop_gradient(O.astype(jnp.float32))
 dy=jnp.zeros(A.shape[:2]+(weights.shape[-1],),jnp.float32)
 for k,m in enumerate(masks):
  dy=dy+jnp.einsum('btnh,nhd->btd',a,weights[k,0])*m[...,None]
  dy=dy+jnp.einsum('btnh,nhd->btd',o,weights[k,1])*m[...,None]
 return dy-jax.lax.stop_gradient(dy)

def attach_split(y,A,O,weights,masks):
 """Identity forward and identity y cotangent, with separately recorded branch VJPs."""
 import jax,jax.numpy as jnp
 @jax.custom_vjp
 def tap(y,a,o,p,m):return y
 def forward(y,a,o,p,m):return y,(a,o,m)
 def backward(res,G):
  a,o,masks=res;G=G.astype(jnp.float32)
  parts=jnp.stack([jnp.stack([jnp.einsum('btnh,btd->nhd',v.astype(jnp.float32),G*mask[...,None])for v in (a,o)])for mask in masks])
  return G.astype(y.dtype),None,None,parts,None
 tap.defvjp(forward,backward)
 return tap(y,A,O,weights,masks)

def run_probe(train,captured,reference,output,sequence,target_layer):
 import jax,jax.numpy as jnp
 from flax import linen as nn
 from layers import attentions,bam_pallas
 cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
 step=captured['step'];assert step in (35635,35636) and jax.process_count()==4 and jax.device_count()==16
 expect={35635:(2.0112438201904297,.21486762166023254),35636:(1.924464225769043,2.1978654861450195)}[step]
 ref=captured['observed'];assert abs(ref['learning/loss']-expect[0])<1e-6 and abs(ref['learning/raw_grad_norm']/expect[1]-1)<1e-6
 rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
 active=[];probes=None;instrument=False;current_batch=None
 native_read=bam_pallas.read;native_splash=attentions._bam_splash_attention
 def read(*args,**kw):
  out=native_read(*args,**kw)
  if instrument:active[-1]['O']=out[3]
  return out
 def splash(*args,**kw):
  out=native_splash(*args,**kw)
  if instrument:active[-1]['A']=out
  return out
 def intercept(next_fun,args,kw,ctx):
  mod=ctx.module
  if isinstance(mod,attentions.BamAttention) and ctx.method_name=='__call__':
   active.append(dict(layer=kw.get('layer_index',jnp.int32(mod.layer_inx))))
   try:return next_fun(*args,**kw)
   finally:active.pop()
  if isinstance(mod,attentions.BamAttention) and ctx.method_name=='out_projection':
   y=next_fun(*args,**kw);v=active[-1];i=v['layer'];H=args[1];A=v['A'];O=v['O']
   # Runtime d54 folds VO scales inside read. Explicit forward sum closure below catches mismatches.
   assert H.shape==A.shape==O.shape and H.shape[0]==128
   ah=jax.lax.stop_gradient(A.astype(jnp.float32));oh=jax.lax.stop_gradient(O.astype(jnp.float32))
   hh=H.astype(jnp.float32);weights=probes[jnp.minimum(i,5)]
   seq=(jnp.arange(128)==9)[:,None];tok=seq&(jnp.arange(H.shape[1])==1211)[None,:]
   masks=(jnp.ones(seq.shape,jnp.float32),seq.astype(jnp.float32),tok.astype(jnp.float32))
   # Keep the real output/cotangent entirely outside layer-selection control flow.
   y=attach_split(y,ah,oh,weights,tuple(m*(i<6).astype(jnp.float32)for m in masks))
   def measured(_):
    vals=[];valid=current_batch['targets_segmentation']!=0
    for mask in (valid,valid&seq,valid&tok):
     w=mask[...,None,None].astype(jnp.float32)
     vals.append(jnp.stack([jnp.sum(z*w)for z in (ah*ah,oh*oh,hh*hh,ah*oh,(hh-ah-oh)**2)]+[jnp.sum(w)*H.shape[-2]*H.shape[-1]]))
    return jnp.stack(vals)
   stats=jax.lax.cond(i<6,measured,lambda _:jnp.zeros((3,6),jnp.float32),None)
   mod.sow('intermediates','wo_split_stats',stats)
   return y
  return next_fun(*args,**kw)
 def objective(params,batch,rng,p,instr):
  nonlocal probes,instrument,current_batch
  probes=p;instrument=instr;current_batch=batch
  if instr:
   with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
   stats=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']['self_attention']['wo_split_stats'][0][:6]
  else:
   loss,_=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True);stats=jnp.zeros((6,3,6),jnp.float32)
  return loss,stats
 shapes=(6,3,2,cfg.num_query_heads,cfg.head_dim,cfg.emb_dim)
 zeros=jnp.zeros(shapes,jnp.float32)
 def plain(params,batch,rng):return jax.value_and_grad(lambda p:objective(p,batch,rng,zeros,False)[0])(params)
 plain_fn=jax.jit(plain,in_shardings=(shardings.params,ds,None),out_shardings=(rep,shardings.params))
 def stats_gradient(params,batch,rng,p):
  return jax.value_and_grad(objective,argnums=(0,3),has_aux=True)(params,batch,rng,p,True)
 fn=jax.jit(stats_gradient,in_shardings=(shardings.params,ds,None,rep),out_shardings=((rep,rep),(shardings.params,rep)))
 def norm(tree):return jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2)for x in jax.tree.leaves(tree)))
 norm_fn=jax.jit(norm,out_shardings=rep)
 def relative(g,b):return norm(jax.tree.map(lambda x,y:x.astype(jnp.float32)-y.astype(jnp.float32),g,b))/norm(b)
 rel_fn=jax.jit(relative,out_shardings=rep)
 bam_pallas.read=read;attentions._bam_splash_attention=splash
 try:
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,base=plain_fn(state.params,captured['data'],captured['rng'])
  loss=float(jax.device_get(loss));raw=float(jax.device_get(norm_fn(base)))
  assert abs(loss-expect[0])<.001 and abs(raw/expect[1]-1)<.01,('plain/native',step,loss,raw,expect)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):(measured_loss,amp),(grad,parts)=fn(state.params,captured['data'],captured['rng'],zeros)
  measured_loss=float(jax.device_get(measured_loss));err=float(jax.device_get(rel_fn(grad,base)))
  assert abs(measured_loss-loss)<1e-6 and err<.01,('instrument/plain',step,measured_loss,loss,err)
  amp=np.asarray(jax.device_get(amp),np.float64);parts=np.asarray(jax.device_get(parts),np.float64)
  wo_fn=jax.jit(lambda tree:tree['params']['decoder']['layers']['block']['self_attention']['out']['kernel'],out_shardings=rep)
  real=np.asarray(jax.device_get(wo_fn(grad)),np.float64)
  plain_wo=np.asarray(jax.device_get(wo_fn(base)),np.float64)
  wo_fn.clear_cache()
  rows=[]
  for i in range(6):
   actual=real[:,i];plain_w=plain_wo[:,i]
   full=parts[i,0].sum(axis=0)
   closure=float(np.linalg.norm(full-actual)/max(np.linalg.norm(actual),1e-30))
   assert closure<.015,('AV+O gradient closure',i,closure)
   wo_error=float(np.linalg.norm(actual-plain_w)/max(np.linalg.norm(plain_w),1e-30))
   print('WO_LAYER_GATE',step,i,wo_error,closure,flush=True)
   assert wo_error<.01,('W_O instrument/plain',step,i,wo_error)
   scopes=[]
   for k,label in enumerate(('fullbatch','sequence9','token9_1211')):
    ga,go=parts[i,k];na,no=np.linalg.norm(ga),np.linalg.norm(go);ns=np.linalg.norm(ga+go)
    a2,o2,h2,cross,sum_error,n=amp[i,k]
    assert n>0 and sum_error/max(h2,1e-30)<1e-4,('forward sum',i,k,sum_error,h2)
    d=dict(scope=label,AV_gradient_norm=float(na),O_gradient_norm=float(no),sum_gradient_norm=float(ns),gradient_cosine=float(np.sum(ga*go)/max(na*no,1e-30)),O_gradient_projection_fraction=float(np.sum(go*(ga+go))/max(ns*ns,1e-30)),AV_rms=float(np.sqrt(a2/n)),O_rms=float(np.sqrt(o2/n)),H_rms=float(np.sqrt(h2/n)),activation_cosine=float(cross/max(np.sqrt(a2*o2),1e-30)),sum_forward_relative_squared_error=float(sum_error/max(h2,1e-30)))
    if k==2:
     # A single position gives a rank-one outer product, so recover the shared G norm twice.
     gv=na/max(np.sqrt(a2),1e-30);gov=no/max(np.sqrt(o2),1e-30)
     assert abs(gv/gov-1)<.01,('shared G rank-one check',i,gv,gov)
     d['output_G_norm_from_AV']=float(gv);d['output_G_norm_from_O']=float(gov)
    scopes.append(d)
   rows.append(dict(layer=i,native_parameter_norm=float(np.linalg.norm(plain_w)),instrumented_parameter_norm=float(np.linalg.norm(actual)),split_relative_closure=closure,scopes=scopes))
  report=dict(owner='msg_0f011e9dd18442e28ec4dc26543b4697',step=step,runtime='d54c4c5f1339b01e406932ec28be0e398722b918',native_control=ref,plain_loss=loss,plain_raw=raw,instrumented_loss=measured_loss,instrumented_vector_relative_error=err,input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest())for s in captured['data']['inputs'].addressable_shards],layers=rows,complete=True,scope='Actual fullbatch loss output cotangent G shared by AV/O; selected-position contributions use global training normalization. Normal35635 and first-spike35636 are adjacent-batch comparisons, not same-batch counterfactuals.')
  output.mkdir(exist_ok=True,parents=True);p=output/'wo-split.json';p.write_text(json.dumps(report,indent=2)+'\n')
  if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'].rstrip('/')+f'/step{step}/wo-split.json'],check=True,timeout=90)
  print('WO_SPLIT_COMPLETE',step,flush=True)
 finally:
  bam_pallas.read=native_read;attentions._bam_splash_attention=native_splash
  plain_fn.clear_cache();fn.clear_cache()

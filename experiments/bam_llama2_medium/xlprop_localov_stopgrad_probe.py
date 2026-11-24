"""Fixed-forward localO/V backward interventions; original full-batch normalization."""
import json,os,hashlib,subprocess
from pathlib import Path

def cut_value(value,cut):
 import jax,jax.numpy as jnp
 return jnp.where(cut,jax.lax.stop_gradient(value),value)

def arm_masks():
 import numpy as np
 z=np.zeros((28,4),bool);arms=[('identity',z.copy())]
 for label,layers,slots in [('O_L5',[5],[3]),('O_first6',range(6),[3]),('V_L5',[5],[2]),('V_first6',range(6),[2]),('OV_first6',range(6),[2,3]),('QK_first6',range(6),[0,1])]:
  m=z.copy()
  for l in layers:m[l,slots]=True
  arms.append((label,m))
 for slot,label in [(3,'O'),(2,'V')]:
  for layer in range(5):
   m=z.copy();m[layer,slot]=True;arms.append((f'{label}_L{layer}',m))
 return arms

def run_probe(train,captured,reference,output,sequence,target_layer):
 import jax,jax.numpy as jnp,numpy as np,orbax.checkpoint as ocp
 from flax import linen as nn,serialization
 from layers import attentions,bam_pallas
 cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
 point=os.environ.get('BAM_LOCALOV_POINT','peak')
 assert point in ('onset','peak'),point
 onset=point=='onset'
 assert captured['step']==(35636 if onset else 35676) and jax.process_count()==4 and jax.device_count()==16
 assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full'
 if onset:
  states=[('onset',state.params)]
 else:
  abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
  cp=ocp.PyTreeCheckpointer();restored=cp.restore(os.environ['BAM_DIAG_REFERENCE_CHECKPOINT'],item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
  old=serialization.from_state_dict(state.params,serialization.to_state_dict(restored['params']));del restored,abstract
  states=[('healthy',old),('peak',state.params)]
 # Original serialized AOT controls, no retained optimizer updates or checkpoint saves.
 controls={}
 for name,params in states:
  # The native executable donates its TrainState buffers. Never give it the frozen source.
  trial=jax.tree.map(lambda x:x.copy() if hasattr(x,'copy') else x,state.replace(params=params))
  updated,metrics=captured['compiled'](trial,captured['data'],captured['rng'])
  del updated
  vals=metrics['scalar'];controls[name]={k:float(jax.device_get(vals[k])) for k in ('learning/loss','learning/raw_grad_norm')}
  expect=(1.924464226,2.197865486) if name=='onset' else ((2.595792532,809414.5625) if name=='peak' else (1.977011561,.215273544))
  assert abs(controls[name]['learning/loss']-expect[0])<1e-6
  assert abs(controls[name]['learning/raw_grad_norm']/expect[1]-1)<1e-6
 original_read=bam_pallas.read;active_layer=None;mask=None
 def read(*args,**kw):
  values=original_read(*args,**kw)
  assert active_layer is not None
  return tuple(cut_value(v,mask[active_layer,i]) if i<4 else v for i,v in enumerate(values))
 def intercept(next_fun,args,kw,ctx):
  nonlocal active_layer
  if isinstance(ctx.module,attentions.BamAttention) and ctx.method_name=='__call__':
   previous=active_layer
   active_layer=kw.get('layer_index',None)
   if active_layer is None:active_layer=jnp.int32(ctx.module.layer_inx)
   try:return next_fun(*args,**kw)
   finally:active_layer=previous
  return next_fun(*args,**kw)
 paths,treedef=jax.tree_util.tree_flatten_with_path(state.params)
 names=['/'.join(str(getattr(k,'key',k))for k in p)for p,v in paths]
 def objective(params,batch,rng,cutmask,instrument):
  nonlocal mask
  mask=cutmask
  if instrument:
   with nn.intercept_methods(intercept):loss,_=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
  else:loss,_=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
  return loss
 def measure(params,batch,rng,cutmask,instrument):
  return jax.value_and_grad(objective)(params,batch,rng,cutmask,instrument)
 rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
 plain=jax.jit(lambda p,b,r:measure(p,b,r,jnp.zeros((28,4),bool),False),in_shardings=(shardings.params,ds,None),out_shardings=(rep,shardings.params))
 def compare(gs,base):
  gl=jax.tree.leaves(gs);bl=jax.tree.leaves(base)
  values=[]
  for (path,x),g,f in zip(paths,gl,bl):
   g=g.astype(jnp.float32);f=f.astype(jnp.float32)
   diff=g-f
   # Per-leaf totals and first-six layer breakdown, scan axis is 1.
   def stat(a,b):return jnp.stack((jnp.sum(a*a),jnp.sum(b*b),jnp.sum((a-b)**2),jnp.sum(a*b)))
   total=stat(g,f)
   layer= jnp.stack([stat(jnp.take(g,l,axis=1),jnp.take(f,l,axis=1)) for l in range(6)]) if '/layers/block/' in names[len(values)] and x.ndim>=2 and x.shape[1]==27 else jnp.zeros((6,4),jnp.float32)
   values.append((total,layer))
  return tuple(values)
 compare_fn=jax.jit(compare,out_shardings=rep)
 report=dict(owner=os.environ.get('BAM_LOCALOV_OWNER','msg_2b5f6bb10fe64aa0912accb1f4c0a944'),runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=captured['step'],layer_index='zero based; first6 = 0..5, L5 = sixth layer',scope='Full-batch backward-only read-output cuts; both residual and memory-write consumers of read output. No training or retained updates.',controls=controls,input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest())for s in captured['data']['inputs'].addressable_shards],cases=[])
 def save():
  (output/'localov-stopgrad.json').write_text(json.dumps(report,indent=2)+'\n')
  if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(output/'localov-stopgrad.json'),os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'].rstrip('/')+'/localov-stopgrad.json'],check=True,timeout=90)
 save()
 try:
  for name,params in states:
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):base_loss,base=plain(params,captured['data'],captured['rng'])
   base_loss=float(jax.device_get(base_loss))
   base_stats=jax.device_get(compare_fn(base,base));base_norm=float(np.sqrt(sum(float(t[0])for t,l in base_stats)))
   ref=controls[name]
   assert abs(base_loss-ref['learning/loss'])<.001 and abs(base_norm/ref['learning/raw_grad_norm']-1)<.01,('unmodified autodiff native fidelity',name,base_loss,base_norm,ref)
   bam_pallas.read=read
   compiled=jax.jit(lambda p,b,r,m:measure(p,b,r,m,True),in_shardings=(shardings.params,ds,None,None),out_shardings=(rep,shardings.params))
   case=dict(checkpoint=name,baseline_loss=base_loss,baseline_raw=base_norm,arms=[]);report['cases'].append(case);save()
   for label,m in arm_masks():
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,g=compiled(params,captured['data'],captured['rng'],jnp.asarray(m))
    loss=float(jax.device_get(loss));stats=jax.device_get(compare_fn(g,base))
    totals=np.sum([x[0]for x in stats],axis=0,dtype=np.float64)
    raw,base_raw,difference=map(float,np.sqrt(np.maximum(totals[:3],0)))
    cosine=float(totals[3]/max(raw*base_raw,1e-30))
    row=dict(name=label,loss=loss,raw=raw,raw_ratio=raw/base_raw,removed_vector_norm=difference,cosine_to_baseline=cosine,leaf_metrics=[dict(name=n,total=np.asarray(t).tolist(),first6=np.asarray(l).tolist())for n,(t,l)in zip(names,stats)])
    case['arms'].append(row);save();print('STOPGRAD_ARM',name,label,loss,raw,flush=True)
    assert abs(loss-base_loss)<1e-6,('stop gradient changed forward',name,label,loss,base_loss)
    if label=='identity':assert difference/base_raw<.01,('instrumented gradient fidelity',name,difference/base_raw)
    assert np.isfinite(totals).all()
    if label in ('O_L5','O_first6','OV_first6'):
     selected=[x for x in row['leaf_metrics'] if '/decoder/layers/block/self_attention/static_o_key' in x['name']]
     assert len(selected)==1,('Cannot verify O cut coordinate',len(selected))
     ls=np.asarray(selected[0]['first6'])
     chosen=[5] if label=='O_L5' else list(range(6))
     assert np.all(ls[chosen,0]==0),('O cut did not zero selected address gradients',label,ls.tolist())
    del g
   bam_pallas.read=original_read
   compiled.clear_cache();del base
  report['complete']=True;save()
 finally:bam_pallas.read=original_read

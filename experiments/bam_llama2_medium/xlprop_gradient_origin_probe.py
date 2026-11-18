"""Disjoint loss-cohort gradient origins, frozen native full-batch forward.

No activations/masks/parameters are changed. Cohort weights multiply final xent
AFTER Transformer returns; train.loss_fn retains its original denominator.
Full-parameter differentiation is deliberately retained for the historical raw
gradient fidelity gate; only first-six V-read/W_O leaves survive the JIT output.
"""
from pathlib import Path
import runpy


def cohort_weights(batch, length, mode):
  import jax.numpy as jnp
  s=jnp.arange(batch)[:,None];t=jnp.arange(length)[None,:]
  seq=(s//(batch//4))==(mode-1)
  pos=(t//(length//4))==(mode-5)
  return jnp.broadcast_to(jnp.where(mode==0,True,jnp.where(mode<=4,seq,pos)),(batch,length))


def selected_leaf(path,shape):
  names=tuple(getattr(x,'key',None)for x in path)
  if 'self_attention' not in names or 'layers' not in names or 'block' not in names:return None
  part=names[names.index('self_attention')+1]
  if part=='out':family='W_O'
  elif part.startswith(('W_lv_','general_v_')) or part=='static_v_key':family='V_read'
  else:return None
  assert len(shape)>1 and shape[1]==27,(names,shape)
  return family


def run_probe(train,captured,reference,output,sequence,target_layer):
  import hashlib,json,os,subprocess
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn,serialization
  from layers.models import Transformer
  helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
  helper['run_probe'].__globals__['case_plans']=lambda:[dict(name='peak_control',families=[],direction='revert'),dict(name='old_control',families=['ALL'],direction='revert')]
  helper['run_probe'](train,captured,reference,output,sequence,target_layer)
  controls={x['name']:x['metrics']for x in json.loads((output/'loss-localization.json').read_text())['cases']}
  cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
  assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full'
  assert jax.process_count()==4 and jax.device_count()==16 and not cfg.qk_norm
  abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
  cp=ocp.PyTreeCheckpointer();raw=cp.restore(os.environ['BAM_DIAG_REFERENCE_CHECKPOINT'],item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
  old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
  paths,leaves=zip(*jax.tree_util.tree_flatten_with_path(state.params)[0]);selected=[(i,selected_leaf(p,x.shape))for i,(p,x)in enumerate(zip(paths,leaves))];selected=[(i,f)for i,f in selected if f]
  assert {f for _,f in selected}=={'W_O','V_read'}
  names=['/'.join(str(getattr(k,'key',k))for k in paths[i])for i,_ in selected]
  def objective(params,batch,rng,mode):
    def intercept(next_fun,args,kwargs,context):
      result=next_fun(*args,**kwargs)
      if isinstance(context.module,Transformer) and context.method_name=='__call__':
        xent,correct,preds=result;assert xent.shape==(128,4096),xent.shape
        return (xent*cohort_weights(128,4096,mode).astype(xent.dtype),correct,preds)
      return result
    with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
    return loss
  def measure(params,batch,rng,mode):
    loss,g=jax.value_and_grad(objective)(params,batch,rng,mode)
    raw=train.max_utils.l2norm_pytree(g);ls=jax.tree.leaves(g)
    chosen=tuple(jnp.take(ls[i],jnp.arange(6),axis=1)for i,_ in selected)
    return loss,raw,chosen
  replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
  data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  chosen_shards=tuple(jax.tree.leaves(shardings.params)[i]for i,_ in selected)
  compiled=jax.jit(measure,in_shardings=(shardings.params,data,None,None),out_shardings=(replicated,replicated,chosen_shards))
  # Scalar contribution dot products retain signed cancellation; norm-squared
  # fractions are NOT additive. All contractions/closures are FP32 on device.
  def metrics(gs,full):
    return jnp.stack([jnp.stack([jnp.sum(g[:,l].astype(jnp.float32)**2),jnp.sum(f[:,l].astype(jnp.float32)**2),jnp.sum(g[:,l].astype(jnp.float32)*f[:,l].astype(jnp.float32))])for g,f in zip(gs,full)for l in range(6)])
  metric_fn=jax.jit(metrics,out_shardings=replicated)
  result=dict(runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,checkpoints=[35635,35675],selected_leaves=[dict(name=n,family=f)for n,(_,f)in zip(names,selected)],input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest())for x in captured['data']['inputs'].addressable_shards],scope='Disjoint final-loss cohorts, original global normalization; token positions refer to loss origins, not V source positions. Full forward unchanged, no retained updates.',cases=[])
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  def save():
    (output/'gradient-origin-local.json').write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(output/'gradient-origin-local.json'),upload+'/gradient-origin.json'],check=True,timeout=90)
  for name,params in [('old_control',old),('peak_control',state.params)]:
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw,full=compiled(params,captured['data'],captured['rng'],jnp.int32(0))
    loss,raw=map(float,jax.device_get((loss,raw)));ref=controls[name]
    assert abs(loss-ref['learning/loss'])<.001 and abs(raw/ref['learning/raw_grad_norm']-1)<.01,('full fidelity',name,loss,raw,ref)
    case=dict(name=name,loss=loss,raw=raw,original_aot=ref,groups=[],closures=[]);result['cases'].append(case);save()
    for kind,modes in [('sequence',range(1,5)),('position',range(5,9))]:
      total=None;loss_sum=0.
      for mode in modes:
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):part_loss,part_raw,gs=compiled(params,captured['data'],captured['rng'],jnp.int32(mode))
        rows=np.asarray(jax.device_get(metric_fn(gs,full)));assert np.isfinite(rows).all()
        pl,pr=map(float,jax.device_get((part_loss,part_raw)));assert np.isfinite(pl)and np.isfinite(pr)
        case['groups'].append(dict(kind=kind,index=(mode-1)%4,loss=pl,raw=pr,leaf_layer_metrics=rows.tolist(),columns=['component_squared_norm','full_squared_norm','component_dot_full']))
        total=tuple(g.astype(jnp.float32)for g in gs)if total is None else tuple(a+g.astype(jnp.float32)for a,g in zip(total,gs));loss_sum+=pl;save()
      errors=metric_fn(tuple(a-f.astype(jnp.float32)for a,f in zip(total,full)),full)
      rows=np.asarray(jax.device_get(errors));rel=np.sqrt(rows[:,0]/np.maximum(rows[:,1],1e-30))
      closure=dict(kind=kind,loss_sum=loss_sum,loss_error=abs(loss_sum-loss),relative_gradient_error_by_leaf_layer=rel.tolist(),passed=bool(abs(loss_sum-loss)<.001 and np.max(rel)<.01))
      case['closures'].append(closure);save();assert closure['passed'],('cohort additivity fidelity',name,closure)
    del full,total
  result['complete']=True;save()

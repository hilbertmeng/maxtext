"""Sparse causal surgery on weak content directions of healthy per-token M.

Full frozen batch loss. Interventions cover only2sequences×12queries and L3,
L5 orL0..5, not the whole batch's M writes. Weakness is relative to each old
M's99%-energy right-singular subspace. Ordinary D-dimensional residual weakness
is a distinct question. Keep the native fused write and add a projected analytic
attention-write correction; report reference-vs-native errors and BF16 leakage.
"""
from pathlib import Path
import runpy
SEQUENCES=(9,104)
QUERIES=(0,1,2,8,16,128,512,1211,1220,2048,3072,4095)
MODES=('identity','remove_attention_weak','restore_attention_weak','matched_remove_attention_strong','matched_remove_attention_random','remove_attention_all','restore_attention_all','remove_total_weak','restore_total_weak')


def weak_projectors(matrix,threshold=.99):
  import numpy as np
  shape=matrix.shape[:-2];width=matrix.shape[-1];projectors=[];ranks=[]
  for x in np.asarray(matrix,np.float64).reshape((-1,*matrix.shape[-2:])):
    _,s,vh=np.linalg.svd(x,full_matrices=False);total=float(np.sum(s*s));rank=int(np.searchsorted(np.cumsum(s*s),threshold*total)+1) if total>0 else 0
    projectors.append(np.eye(width)-vh[:rank].T@vh[:rank]);ranks.append(rank)
  return np.asarray(projectors,np.float32).reshape((*shape,width,width)),np.asarray(ranks).reshape(shape)


def correction(attention,total,old_attention,old_total,projector,mode):
  import jax
  import jax.numpy as jnp
  import numpy as np
  weak=attention@projector;strong=attention-weak
  delta=old_attention-attention;dw=delta@projector
  permutation=jnp.asarray(np.random.default_rng(9876).permutation(projector.shape[-1]))
  random_projector=jnp.take(jnp.take(projector,permutation,axis=-2),permutation,axis=-1)
  random=attention@random_projector
  norm=lambda x:jnp.sqrt(jnp.sum(x*x,axis=(-2,-1),keepdims=True))
  matched=lambda x:x*norm(weak)/jnp.maximum(norm(x),1e-30)
  return jax.lax.switch(mode,(lambda:jnp.zeros_like(attention),lambda:-weak,lambda:dw,
      lambda:-matched(strong),lambda:-matched(random),lambda:-attention,
      lambda:delta,lambda:-(total@projector),lambda:(old_total-total)@projector))


def control_plans():
  return [dict(name='peak_control',families=[],direction='revert'),dict(name='old_control',families=['ALL'],direction='revert')]


def run_probe(train,captured,reference,output,sequence,target_layer):
  import json,os,hashlib,subprocess
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn,serialization
  pattern=runpy.run_path(str(Path(__file__).with_name('xlprop_pattern_write_probe.py')))
  attentions,fusion,bam=pattern['runtime_modules']();write_reference=pattern['write_reference']
  helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
  helper['run_probe'].__globals__['case_plans']=control_plans
  helper['run_probe'](train,captured,reference,output,sequence,target_layer)
  controls={c['name']:c['metrics'] for c in json.loads((output/'loss-localization.json').read_text())['cases']}
  cfg,mesh,state=captured['config'],captured['mesh'],captured['state'];model,_,shardings=captured['functional_train'].args
  assert jax.device_count()==16 and jax.process_count()==4
  assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full' and cfg.bam_pallas_body=='v8t'
  abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
  cp=ocp.PyTreeCheckpointer();raw=cp.restore(os.environ['BAM_DIAG_REFERENCE_CHECKPOINT'],item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
  old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
  si=jnp.asarray(SEQUENCES);qi=jnp.asarray(QUERIES);stack=[];active=None;active_valid=None;original_write=bam.write
  sample=lambda x:x[si[:,None],qi[None,:]]
  def write(m,groups,**kwargs):
    result=original_write(m,groups,**kwargs);module,layer=stack[-1];assert len(groups)==2
    assert not any(x is not None for x in kwargs.get('content_adapters',()))
    divisors=kwargs.get('content_divisors',(None,None));refs=[]
    for i in range(2):
      content,logits,address=groups[i];refs.append(write_reference(sample(content),sample(logits),sample(address),kwargs['epsilon'],divisors[i]))
    m0=sample(m);native=sample(result);actual=(native.astype(jnp.float32)-m0.astype(jnp.float32))
    projectors,healthy_attention,healthy_total,mode,mask=active;index=jnp.minimum(layer,5)
    chosen=(layer<6)&((jnp.right_shift(mask,layer)&1)!=0)&(mode!=0)
    def change(value):
      addition=correction(refs[0],actual,healthy_attention[index],healthy_total[index],projectors[index],mode)
      addition=jnp.where(sample(active_valid)[...,None,None]!=0,addition,0.)
      changed=(sample(value).astype(jnp.float32)+addition).astype(value.dtype)
      return value.at[si[:,None],qi[None,:]].set(changed)
    result=jax.lax.cond(chosen,change,lambda x:x,result)
    for name,value in [('M_in',m0),('M_native',native),('M_out',sample(result)),('attention_reference',refs[0]),('memory_reference',refs[1])]:module.sow('intermediates','weak_'+name,value)
    return result
  def intercept(next_fun,args,kwargs,context):
    module=context.module
    if isinstance(module,fusion.SubDecoderLayer) and context.method_name=='__call__':
      layer=kwargs.get('layer_index',module.layer_inx);layer=module.layer_inx if layer is None else layer
      stack.append((module,jnp.asarray(layer)))
      try:return next_fun(*args,**kwargs)
      finally:stack.pop()
    return next_fun(*args,**kwargs)
  def objective(params,batch,rng,projectors,healthy_attention,healthy_total,mode,mask):
    nonlocal active,active_valid
    active_valid=batch['inputs_segmentation']
    active=(projectors,healthy_attention,healthy_total,mode,mask)
    with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
    body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
    return loss,{k.removeprefix('weak_'):v[0][:6] for k,v in body.items() if k.startswith('weak_')}
  def measure(params,*args):
    (loss,values),grads=jax.value_and_grad(objective,has_aux=True)(params,*args)
    values['parameter_raw_grad_norm']=train.max_utils.l2norm_pytree(grads)
    return loss,values
  replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());batch_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  compiled=jax.jit(measure,in_shardings=(shardings.params,batch_sharding,None,replicated,replicated,replicated,None,None),out_shardings=replicated)
  projectors=np.zeros((6,2,12,96,96),np.float32);healthy_attention=np.zeros((6,2,12,40,96),np.float32);healthy_total=np.zeros_like(healthy_attention)
  valid_samples=np.asarray(jax.device_get(jax.jit(sample,out_shardings=replicated)(captured['data']['inputs_segmentation'])))!=0
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  result=dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,sequences=SEQUENCES,queries=QUERIES,valid_samples=valid_samples.tolist(),scope='Sparse24Mpositions atselectedlayers; attention correction uses analytic pre-W_O write reference; total correction uses actual native M delta. Not ordinary residual weak-subspace or whole-batch intervention.',input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest()) for x in captured['data']['inputs'].addressable_shards],cases=[])
  def save(name,loss,values,**extra):
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in values.values())
    result['cases'].append(dict(name=name,loss=float(loss),raw_grad_norm=float(values['parameter_raw_grad_norm']),**extra))
    if jax.process_index()==0:
      p=output/(name+'-weak-memory.npz');np.savez_compressed(p,**values);subprocess.run(['gsutil','-q','cp',str(p),upload+'/'+p.name],check=True,timeout=180)
    p=output/'weak-memory.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/weak-memory.json'],check=True,timeout=90)
    print('WEAK_MEMORY_CASE',name,float(loss),flush=True)
  bam.write=write
  try:
    for label,params in [('old',old),('peak',state.params)]:
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=jax.device_get(compiled(params,captured['data'],captured['rng'],projectors,healthy_attention,healthy_total,jnp.int32(0),jnp.int32(0)))
      ref=controls[label+'_control'];error=abs(float(values['parameter_raw_grad_norm'])/ref['learning/raw_grad_norm']-1)
      assert abs(float(loss)-ref['learning/loss'])<1e-3 and error<.05,('Native weak no-op fidelity failed',label,float(loss),error)
      actual=values['M_native'].astype(np.float64)-values['M_in'].astype(np.float64);analytic=values['attention_reference']+values['memory_reference']
      ref_error=float(np.linalg.norm(actual-analytic)/max(np.linalg.norm(actual),1e-30))
      assert ref_error<.03,('Analytic combined reference mismatch',ref_error)
      if label=='old':
        assert values['M_in'].shape==(6,2,12,40,96),values['M_in'].shape
        from jax.experimental import multihost_utils
        if jax.process_index()==0:projectors,ranks=weak_projectors(values['M_in'])
        else:ranks=np.zeros((6,2,12),np.int32)
        projectors,ranks=multihost_utils.broadcast_one_to_all((projectors,np.asarray(ranks,np.int32)),is_source=jax.process_index()==0)
        healthy_attention=values['attention_reference'];healthy_total=actual.astype(np.float32);result['rank99']=ranks.tolist()
      else:peak_reference=values
      save(label+'_control',loss,values,fidelity_passed=True,gradient_relative_error=error,combined_reference_relative_l2_error=ref_error)
    plans=[('peak_noop',0,63)]+[(scope+'_'+MODES[mode],mode,mask) for scope,mask in [('L03',8),('L05',32),('L00_05',63)] for mode in range(1,9)]
    for name,mode,mask in plans:
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=jax.device_get(compiled(state.params,captured['data'],captured['rng'],projectors,healthy_attention,healthy_total,jnp.int32(mode),jnp.int32(mask)))
      extra={}
      if mode==0:
        assert all(np.array_equal(values[k],peak_reference[k]) for k in values), 'No-op M identity failed'
        extra['internal_identity_passed']=True
      save(name,loss,values,**extra)
    result['complete']=True;p=output/'weak-memory.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/weak-memory.json'],check=True,timeout=90)
  finally:bam.write=original_write

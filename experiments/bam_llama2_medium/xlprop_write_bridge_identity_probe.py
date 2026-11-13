"""Same-full-gradient-graph donors with internal identity fidelity gates.

Immutable d54 model; no optimizer writes. Healthy donor writes remain sharded on
TPU. Peak reference writes are also fixed, so direction/amplitude comparisons do
not silently change their reference as upstream interventions change the state.
Only small samples and downstream normalized MLP inputs are exported.
"""
from pathlib import Path
import runpy

MODES=('identity','toward_025','toward_050','toward_100','away_100','random_91','random_9876','random_17','old_amplitude','old_write')
SEQUENCES=(9,104)
QUERIES=(0,1,2,8,16,128,512,1211,1220,2048,3072,4095)


def patch_write(current,old,mode):
  """Norm-preserving spherical controls; random arms match healthy angle."""
  import jax
  import jax.numpy as jnp
  import numpy as np
  c=current.astype(jnp.float32);o=old.astype(jnp.float32)
  cn=jnp.linalg.norm(c,axis=-1,keepdims=True);on=jnp.linalg.norm(o,axis=-1,keepdims=True)
  p=c/jnp.maximum(cn,1e-30);q=o/jnp.maximum(on,1e-30)
  dot=jnp.clip(jnp.sum(p*q,axis=-1,keepdims=True),-1.,1.)
  theta=jnp.arccos(dot);t=q-dot*p
  t=t/jnp.maximum(jnp.linalg.norm(t,axis=-1,keepdims=True),1e-30)
  def rotate(alpha,tangent):
    angle=alpha*theta
    y=(jnp.cos(angle)*p+jnp.sin(angle)*tangent)*cn
    return jnp.where(theta<1e-5,c,y).astype(current.dtype)
  random=[]
  for seed in (91,9876,17):
    v=jnp.take(q,jnp.asarray(np.random.default_rng(seed).permutation(current.shape[-1])),axis=-1)
    v=v-jnp.sum(v*p,axis=-1,keepdims=True)*p
    v=v/jnp.maximum(jnp.linalg.norm(v,axis=-1,keepdims=True),1e-30)
    random.append(v)
  return jax.lax.switch(mode,(
    lambda:current,
    lambda:rotate(.25,t),lambda:rotate(.5,t),lambda:rotate(1.,t),
    lambda:rotate(-1.,t),
    lambda:rotate(1.,random[0]),lambda:rotate(1.,random[1]),lambda:rotate(1.,random[2]),
    lambda:(p*on).astype(current.dtype),lambda:old.astype(current.dtype)))


def inject_samples(value,probe,sequences=SEQUENCES,queries=QUERIES):
  import jax.numpy as jnp
  return value.at[jnp.asarray(sequences)[:,None],jnp.asarray(queries)[None,:]].add(probe)


def control_plans():
  return [dict(name='peak_control',families=[],direction='revert'),
          dict(name='old_control',families=['ALL'],direction='revert')]


def run_probe(train,captured,reference,output,sequence,target_layer):
  import hashlib,json,os,subprocess,time
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn,serialization
  modules=runpy.run_path(str(Path(__file__).with_name('xlprop_pattern_write_probe.py')))
  attentions,fusion,_=modules['runtime_modules']()
  from layers import linears
  helper=runpy.run_path(str(Path(__file__).with_name('xlprop_routing_localization_probe.py')))
  helper['run_probe'].__globals__['case_plans']=control_plans
  helper['run_probe'](train,captured,reference,output,sequence,target_layer)
  controls={c['name']:c['metrics']['learning/loss'] for c in json.loads((output/'loss-localization.json').read_text())['cases']}
  cfg,mesh,state=captured['config'],captured['mesh'],captured['state']
  model,_,shardings=captured['functional_train'].args
  assert cfg.scan_layers and cfg.param_scan_axis==1 and cfg.remat_policy=='full'
  assert cfg.bam_pallas_body=='v8t' and jax.device_count()==16 and jax.process_count()==4
  abstract={'params':jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state.params)}
  cp=ocp.PyTreeCheckpointer()
  raw=cp.restore(os.environ['BAM_DIAG_REFERENCE_CHECKPOINT'],item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract));cp.close()
  old=serialization.from_state_dict(state.params,serialization.to_state_dict(raw['params']));del raw,abstract
  si=jnp.asarray(SEQUENCES);qi=jnp.asarray(QUERIES)
  def sample(x):return x[si[:,None],qi[None,:]]
  def make_objective(export_donors):
    stack=[];mlp_stack=[];active=None;active_probes=None
    def intercept(next_fun,args,kwargs,context):
      module=context.module;method=context.method_name
      if isinstance(module,fusion.SubDecoderLayer) and method=='__call__':
        layer=kwargs.get('layer_index',module.layer_inx);layer=module.layer_inx if layer is None else layer
        stack.append(layer)
        try:return next_fun(*args,**kwargs)
        finally:stack.pop()
      if isinstance(module,attentions.BamAttention) and method=='__call__':
        result=next_fun(*args,**kwargs);write=result[0];layer=jnp.asarray(stack[-1])
        if active is not None:
          donors,peak_donors,mode,mask=active
          selected=(layer<6)&((jnp.right_shift(mask,layer)&1)!=0)
          write=jax.lax.cond(selected,lambda _:patch_write(peak_donors[jnp.minimum(layer,5)],donors[jnp.minimum(layer,5)],mode),lambda _:write,None)
        module.sow('intermediates','direction_sampled_write',sample(write))
        if export_donors:module.sow('intermediates','direction_full_write',write)
        # M and its pre-W_O write factors are returned unchanged at this layer.
        return (write,)+tuple(result[1:])
      if isinstance(module,linears.DenseGeneral) and module.name in ('wi_0','wi_1') and mlp_stack and method=='__call__':
        y=next_fun(*args,**kwargs)
        mlp_stack[-1].sow('intermediates','bridge_'+module.name,sample(y))
        return y
      if isinstance(module,linears.MlpBlock) and module.name=='mlp' and method=='__call__':
        layer=jnp.asarray(stack[-1]);selected=(layer==6)|(layer==13)|(layer==14)
        index=jnp.argmax(layer==jnp.asarray([6,13,14]))
        x=args[0]
        if active_probes is not None:
          x=jax.lax.cond(selected,lambda z:inject_samples(z,active_probes[0,index]),lambda z:z,x)
        module.sow('intermediates','direction_normalized_input',sample(x))
        mlp_stack.append(module)
        try:y=next_fun(x,*args[1:],**kwargs)
        finally:mlp_stack.pop()
        module.sow('intermediates','direction_mlp_output',sample(y))
        if active_probes is not None:
          y=jax.lax.cond(selected,lambda z:inject_samples(z,active_probes[1,index]),lambda z:z,y)
        return y
      return next_fun(*args,**kwargs)
    def objective(params,batch,rng,*extra):
      nonlocal active,active_probes
      active=extra[:4] if extra else None
      active_probes=extra[4] if len(extra)>4 else None
      with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
      body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
      values=dict(write=body['self_attention']['direction_sampled_write'][0][:6],
        mlp_input=body['mlp']['direction_normalized_input'][0][jnp.asarray([6,13,14])],
        mlp_output=body['mlp']['direction_mlp_output'][0][jnp.asarray([6,13,14])],
        mlp_gate=body['mlp']['bridge_wi_0'][0][jnp.asarray([6,13,14])],
        mlp_up=body['mlp']['bridge_wi_1'][0][jnp.asarray([6,13,14])])
      if export_donors:values['donors']=body['self_attention']['direction_full_write'][0][:6]
      return loss,values
    return objective
  replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
  batch_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  donor_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(None,*cfg.data_sharding))
  old_fn=jax.jit(make_objective(True),in_shardings=(shardings.params,batch_sharding,None),
    out_shardings=(replicated,dict(donors=donor_sharding,write=replicated,mlp_input=replicated,mlp_output=replicated,mlp_gate=replicated,mlp_up=replicated)))
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):old_loss,values=old_fn(old,captured['data'],captured['rng'])
  donors=values.pop('donors')
  assert donors.shape==(6,128,4096,1920),donors.shape
  assert donors.sharding==donor_sharding,(donors.sharding,donor_sharding)
  loss,small=jax.device_get((old_loss,values))
  assert abs(float(loss)-controls['old_control'])<1e-3,('Old donor forward fidelity',float(loss))
  result=dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,
    sequences=SEQUENCES,queries=QUERIES,mlp_layers=[6,13,14],
    donor_shape=donors.shape,donor_dtype=str(donors.dtype),donor_sharding=str(donors.sharding),
    negative_control='Three permuted old directions projected orthogonal to peak, rotated by healthy angle: matched token norm AND displacement; not isotropic noise.',
    scope='Ordinary attention vector output patch using frozen old and peak references; same-layer pre-W_O M-write factors untouched. Downstream states may respond. Conditional full-parameter gradients and actual sampled ordinary-MLP input/output cotangents. Global norm drops after clamping are not loss-cause proof.',
    input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest()) for x in captured['data']['inputs'].addressable_shards],cases=[])
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  def save(name,loss,values,**extra):
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in values.values()),name
    if jax.process_index()==0:
      p=output/(name+'-write-bridge.npz');np.savez_compressed(p,**{k:np.asarray(v,dtype=np.float32) for k,v in values.items()})
      subprocess.run(['gsutil','-q','cp',str(p),upload+'/'+p.name],check=True,timeout=180)
    result['cases'].append(dict(name=name,loss=float(loss),**extra))
    p=output/'write-bridge.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/write-bridge.json'],check=True,timeout=90)
    print('WRITE_BRIDGE_CASE',name,float(loss),flush=True)
  save('old_control',float(loss),small,original_aot_loss=controls['old_control'],fidelity_passed=True)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):peak_loss,values=old_fn(state.params,captured['data'],captured['rng'])
  peak_donors=values.pop('donors');peak_loss,small=jax.device_get((peak_loss,values))
  assert abs(float(peak_loss)-controls['peak_control'])<1e-3,('Peak donor forward fidelity',float(peak_loss))
  save('peak_control',float(peak_loss),small,original_aot_loss=controls['peak_control'],fidelity_passed=True)
  valid=captured['data']['inputs_segmentation']!=0
  zero_counts=[int(jax.device_get(jnp.sum(jnp.all(d==0,axis=-1)&valid[None]))) for d in (donors,peak_donors)]
  assert zero_counts==[0,0],('Direction undefined for valid donor tokens',zero_counts)
  result['zero_valid_donor_counts']=zero_counts
  objective=make_objective(True)
  def measure(params,batch,rng,donors,peak_donors,mode,mask,probes):
    def loss_fn(params,probes):return objective(params,batch,rng,donors,peak_donors,mode,mask,probes)
    (loss,values),(grads,dp)=jax.value_and_grad(loss_fn,argnums=(0,1),has_aux=True)(params,probes)
    values.update(parameter_raw_grad_norm=train.max_utils.l2norm_pytree(grads),mlp_input_vjp=dp[0],mlp_output_vjp=dp[1])
    return loss,values
  probes=jnp.zeros((2,3,2,12,1920),cfg.dtype)
  compiled=jax.jit(measure,in_shardings=(shardings.params,batch_sharding,None,donor_sharding,donor_sharding,None,None,replicated),out_shardings=(replicated,dict(donors=donor_sharding,write=replicated,mlp_input=replicated,mlp_output=replicated,mlp_gate=replicated,mlp_up=replicated,parameter_raw_grad_norm=replicated,mlp_input_vjp=replicated,mlp_output_vjp=replicated)))
  # Unclamped old/peak controls must reproduce original raw gradients. Frozen
  # clamping disconnects upstream parameter paths and is FORWARD-gated only.
  for label,params in [('old',old),('peak',state.params)]:
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(params,captured['data'],captured['rng'],donors,peak_donors,jnp.int32(0),jnp.int32(0),probes)
    gradient_donors=values.pop('donors')
    if label=='old':donors=gradient_donors
    else:peak_donors=gradient_donors
    loss,values=jax.device_get((loss,values))
    if label=='peak':peak_internal_reference={k:np.array(v) for k,v in values.items()}
    refs={c['name']:c['metrics'] for c in json.loads((output/'loss-localization.json').read_text())['cases']}
    expected=refs[label+'_control'];raw=float(values['parameter_raw_grad_norm'])
    err=abs(raw/expected['learning/raw_grad_norm']-1)
    assert abs(float(loss)-expected['learning/loss'])<1e-3 and err<.05,('Full-gradient fidelity failed',label,float(loss),raw,err)
    save(label+'_gradient_control',float(loss),values,gradient_relative_error=err,fidelity_passed=True)
  plans=[('frozen_peak_control',0,63)]+[(f'{label}_{MODES[mode]}',mode,mask) for label,mask in [('L00_05',63),('L03',8)] for mode in range(1,10)]
  for name,mode,mask in plans:
    start=time.monotonic()
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(state.params,captured['data'],captured['rng'],donors,peak_donors,jnp.int32(mode),jnp.int32(mask),probes)
    del values['donors']  # Never gather full donors to host.
    loss,values=jax.device_get((loss,values));extra={}
    if name=='frozen_peak_control':
      error=abs(float(loss)-controls['peak_control']);assert error<1e-3,('Peak noop fidelity',error)
      extra=dict(original_aot_loss=controls['peak_control'],fidelity_passed=True,loss_abs_error=error)
      assert np.array_equal(values['write'],peak_internal_reference['write']), 'Frozen write samples differ from same-graph donors'
      state_errors={}
      for k in ('mlp_input','mlp_output','mlp_gate','mlp_up','mlp_input_vjp','mlp_output_vjp'):
        ref=peak_internal_reference[k].astype(np.float64);actual=values[k].astype(np.float64)
        rel=np.linalg.norm((actual-ref).reshape((3,-1)),axis=-1)/np.maximum(np.linalg.norm(ref.reshape((3,-1)),axis=-1),1e-30)
        state_errors[k]=rel.tolist()
        assert np.max(rel)<(.05 if k.endswith('_vjp') else .005), ('Internal identity fidelity failed',k,rel.tolist())
      extra['internal_identity_relative_errors']=state_errors
      extra['internal_identity_passed']=True
    save(name,float(loss),values,seconds=time.monotonic()-start,**extra)
  result['complete']=True
  p=output/'write-bridge.json';p.write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/write-bridge.json'],check=True,timeout=90)

"""Same-full-gradient-graph donors with internal identity fidelity gates.

Immutable d54 model; no optimizer writes. Healthy donor writes remain sharded on
TPU. Peak reference writes are also fixed, so direction/amplitude comparisons do
not silently change their reference as upstream interventions change the state.
Only small write/MLP samples and healthy second moments/bases are host-exported. Covariances are stop-gradient observations; no optimizer writes.
"""
from pathlib import Path
import runpy

MODES=('identity','weak99_remove','weak99_restore','weak99_matched_restore_strong','weak99_matched_restore_random_weak','weak95_remove','weak95_restore','weak95_matched_restore_strong','weak95_matched_restore_random_weak','old_write')
SEQUENCES=(9,104)
QUERIES=(0,1,2,8,16,128,512,1211,1220,2048,3072,4095)


def patch_write(current,old,projectors,mode):
  import jax
  import jax.numpy as jnp
  import numpy as np
  p=current.astype(jnp.float32);o=old.astype(jnp.float32)
  basis=projectors[jnp.where(mode<=4,0,1)]
  mm=lambda x:jnp.matmul(x,basis,precision=jax.lax.Precision.HIGHEST)
  delta=o-p;weak=mm(delta);strong=delta-weak
  permutation=jnp.asarray(np.random.default_rng(9876).permutation(p.shape[-1]))
  random=mm(jnp.take(weak,permutation,axis=-1))
  norm=lambda x:jnp.linalg.norm(x,axis=-1,keepdims=True)
  matched=lambda x:x*norm(weak)/jnp.maximum(norm(x),1e-30)
  local_mode=jnp.where(mode==9,5,jnp.where(mode==0,0,(mode-1)%4+1))
  return jax.lax.switch(local_mode,(lambda:current,lambda:(p-mm(p)).astype(current.dtype),lambda:(p+weak).astype(current.dtype),lambda:(p+matched(strong)).astype(current.dtype),lambda:(p+matched(random)).astype(current.dtype),lambda:old.astype(current.dtype)))


def residual_moment(inputs,valid,positions):
  import jax
  import jax.numpy as jnp
  x=jnp.take(inputs,positions,axis=1).astype(jnp.float32)
  selected=jnp.take(valid,positions,axis=1)
  x=jnp.where(selected[...,None],x,0).reshape((-1,x.shape[-1]))
  count=jnp.sum(selected.astype(jnp.float32))
  return jnp.matmul(x.T,x,precision=jax.lax.Precision.HIGHEST)/jnp.maximum(count,1)


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
    stack=[];mlp_stack=[];active=None;active_probes=None;active_valid=None
    def intercept(next_fun,args,kwargs,context):
      module=context.module;method=context.method_name
      if isinstance(module,fusion.SubDecoderLayer) and method=='__call__':
        layer=kwargs.get('layer_index',module.layer_inx);layer=module.layer_inx if layer is None else layer
        # First-six healthy pre-attention residual second moments. All sequences,
        # disjoint regular grids; never gather token activations to host.
        def moments(positions):
          def compute(_):
            return residual_moment(args[0],active_valid,positions)
          return jax.lax.cond(jnp.asarray(layer)<6,compute,lambda _:jnp.zeros((1920,1920),jnp.float32),None)
        module.sow('intermediates','ordinary_calibration',jax.lax.stop_gradient(moments(jnp.arange(32,4096,64))))
        module.sow('intermediates','ordinary_validation',jax.lax.stop_gradient(moments(jnp.arange(16,4096,64))))
        stack.append(layer)
        try:return next_fun(*args,**kwargs)
        finally:stack.pop()
      if isinstance(module,attentions.BamAttention) and method=='__call__':
        result=next_fun(*args,**kwargs);write=result[0];layer=jnp.asarray(stack[-1])
        if active is not None:
          donors,peak_donors,mode,mask,bases=active
          selected=(layer<6)&((jnp.right_shift(mask,layer)&1)!=0)
          write=jax.lax.cond(selected,lambda _:patch_write(peak_donors[jnp.minimum(layer,5)],donors[jnp.minimum(layer,5)],bases[:,jnp.minimum(layer,5)],mode),lambda _:write,None)
        # Full-batch quantization diagnostics: sums, not sampled estimates.
        stats=jnp.zeros((6,),jnp.float32)
        if active is not None:
          def correction_stats(_):
            li=jnp.minimum(layer,5);ti=jnp.where(mode<=4,0,1)
            peak=peak_donors[li].astype(jnp.float32);healthy=donors[li].astype(jnp.float32)
            basis=bases[ti,li]
            intended=patch_write(peak,healthy,bases[:,li],mode)-peak
            actual=write.astype(jnp.float32)-peak
            strong=actual-jnp.matmul(actual,basis,precision=jax.lax.Precision.HIGHEST)
            keep=active_valid[...,None]
            energy=lambda x:jnp.sum(jnp.where(keep,x*x,0))
            return jnp.stack([energy(intended),energy(actual),energy(actual-intended),energy(strong),jnp.sum(jnp.where(keep,actual*intended,0)),jnp.sum(active_valid & (jnp.linalg.norm(intended,axis=-1)==0))])
          stats=jax.lax.cond(selected,correction_stats,lambda _:stats,None)
        module.sow('intermediates','ordinary_correction_stats',stats)
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
      nonlocal active,active_probes,active_valid
      active=extra[:5] if extra else None
      active_probes=extra[5] if len(extra)>5 else None
      active_valid=batch['inputs_segmentation']!=0
      with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
      body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
      values=dict(correction_stats=body['self_attention']['ordinary_correction_stats'][0][:6],calibration=body['ordinary_calibration'][0][:6],validation=body['ordinary_validation'][0][:6],write=body['self_attention']['direction_sampled_write'][0][:6],
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
    out_shardings=(replicated,dict(correction_stats=replicated,calibration=replicated,validation=replicated,donors=donor_sharding,write=replicated,mlp_input=replicated,mlp_output=replicated,mlp_gate=replicated,mlp_up=replicated)))
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):old_loss,values=old_fn(old,captured['data'],captured['rng'])
  donors=values.pop('donors')
  del values['calibration'],values['validation']
  assert donors.shape==(6,128,4096,1920),donors.shape
  assert donors.sharding==donor_sharding,(donors.sharding,donor_sharding)
  loss,small=jax.device_get((old_loss,values))
  assert abs(float(loss)-controls['old_control'])<1e-3,('Old donor forward fidelity',float(loss))
  result=dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,
    sequences=SEQUENCES,queries=QUERIES,mlp_layers=[6,13,14],
    donor_shape=donors.shape,donor_dtype=str(donors.dtype),donor_sharding=str(donors.sharding),
    negative_control='Healthy strong-direction restoration and randomized-within-weak restoration match the healthy weak-restoration displacement norm per token before BF16. Random seed9876; zero control vectors reported; exact BF16 norm mismatches assessed.',
    scope='Whole-batch ordinary attention vector output patch; weak basis is the healthy pre-attention residual uncentered second moment, with disjoint 64-position grids over all128sequences. Same frozen batch, not independent data. Ordinary attention vector output patch using frozen old and peak references; same-layer pre-W_O M-write factors untouched. Downstream states may respond. Conditional full-parameter gradients and actual sampled ordinary-MLP input/output cotangents. Global norm drops after clamping are not loss-cause proof.',
    input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest()) for x in captured['data']['inputs'].addressable_shards],cases=[])
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  def save(name,loss,values,**extra):
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in values.values()),name
    if jax.process_index()==0:
      p=output/(name+'-ordinary-weak.npz');np.savez_compressed(p,**{k:np.asarray(v,dtype=np.float32) for k,v in values.items()})
      subprocess.run(['gsutil','-q','cp',str(p),upload+'/'+p.name],check=True,timeout=180)
    result['cases'].append(dict(name=name,loss=float(loss),**extra))
    p=output/'ordinary-weak.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/ordinary-weak.json'],check=True,timeout=90)
    print('ORDINARY_WEAK_CASE',name,float(loss),flush=True)
  save('old_control',float(loss),small,original_aot_loss=controls['old_control'],fidelity_passed=True)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):peak_loss,values=old_fn(state.params,captured['data'],captured['rng'])
  peak_donors=values.pop('donors');del values['calibration'],values['validation']
  peak_loss,small=jax.device_get((peak_loss,values))
  assert abs(float(peak_loss)-controls['peak_control'])<1e-3,('Peak donor forward fidelity',float(peak_loss))
  save('peak_control',float(peak_loss),small,original_aot_loss=controls['peak_control'],fidelity_passed=True)
  valid=captured['data']['inputs_segmentation']!=0
  zero_counts=[int(jax.device_get(jnp.sum(jnp.all(d==0,axis=-1)&valid[None]))) for d in (donors,peak_donors)]
  assert zero_counts==[0,0],('Direction undefined for valid donor tokens',zero_counts)
  result['zero_valid_donor_counts']=zero_counts
  objective=make_objective(True)
  def measure(params,batch,rng,donors,peak_donors,mode,mask,bases,probes):
    def loss_fn(params,probes):return objective(params,batch,rng,donors,peak_donors,mode,mask,bases,probes)
    (loss,values),(grads,dp)=jax.value_and_grad(loss_fn,argnums=(0,1),has_aux=True)(params,probes)
    values.update(parameter_raw_grad_norm=train.max_utils.l2norm_pytree(grads),mlp_input_vjp=dp[0],mlp_output_vjp=dp[1])
    return loss,values
  probes=jnp.zeros((2,3,2,12,1920),cfg.dtype)
  compiled=jax.jit(measure,in_shardings=(shardings.params,batch_sharding,None,donor_sharding,donor_sharding,None,None,replicated,replicated),out_shardings=(replicated,dict(correction_stats=replicated,calibration=replicated,validation=replicated,donors=donor_sharding,write=replicated,mlp_input=replicated,mlp_output=replicated,mlp_gate=replicated,mlp_up=replicated,parameter_raw_grad_norm=replicated,mlp_input_vjp=replicated,mlp_output_vjp=replicated)))
  bases=jnp.zeros((2,6,1920,1920),jnp.float32)
  bases=jax.device_put(bases,replicated)
  # Unclamped old/peak controls must reproduce original raw gradients. Frozen
  # clamping disconnects upstream parameter paths and is FORWARD-gated only.
  for label,params in [('old',old),('peak',state.params)]:
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(params,captured['data'],captured['rng'],donors,peak_donors,jnp.int32(0),jnp.int32(0),bases,probes)
    gradient_donors=values.pop('donors')
    covariance={k:values.pop(k) for k in ('calibration','validation')}
    if label=='old':
      # This is the only host covariance gather. Other cases discard matrices.
      healthy_moments=jax.device_get(covariance)
    if label=='peak':peak_moments=covariance
    del covariance
    if label=='old':donors=gradient_donors
    else:peak_donors=gradient_donors
    loss,values=jax.device_get((loss,values))
    if label=='peak':peak_internal_reference={k:np.array(v) for k,v in values.items()}
    refs={c['name']:c['metrics'] for c in json.loads((output/'loss-localization.json').read_text())['cases']}
    expected=refs[label+'_control'];raw=float(values['parameter_raw_grad_norm'])
    err=abs(raw/expected['learning/raw_grad_norm']-1)
    assert abs(float(loss)-expected['learning/loss'])<1e-3 and err<.05,('Full-gradient fidelity failed',label,float(loss),raw,err)
    save(label+'_gradient_control',float(loss),values,gradient_relative_error=err,fidelity_passed=True)
  from jax.experimental import multihost_utils
  if jax.process_index()==0:
    utility=runpy.run_path(str(Path(__file__).with_name('ordinary_weak_projection.py')))
    basis_host,metadata=utility['calibrate'](healthy_moments['calibration'],healthy_moments['validation'])
    artifact=output/'ordinary-healthy-basis.npz'
    np.savez_compressed(artifact,**healthy_moments,projectors=basis_host)
    subprocess.run(['gsutil','-q','cp',str(artifact),upload+'/'+artifact.name],check=True,timeout=180)
    meta_bytes=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8)
  else:
    basis_host=np.zeros((2,6,1920,1920),np.float32);meta_bytes=np.zeros(0,np.uint8)
  basis_host=multihost_utils.broadcast_one_to_all(basis_host,is_source=jax.process_index()==0)
  meta_size=int(multihost_utils.broadcast_one_to_all(np.int32(len(meta_bytes)),is_source=jax.process_index()==0))
  if jax.process_index()!=0:meta_bytes=np.zeros(meta_size,np.uint8)
  meta_bytes=multihost_utils.broadcast_one_to_all(meta_bytes,is_source=jax.process_index()==0)
  result['basis']=json.loads(meta_bytes.tobytes())
  result['basis']['valid_counts']={label:int(jax.device_get(jnp.sum(jnp.take(valid,jnp.arange(offset,4096,64),axis=1)))) for label,offset in [('calibration',32),('validation',16)]}
  assert all(n>1920 for n in result['basis']['valid_counts'].values()),result['basis']['valid_counts']
  bases=jax.device_put(basis_host,replicated);del basis_host,healthy_moments
  # Peak residual energy in the frozen healthy bases: shift is measured on the
  # same held-out grid without gathering peak matrices to host.
  result['peak_energy']={k:dict(total=np.asarray(jax.device_get(jnp.trace(v,axis1=-2,axis2=-1))).tolist(),weak=np.asarray(jax.device_get(jnp.einsum('tlij,lji->tl',bases,v,precision=jax.lax.Precision.HIGHEST))).tolist()) for k,v in peak_moments.items()}
  del peak_moments
  plans=[('frozen_peak_control',0,63)]+[(f'{label}_{MODES[mode]}',mode,mask) for label,mask in [('L00_05',63),('L03',8)] for mode in range(1,10)]
  for name,mode,mask in plans:
    start=time.monotonic()
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(state.params,captured['data'],captured['rng'],donors,peak_donors,jnp.int32(mode),jnp.int32(mask),bases,probes)
    del values['donors'],values['calibration'],values['validation']  # Never gather full donors or per-case matrices.
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
  p=output/'ordinary-weak.json';p.write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/ordinary-weak.json'],check=True,timeout=90)

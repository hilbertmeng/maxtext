"""Full-batch old-direction vs old-amplitude ordinary-attention write patches.

Immutable d54 model; no optimizer writes. Healthy donor writes remain sharded on
TPU. Peak reference writes are also fixed, so direction/amplitude comparisons do
not silently change their reference as upstream interventions change the state.
Only small samples and downstream normalized MLP inputs are exported.
"""
from pathlib import Path
import runpy

MODES=('identity','old_direction','old_amplitude','old_write','random_direction')
SEQUENCES=(9,104)
QUERIES=(0,1,2,8,16,128,512,1211,1220,2048,3072,4095)


def patch_write(current,old,mode):
  import jax
  import jax.numpy as jnp
  import numpy as np
  c=current.astype(jnp.float32);o=old.astype(jnp.float32)
  cn=jnp.linalg.norm(c,axis=-1,keepdims=True);on=jnp.linalg.norm(o,axis=-1,keepdims=True)
  unit_old=jnp.where(on>0,o/jnp.maximum(on,1e-30),c/jnp.maximum(cn,1e-30))
  unit_current=jnp.where(cn>0,c/jnp.maximum(cn,1e-30),unit_old)
  return jax.lax.switch(mode,(
    lambda:current,
    lambda:(unit_old*cn).astype(current.dtype),
    lambda:(unit_current*on).astype(current.dtype),
    lambda:old.astype(current.dtype),
    lambda:(jnp.take(unit_old,jnp.asarray(np.random.default_rng(9876).permutation(current.shape[-1])),axis=-1)*cn).astype(current.dtype)))


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
    stack=[];active=None
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
      if isinstance(module,linears.MlpBlock) and module.name=='mlp' and method=='__call__':
        module.sow('intermediates','direction_normalized_input',sample(args[0]))
      return next_fun(*args,**kwargs)
    def objective(params,batch,rng,*extra):
      nonlocal active
      active=extra if extra else None
      with nn.intercept_methods(intercept):loss,aux=train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)
      body=aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']
      values=dict(write=body['self_attention']['direction_sampled_write'][0][:6],
        mlp_input=body['mlp']['direction_normalized_input'][0][jnp.asarray([6,13,14])])
      if export_donors:values['donors']=body['self_attention']['direction_full_write'][0][:6]
      return loss,values
    return objective
  replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
  batch_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
  donor_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(None,*cfg.data_sharding))
  old_fn=jax.jit(make_objective(True),in_shardings=(shardings.params,batch_sharding,None),
    out_shardings=(replicated,dict(donors=donor_sharding,write=replicated,mlp_input=replicated)))
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):old_loss,values=old_fn(old,captured['data'],captured['rng'])
  donors=values.pop('donors')
  assert donors.shape==(6,128,4096,1920),donors.shape
  assert donors.sharding==donor_sharding,(donors.sharding,donor_sharding)
  loss,small=jax.device_get((old_loss,values))
  assert abs(float(loss)-controls['old_control'])<1e-3,('Old donor forward fidelity',float(loss))
  result=dict(model_runtime='d54c4c5f1339b01e406932ec28be0e398722b918',batch_step=35676,
    sequences=SEQUENCES,queries=QUERIES,mlp_layers=[6,13,14],
    donor_shape=donors.shape,donor_dtype=str(donors.dtype),donor_sharding=str(donors.sharding),
    negative_control='Seed9876 permutation of old write coordinates, with frozen peak token norm; not isotropic noise.',
    scope='Ordinary attention vector output patch using frozen old and peak references; same-layer pre-W_O M-write factors untouched. Downstream states may respond. No instrumented gradient/gain claims.',
    input_shards=[dict(index=str(x.index),sha256=hashlib.sha256(np.asarray(x.data).tobytes()).hexdigest()) for x in captured['data']['inputs'].addressable_shards],cases=[])
  upload=helper['upload_destination'](os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'])
  def save(name,loss,values,**extra):
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in values.values()),name
    if jax.process_index()==0:
      p=output/(name+'-write-direction.npz');np.savez_compressed(p,**{k:np.asarray(v,dtype=np.float32) for k,v in values.items()})
      subprocess.run(['gsutil','-q','cp',str(p),upload+'/'+p.name],check=True,timeout=180)
    result['cases'].append(dict(name=name,loss=float(loss),**extra))
    p=output/'write-direction.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/write-direction.json'],check=True,timeout=90)
    print('WRITE_DIRECTION_CASE',name,float(loss),flush=True)
  save('old_control',float(loss),small,original_aot_loss=controls['old_control'],fidelity_passed=True)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):peak_loss,values=old_fn(state.params,captured['data'],captured['rng'])
  peak_donors=values.pop('donors');peak_loss,small=jax.device_get((peak_loss,values))
  assert abs(float(peak_loss)-controls['peak_control'])<1e-3,('Peak donor forward fidelity',float(peak_loss))
  save('peak_control',float(peak_loss),small,original_aot_loss=controls['peak_control'],fidelity_passed=True)
  valid=captured['data']['inputs_segmentation']!=0
  zero_counts=[int(jax.device_get(jnp.sum(jnp.all(d==0,axis=-1)&valid[None]))) for d in (donors,peak_donors)]
  assert zero_counts==[0,0],('Direction undefined for valid donor tokens',zero_counts)
  result['zero_valid_donor_counts']=zero_counts
  compiled=jax.jit(make_objective(False),in_shardings=(shardings.params,batch_sharding,None,donor_sharding,donor_sharding,None,None),out_shardings=replicated)
  plans=[('frozen_peak_control',0,63)]+[(f'{label}_{MODES[mode]}',mode,mask) for label,mask in [('L00_05',63),('L03',8)] for mode in range(1,5)]
  for name,mode,mask in plans:
    start=time.monotonic()
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,values=compiled(state.params,captured['data'],captured['rng'],donors,peak_donors,jnp.int32(mode),jnp.int32(mask))
    loss,values=jax.device_get((loss,values));extra={}
    if name=='frozen_peak_control':
      error=abs(float(loss)-controls['peak_control']);assert error<1e-3,('Peak noop fidelity',error)
      extra=dict(original_aot_loss=controls['peak_control'],fidelity_passed=True,loss_abs_error=error)
    save(name,float(loss),values,seconds=time.monotonic()-start,**extra)
  result['complete']=True
  p=output/'write-direction.json';p.write_text(json.dumps(result,indent=2)+'\n')
  if jax.process_index()==0:subprocess.run(['gsutil','-q','cp',str(p),upload+'/write-direction.json'],check=True,timeout=90)

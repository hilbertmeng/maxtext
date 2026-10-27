"""Read-only follow-up: actual native A vs dynamic/static O over identical checkpoints."""
import argparse,json,subprocess
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import checkpointing
import no_self_capture as base
import no_self_o_components as component
from no_self_probe import setup,cohort,hash_sequence,save_npz,read_remote,publish_pair,unpack

def main():
 p=argparse.ArgumentParser()
 for k in ['manifest','family','training-config','cohort-reference','output','gcs-output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());m0=m['models'][a.family];a.experiment=m0['experiment'];a.expected_parameters=m0['parameter_count'];a.label='noself_components_'+a.family;a.length=2048;a.batch_size=4;a.seed=20261021
 source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source==m['commit'];Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True);cfg,_,checked,mesh,model,rng,abstract=setup(a)
 data=cohort('/tmp/bam_stream_probe/noself_cohort32.npz',m['cohort_source'],a.seed,32);hashes=[hash_sequence(x[:2049]) for x in data['tokens']];assert hashes==m['sequence_hashes']==json.loads(Path(a.cohort_reference).read_text())['sequence_hashes']
 batches=[]
 for i in range(0,32,4):
  v=data['tokens'][i:i+4];shape=(4,2048);batches.append(dict(inputs=jnp.asarray(v[:,:2048]),targets=jnp.asarray(v[:,1:2049]),mask=jnp.ones(shape,jnp.int32),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape)))
 def forward(params,b,observed):
  base.ACTIVE=observed;base.OBSERVE=observed;base.CONTROLS=jnp.ones((32,4),jnp.float32) if observed else None
  try:
   kw=dict(decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng});result=model.apply(params,b['inputs'],b['positions'],**kw,**({'mutable':['intermediates']} if observed else {}))
   if not observed:return result[0]
   out,col=result;raw={'/'.join(k):v for k,v in flatten_dict(col['intermediates']).items() if k[-1].startswith('readout_')};return out[0],raw
  finally:base.ACTIVE=False;base.OBSERVE=False;base.CONTROLS=None
 native_fn=jax.jit(lambda p,b:forward(p,b,False));observe_fn=jax.jit(lambda p,b:forward(p,b,True));installed=False
 for step in m['step_order']:
  cp=m['checkpoint_roots'][a.family]+'/'+str(step)+'/items';dest=a.gcs_output+'/'+a.family+'/'+str(step);folder=Path(a.output)/a.family/str(step);folder.mkdir(parents=True,exist_ok=True)
  common=dict(source_commit=source,base_runtime_commit=m['base_runtime_commit'],training_commit=m['training_commit'],checkpoint=cp,step=step,family=a.family,experiment=a.experiment,parameter_count=a.expected_parameters,sequence_hashes=hashes,source_names=list(component.SOURCE_NAMES),query_positions=base.positions(2048).tolist())
  done=read_remote(dest+'/complete.json')
  if done is not None:
   for k in common:assert done[k]==common[k]
   print('COMPONENTS_PREVIOUSLY_COMPLETE',a.family,step,flush=True);continue
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=checkpointing.load_params_from_path(cp,abstract.params)
  assert sum(x.size for x in jax.tree.leaves(params))==a.expected_parameters
  native=[]
  for b in batches:
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):native.append(np.asarray(native_fn(params,b)))
  if not installed:component.install();installed=True
  arrays={};losses=[];first=None
  for i,b in enumerate(batches):
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(observe_fn(params,b))
   losses.append(loss)
   for k,v in unpack(raw,32).items():arrays.setdefault(k,[]).append(v)
   if i==0:first=(loss,raw)
  native=np.concatenate(native);losses=np.concatenate(losses);merged={k:np.concatenate(v,0) for k,v in arrays.items()};assert merged['component_pre'].shape==(32,32,32,4,4)
  drift=float(np.max(np.abs(losses.mean(1)-native.mean(1))));assert drift<.005 and all(np.isfinite(v).all() for v in merged.values());assert merged['empty_history_mha_max'].max()==0
  closure=merged['component_closure'].astype(float);den=closure[...,2].sum(2);errors={key:float(np.sqrt(closure[...,i].sum(2)/np.maximum(den,1e-30)).max()) for i,key in enumerate(['decomposition','native_key_reconstruction'])};assert errors['decomposition']<.02 and errors['native_key_reconstruction']<.01,errors
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,repeat=jax.device_get(observe_fn(params,batches[0]))
  np.testing.assert_array_equal(first[0],again)
  for k,v in first[1].items():np.testing.assert_array_equal(v[0],repeat[k][0])
  f=folder/'components.npz';save_npz(f,**merged,inputs=data['tokens'][:,:2048],targets=data['tokens'][:,1:2049],native_loss=native,observed_loss=losses)
  meta=dict(**common,native_observer_CE_max_difference=drift,closure_max_relative_RMS=errors,repeat_exact=True,verified_config_keys=checked,jax_version=jax.__version__,devices=[str(v) for v in jax.devices()],definition='Actual native MHA/O unchanged. DynamicO=FP32(normalized native prebiased Okey * native sigmoid * key_scale) dot native shifted BF16M; staticO=FP32(native staticOkey * actual native static multiplier) dot sameM. NativePallas rounds sumkey toBF16 beforedot and finalO toBF16; decomposition closure retained, not allocated to either component. PostWo FP32 nativeBF16 kernel, bias separate;64queries>=64.')
  publish_pair(f,folder/'components.json',meta,dest);marker=folder/'complete.json';marker.write_text(json.dumps(common,indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(marker),dest+'/complete.json'],check=True)
  print('COMPONENTS_CHECKPOINT_COMPLETE',a.family,step,drift,errors,flush=True);del params,arrays,merged,raw,first,repeat,native,losses
 print('COMPONENTS_MODEL_COMPLETE',a.family,flush=True)
if __name__=='__main__':main()

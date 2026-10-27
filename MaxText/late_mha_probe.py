"""Matched frozen loss trajectories for whole-layer MHA interventions, four models."""
import argparse,ast,json,subprocess,time
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import checkpointing
import no_self_capture as base
import late_mha_capture as hook
from late_mha_cases import plan,digest
from no_self_probe import setup,cohort,hash_sequence,save_npz,read_remote,publish_pair,unpack

def main():
 p=argparse.ArgumentParser()
 for key in ['manifest','family','training-config','reference','replays','output','gcs-output']:p.add_argument('--'+key,required=True)
 a=p.parse_args();m=json.loads(Path(a.manifest).read_text());info=m['models'][a.family];a.experiment=info['experiment'];a.expected_parameters=info['parameter_count'];a.label='late_'+a.family;a.length=2048;a.batch_size=4;a.seed=20261021;source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source==m['commit'];Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True)
 cfg,critical,checked,mesh,model,rng,abstract=setup(a)
 assert bool(cfg.bam_enabled)==info['bam'] and bool(getattr(cfg,'bam_attention_negative_diagonal',False))==info['no_self'];assert cfg.num_decoder_layers==cfg.num_query_heads==32 and cfg.head_dim==80
 extra={'dense_conn','dynamic_dense_type','dynamic_mlp_dim','scan_layers','scan_use_mudd','fused_qkv','fused_mlp'}
 for line in Path(a.training_config).read_text().splitlines():
  key,sep,raw=line.partition(': ')
  if not sep or key not in cfg.get_keys() or not(key in extra or key.startswith('dynamic_dense_')):continue
  try:want=ast.literal_eval(raw)
  except (ValueError,SyntaxError):want=raw
  value=getattr(cfg,key);value=list(value) if isinstance(value,tuple) else value;want=list(want) if isinstance(want,tuple) else want
  assert value==want,('Extra configuration mismatch',key,value,want);checked.append(key)
 data=cohort('/tmp/bam_stream_probe/late_mha_cohort32.npz',m['cohort_source'],a.seed,32);hashes=[hash_sequence(x[:2049]) for x in data['tokens']];assert hashes==m['sequence_hashes']==json.loads(Path(a.reference).read_text())['sequence_hashes'];replays=json.loads(Path(a.replays).read_text()).get(a.family,{})
 batches=[]
 for i in range(0,32,4):
  v=data['tokens'][i:i+4];shape=(4,2048);batches.append(dict(inputs=jnp.asarray(v[:,:2048]),targets=jnp.asarray(v[:,1:2049]),mask=jnp.ones(shape,jnp.int32),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(2048),shape)))
 def native(p,b):return model.apply(p,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
 native_fn=jax.jit(native)
 def controlled(p,b,c,observe):
  base.ACTIVE=True;base.OBSERVE=observe;base.CONTROLS=c
  try:
   value=model.apply(p,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng},**({'mutable':['intermediates']} if observe else {}))
   if not observe:return value[0]
   out,col=value;return out[0],{'/'.join(k):v for k,v in flatten_dict(col['intermediates']).items() if k[-1].startswith('readout_')}
  finally:base.ACTIVE=False;base.OBSERVE=False;base.CONTROLS=None
 ones=jnp.ones((32,4),jnp.float32);ablate_fn=jax.jit(lambda p,b,c:controlled(p,b,c,False));observe_fn=jax.jit(lambda p,b:controlled(p,b,ones,True));installed=False;cases=plan(info['bam']);pdigest=digest(cases)
 def evaluate(params,c=None):
  values=[]
  for b in batches:
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):v=native_fn(params,b) if c is None else ablate_fn(params,b,c)
   values.append(np.asarray(v,np.float32))
  result=np.concatenate(values);assert result.shape==(32,2048) and np.isfinite(result).all();return result
 def validate(z,common):
  for key in ['source_commit','checkpoint','plan_digest','sequence_hashes','experiment','parameter_count']:assert z[key]==common[key],('Artifact identity',key)
 previous=None
 for step in m['step_order']:
  cp=info['checkpoint_root']+'/'+str(step)+'/items';destination=a.gcs_output+'/'+a.family+'/'+str(step);folder=Path(a.output)/a.family/str(step);folder.mkdir(parents=True,exist_ok=True)
  common=dict(source_commit=source,training_commit=info['training_commit'],family=a.family,experiment=a.experiment,checkpoint=cp,step=step,parameter_count=a.expected_parameters,sequence_hashes=hashes,plan_digest=pdigest,bam=info['bam'],no_self=info['no_self'],length=2048,sequences=32,heads=32,layers=32,query_positions=base.positions(2048).tolist())
  done=read_remote(destination+'/complete.json')
  if done is not None:validate(done,common);print('LATE_PREVIOUS_COMPLETE',a.family,step,flush=True);continue
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=checkpointing.load_params_from_path(cp,abstract.params)
  assert sum(v.size for v in jax.tree.leaves(params))==a.expected_parameters;native_loss=evaluate(params)
  if previous is not None:assert not np.array_equal(native_loss,previous)
  previous=native_loss.copy()
  if not installed:hook.install();installed=True
  neutral=evaluate(params,ones);drift=float(np.max(np.abs(native_loss.mean(1)-neutral.mean(1))));assert drift<.005,('Neutral-native drift',drift)
  replay=replays.get(str(step),{});native_replay=None
  if 'native_CE' in replay:
   native_replay=float(np.max(np.abs(native_loss.mean(1)-np.asarray(replay['native_CE']))));assert native_replay<.005,('Historical native CE mismatch',a.family,step,native_replay)
  captured=read_remote(destination+'/capture.json')
  if captured is None:
   arrays={};losses=[];first=None
   for i,b in enumerate(batches):
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,raw=jax.device_get(observe_fn(params,b))
    losses.append(loss)
    for key,v in unpack(raw,32).items():arrays.setdefault(key,[]).append(v)
    if i==0:first=(loss,raw)
   merged={key:np.concatenate(v,0) for key,v in arrays.items()};observed=np.concatenate(losses);odrift=float(np.max(np.abs(observed.mean(1)-native_loss.mean(1))));assert odrift<.005
   assert merged['pre'].shape==merged['post'].shape==(32,32,5);assert np.all(merged['geometry']==np.array([2048,32,80,2560])) and np.all(merged['controls']==1)
   assert all(np.isfinite(x).all() for x in merged.values())
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,repeat=jax.device_get(observe_fn(params,batches[0]))
   np.testing.assert_array_equal(first[0],again)
   for key,v in first[1].items():np.testing.assert_array_equal(v[0],repeat[key][0])
   file=folder/'capture.npz';save_npz(file,**merged,native_loss=native_loss,neutral_loss=neutral,observed_loss=observed,inputs=data['tokens'][:,:2048],targets=data['tokens'][:,1:2049])
   publish_pair(file,folder/'capture.json',dict(**common,neutral_native_CE_max_difference=drift,observer_native_CE_max_difference=odrift,historical_native_CE_max_difference=native_replay,repeat_exact=True,verified_config_keys=checked,config={key:value for key,value in cfg.get_keys().items() if key in critical or key in extra or key.startswith(('bam_','mudd_','pythia_','dynamic_dense_'))},devices=[str(x) for x in jax.devices()],jax_version=jax.__version__),destination)
   print('LATE_CAPTURE_COMPLETE',a.family,step,drift,odrift,'historical',native_replay,flush=True);del merged,arrays,losses,first,repeat,raw
  else:validate(captured,common)
  for chunk,start in enumerate(range(0,len(cases),7)):
   selected=cases[start:start+7];stem=f'cases_{chunk:03d}';done=read_remote(destination+'/'+stem+'.json')
   if done is not None:validate(done,common);assert done['cases']==selected;continue
   losses=[];case_replay={}
   for case in selected:
    value=evaluate(params,jnp.asarray(case['controls'],jnp.float32));losses.append(value)
    if case['name'] in replay.get('cases',{}):
     delta=(value.astype(float)-neutral).mean(1);rd=float(np.max(np.abs(delta-np.asarray(replay['cases'][case['name']]))));assert rd<.001,('Previous ablation mismatch',case['name'],rd);case_replay[case['name']]=rd
    print('LATE_CASE',a.family,step,case['name'],float((value-neutral).mean()),flush=True)
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again=np.asarray(ablate_fn(params,batches[0],ones))
   np.testing.assert_array_equal(again,neutral[:4]);file=folder/(stem+'.npz');save_npz(file,losses=np.stack(losses),neutral=neutral)
   publish_pair(file,folder/(stem+'.json'),dict(**common,cases=selected,repeat_neutral_exact=True,historical_case_max_differences=case_replay),destination)
  marker=folder/'complete.json';marker.write_text(json.dumps(dict(**common,total_cases=len(cases)),indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(marker),destination+'/complete.json'],check=True)
  print('LATE_CHECKPOINT_COMPLETE',a.family,step,len(cases),flush=True);del params,native_loss,neutral
 print('LATE_MODEL_COMPLETE',a.family,flush=True)
if __name__=='__main__':main()

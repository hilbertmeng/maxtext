"""Read-only scoped diagonal attention-bias counterfactuals; no optimizer or updates."""
import argparse,ast,hashlib,json,subprocess,time
from pathlib import Path
import re
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import checkpointing,max_utils
from layers.models import Transformer
from cross_scale_probe import cohort,config_for,hash_sequence,variant,save_capture

def main():
 p=argparse.ArgumentParser()
 for key in ['experiment','checkpoint','training-config','label','output','gcs-output']:p.add_argument('--'+key,required=True)
 p.add_argument('--length',type=int,default=2048);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--sequences',type=int,default=32)
 p.add_argument('--reference-json',default='/tmp/bam_stream_probe/self_bias_reference.json');p.add_argument('--seed',type=int,default=20261021);p.add_argument('--cohort',default='/tmp/bam_stream_probe/self_bias_cohort.npz');p.add_argument('--cohort-source',default='gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/cross_scale_20261025/read_gradient/cohort32.npz')
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time();Path('/tmp/bam_stream_probe',a.label).mkdir(parents=True,exist_ok=True);cfg=config_for(a)
 expected={}
 for line in Path(a.training_config).read_text().splitlines():
  key,sep,val=line.partition(': ')
  if sep:expected[key]=val
 critical={'qk_norm','normalization_layer_epsilon','normalization_layer_type','decoder_block','vocab_size','head_dim','num_decoder_layers','emb_dim','num_query_heads','num_kv_heads','mlp_dim','mlp_dim_by_block','mlp_activations','dynamic_mlp_dim','scan_layers','scan_use_mudd','dtype','weight_dtype','attention','dense_conn','fused_mlp'}
 for key,raw in expected.items():
  if key not in cfg.get_keys() or not(key in critical or key.startswith(('bam_','mudd_','pythia_'))):continue
  try:want=ast.literal_eval(raw)
  except (ValueError,SyntaxError):want=raw
  value=getattr(cfg,key)
  if isinstance(value,type):value=getattr(value,'__name__',str(value))
  if isinstance(value,tuple):value=list(value)
  if isinstance(want,tuple):want=list(want)
  if value!=want:raise ValueError(('Model config mismatch',key,want,str(value)))
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rng=jax.random.key(a.seed)
 with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
  native=max_utils.unbox_logicallypartioned
  max_utils.unbox_logicallypartioned=lambda tree:jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,tree,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  try:abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
  finally:max_utils.unbox_logicallypartioned=native
  params=checkpointing.load_params_from_path(a.checkpoint,abstract.params)
 cohort_path=Path(a.cohort);cohort_path.parent.mkdir(parents=True,exist_ok=True)
 if not cohort_path.exists():subprocess.run(['gsutil','-q','cp',a.cohort_source,str(cohort_path)],check=True)
 ref_path=Path(a.reference_json)
 if not ref_path.exists():subprocess.run(['gsutil','-q','cp',a.cohort_source.rsplit('/',1)[0]+'/reference.json',str(ref_path)],check=True)
 reference=json.loads(ref_path.read_text())
 with np.load(cohort_path) as z:data={k:z[k] for k in z.files}
 # Same label shift, mask and packing as the registered native cohort probe.
 inputs=data['tokens'][:a.sequences,:a.length];targets=data['tokens'][:a.sequences,1:a.length+1]
 mask=(data['labels'][:a.sequences,1:a.length+1]>=0).astype(np.int32)
 batches=[]
 for i in range(0,a.sequences,a.batch_size):
  shape=inputs[i:i+a.batch_size].shape
  batches.append(dict(inputs=jnp.asarray(inputs[i:i+a.batch_size]),targets=jnp.asarray(targets[i:i+a.batch_size]),mask=jnp.asarray(mask[i:i+a.batch_size]),segments=jnp.ones(shape,jnp.int32),positions=jnp.broadcast_to(jnp.arange(a.length),shape)))
 def forward(weights,b):
  value=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
  return value[0]
 import precise_ce
 precise_ce.install()
 fn=jax.jit(forward)
 def evaluate(weights):
  values=[]
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   for b in batches:values.append(np.asarray(jax.device_get(fn(weights,b)),np.float32))
  result=np.concatenate(values);assert result.shape==(a.sequences,a.length) and np.isfinite(result).all()
  return result
 key=a.label.removeprefix('self_bias_');expected=reference['models'][key]
 assert expected['experiment']==a.experiment and expected['checkpoint']==a.checkpoint
 assert np.all(data['objective_all']==1) and int(data['objective_midcontent'].sum())==reference['objective_token_counts']['midcontent']
 assert hashlib.sha256(data['objective_midcontent'].tobytes()).hexdigest()==reference['mask_sha256']
 assert cfg.bam_enabled and cfg.bam_local_full_m_read_share_down
 if cfg.get_keys().get('bam_share_read_attention_write_down',False):raise ValueError('This diagnostic needs independent attention write/read downs')
 assert sum(v.size for v in jax.tree.leaves(params))==expected['parameter_count']
 hashes=[hash_sequence(x) for x in data['tokens'][:a.sequences,:a.length+1]];assert hashes==reference['sequence_hashes']
 baseline=evaluate(params);native_error=float(np.max(np.abs(baseline.mean(1)-expected['native_losses'])))
 if native_error>.005:raise ValueError(('Native CE differs from reference',native_error))
 np.savez_compressed(out/'token_losses.npz',loss=baseline,inputs=inputs,targets=targets,mask=mask)
 import self_bias_capture as hook
 hook.install();L=int(cfg.num_decoder_layers);zero=np.zeros(L,np.float32)
 def controlled(weights,b,biases):
  hook.BIASES=biases
  try:return model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_segment_ids=b['segments'],decoder_target_mask=b['mask'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
  finally:hook.BIASES=None
 cf=jax.jit(controlled)
 def evaluate_bias(biases):
  values=[]
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   for b in batches:values.append(np.asarray(jax.device_get(cf(params,b,jnp.asarray(biases))),np.float32))
  result=np.concatenate(values);assert result.shape==baseline.shape and np.isfinite(result).all()
  return result
 neutral=evaluate_bias(zero);drift=float(np.max(np.abs(neutral.mean(1)-baseline.mean(1))))
 if drift>.005:raise ValueError(('Samegraph zero differs from native CE',drift))
 np.savez_compressed(out/'neutral.npz',loss=neutral,biases=zero)
 width=L//4;scopes={'first_quarter':list(range(width)),'last_quarter':list(range(L-width,L))};grid=[-1.,-.5,-.25,.25,.5,1.];variants=[]
 for scope,indices in scopes.items():
  for bias in grid:
   biases=zero.copy();biases[indices]=bias;loss=evaluate_bias(biases);filename=f'{scope}_{bias:+.2f}.npz'
   np.savez_compressed(out/filename,loss=loss,biases=biases)
   variants.append(dict(scope=scope,bias=bias,file=filename,layer_indices_zero_based=indices))
   subprocess.run(['gsutil','-q','cp',str(out/filename),a.gcs_output+'/'+filename],check=True)
   print('SELF_BIAS_VARIANT_COMPLETE',a.label,scope,bias,'TRAIN16_DELTA',float((loss[:16]-neutral[:16]).mean()),flush=True)
 repeated=evaluate_bias(zero);np.testing.assert_array_equal(repeated,neutral)
 meta=dict(label=a.label,experiment=a.experiment,checkpoint=a.checkpoint,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameter_count=expected['parameter_count'],sequence_hashes=hashes,sequences=a.sequences,length=a.length,batch_size=a.batch_size,native_reference_max_CE_delta=native_error,zero_native_max_CE_delta=drift,zero_native_abs_token_CE_delta_quantiles=np.quantile(np.abs(neutral-baseline),[.5,.9,.99,1]).tolist(),zero_repeat_bitwise_equal=True,query_start=64,scopes=scopes,bias_grid_including_zero=[-1.,-.5,-.25,0.,.25,.5,1.],variants=variants,config=dict(num_decoder_layers=L,num_query_heads=cfg.num_query_heads,head_dim=cfg.head_dim),selection_protocol='Choose minimum mean FP32 samegraph CE on first16texts, including neutral0, independently per scope/model. Validate selected candidate once on last16texts. No retuning on validation. Claim rescue only if val delta<-.002 and CIbelow0, then independent explicit-softmax replication required. Smaller changes numerically inconclusive. This is frozen-model counterfactual, not retraining benefit.',definition='Only q>=64. Adjust unchanged native MHA via exact diagonal-logit algebra o+expm1(b)*p_self/(1+expm1(b)*p_self)*(V_self-o), before W_o and native M writes. No parameter updates/optimizer/checkpoint writes. Scope first or last quarter. All unselected layers exact zero bias.',elapsed_seconds=time.time()-start)
 assert len(variants)==12
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');subprocess.run(['gsutil','-m','rsync','-r',str(out),a.gcs_output],check=True)
 (out/'complete.json').write_text(json.dumps(dict(label=a.label,variants=len(variants)))+'\n');subprocess.run(['gsutil','-q','cp',str(out/'complete.json'),a.gcs_output+'/complete.json'],check=True)
 print('SELF_BIAS_COMPLETE',a.label,flush=True)
if __name__=='__main__':main()

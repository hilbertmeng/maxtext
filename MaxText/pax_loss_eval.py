"""Read-only MaxText evaluator matching the historical PaxML Pile/FLAN protocol.

Each published chunk is one logical PaxML global batch (1024 examples). Smaller
physical inference batches preserve examples, masks and logical-batch metrics.
"""
import argparse,hashlib,itertools,json,os,subprocess,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import checkpointing,max_utils,pyconfig,exp
from layers.models import Transformer
from pax_loss_data import iter_records,make_batch,aggregate

EXPERIMENT='BamPythia2p8BHUnit300BFastKernelMPreNormQKScale'

def stable_ce(logits,targets,z_loss=0.):
  z=logits.astype(jnp.float32)
  loss=-jnp.sum(targets.astype(jnp.float32)*jax.nn.log_softmax(z,axis=-1),axis=-1)
  aux=jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,axis=-1)**2
  return loss+aux,aux


def setup(batch,length=2048,distributed=False):
  attrs=dict(only_eval=True,enable_checkpointing=False,dataset_type='synthetic',
      per_device_batch_size=float(batch)/jax.device_count(),eval_per_device_batch_size=float(batch)/jax.device_count(),
      target_train_batch_size=batch,max_target_length=length,max_prefill_predict_length=length,
      record_training_health_metrics=False,record_internal_nn_metrics=False,record_logits_health_metrics=False,
      log_config=False,jax_cache_dir='',enable_goodput_recording=False,monitor_goodput=False,
      base_output_directory='/tmp/pax_loss/',tensorboard_dir='/tmp/pax_loss/tb/')
  if distributed:
    attrs.update(ici_data_parallelism=-1,ici_fsdp_parallelism=1,skip_jax_distributed_system=True)
  exp.PaxLossEval=type('PaxLossEval',(getattr(exp,EXPERIMENT),),attrs)
  Path('/tmp/pax_loss/eval').mkdir(parents=True,exist_ok=True)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class='PaxLossEval',run_name='eval',
      base_output_directory='/tmp/pax_loss/',only_eval=True,enable_checkpointing=False)
  assert (cfg.emb_dim,cfg.num_decoder_layers,cfg.num_query_heads,cfg.head_dim,cfg.bam_v)==(2560,32,32,80,64)
  assert cfg.bam_m_read_norm=='rms' and cfg.bam_common_qk_read_scale and cfg.bam_attention_negative_diagonal
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  model=Transformer(cfg,mesh,quant=None);rng=jax.random.key(0)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    native=max_utils.unbox_logicallypartioned
    max_utils.unbox_logicallypartioned=lambda tree:jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,tree,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
    try:abstract,_,_=max_utils.get_abstract_state(model,None,cfg,rng,mesh,is_training=False)
    finally:max_utils.unbox_logicallypartioned=native
  return cfg,mesh,model,rng,abstract


def main():
  p=argparse.ArgumentParser()
  for arg in ['checkpoint','pile','flan','output']:p.add_argument('--'+arg,required=True)
  p.add_argument('--batch-size',type=int,default=32)
  p.add_argument('--logical-batch-size',type=int,default=1024)
  p.add_argument('--pile-batches',type=int,default=162)
  p.add_argument('--flan-batches',type=int,default=320)
  p.add_argument('--gcs-output')
  p.add_argument('--smoke-only',action='store_true')
  p.add_argument('--distributed',action='store_true')
  p.add_argument('--resume-source-commit')
  a=p.parse_args();assert a.logical_batch_size%a.batch_size==0
  out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
  if a.distributed:jax.distributed.initialize()
  cfg,mesh,model,rng,abstract=setup(a.batch_size,distributed=a.distributed)
  leader=jax.process_index()==0
  from jax.experimental import multihost_utils
  def barrier(name):
    if a.distributed:multihost_utils.sync_global_devices(name)
  max_utils.cross_entropy_with_logits=stable_ce
  source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  identity=dict(experiment=EXPERIMENT,checkpoint=a.checkpoint,source_commit=source,
      length=2048,batch_size=a.batch_size,logical_batch_size=a.logical_batch_size,
      mask='pile:all incl EOS/padding;flan:raw labels[1:]>0',precision='BF16 model/FP32 logits+CE')
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    weights=checkpointing.load_params_from_path(a.checkpoint,abstract.params)
  count=sum(v.size for v in jax.tree.leaves(weights));assert count==2775190944,count
  identity['parameter_count']=count
  (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
  def f(weights,b):
    result=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],
        decoder_target_mask=b['mask'],decoder_segment_ids=b['segments'],enable_dropout=False,
        rngs={'params':rng,'dropout':rng,'aqt':rng})
    return result[0]
  batch_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data',None)) if a.distributed else None
  forward=jax.jit(f,out_shardings=batch_sharding) if a.distributed else jax.jit(f)
  def infer(b):
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      if a.distributed:
        inputs={k:jax.make_array_from_callback(v.shape,batch_sharding,lambda idx,v=v:v[idx]) for k,v in b.items()}
        result=np.asarray(multihost_utils.process_allgather(forward(weights,inputs),tiled=True),np.float32)
      else:
        result=np.asarray(jax.device_get(forward(weights,{k:jnp.asarray(v) for k,v in b.items()})),np.float32)
    assert result.shape==b['targets'].shape and np.isfinite(result).all() and result.min()>-1e-5
    return result
  records=list(itertools.islice(iter_records(a.pile),a.batch_size));assert len(records)==a.batch_size
  b=make_batch(records,'pile');begin=time.monotonic();loss=infer(b);repeat=infer(b)
  np.testing.assert_array_equal(loss,repeat)
  changed={k:v.copy() for k,v in b.items()};changed['inputs'][:,1024:]=(changed['inputs'][:,1024:]+71)%50277
  new=infer(changed);err=float(np.max(np.abs(loss[:,:1024]-new[:,:1024])));assert err<1e-5,err
  flan_records=list(itertools.islice(iter_records(a.flan),a.batch_size));fb=make_batch(flan_records,'flan');floss=infer(fb)
  changed={k:v.copy() for k,v in fb.items()}
  for i,e in enumerate(flan_records):
    from pax_loss_data import feature_ints
    n=min(len(feature_ints(e,'input_ids'))-1,2048);changed['inputs'][i,n:]=71
  newer=infer(changed);padding=float(np.max(np.abs(newer-floss)*fb['mask']));assert padding<1e-5,padding
  smoke=dict(**identity,repeat_exact=True,future_prefix_max_error=err,padding_scored_max_error=padding,
      compile_and_smoke_seconds=time.monotonic()-begin,pile_first_batch=aggregate((loss*b['mask']).sum(1,dtype=np.float64),b['mask'].sum(1)),
      flan_first_batch=aggregate((floss*fb['mask']).sum(1,dtype=np.float64),fb['mask'].sum(1)))
  smoke['hardware']=dict(devices=jax.device_count(),processes=jax.process_count(),local_devices=jax.local_device_count(),kind=jax.devices()[0].device_kind)
  # Cross-hardware gate before accepting durable chunks from the sealed parent.
  if a.resume_source_commit:
    prior=json.loads((out/'pile/batch-0000.json').read_text());arrays=out/'pile/batch-0000.npz'
    assert prior['source_commit']==a.resume_source_commit and prior['checkpoint']==a.checkpoint
    assert hashlib.sha256(arrays.read_bytes()).hexdigest()==prior['arrays_sha256']
    with np.load(arrays) as z:
      old=z['nll_sum'][:a.batch_size]/np.maximum(z['scored_tokens'][:a.batch_size],1)
    current=(loss*b['mask']).sum(1,dtype=np.float64)/np.maximum(b['mask'].sum(1),1)
    delta=np.abs(current-old)
    assert delta.max()<.002 and abs(np.mean(current-old))<.0002,('Cross-hardware parity failed',delta.max(),np.mean(current-old))
    smoke['resume_parity']=dict(source_commit=a.resume_source_commit,examples=len(old),max_example_mean_error=float(delta.max()),mean_error=float(np.mean(current-old)))
  (out/'smoke.json').write_text(json.dumps(smoke,indent=2)+'\n');print('SMOKE_PASS',json.dumps(smoke),flush=True)
  if a.gcs_output and leader:subprocess.run(['gsutil','-q','cp',str(out/'smoke.json'),a.gcs_output+'/smoke.json'],check=True)
  barrier('smoke-uploaded')
  if a.smoke_only:return
  for dataset,num_batches in [('pile',a.pile_batches),('flan',a.flan_batches)]:
    folder=out/dataset;folder.mkdir(exist_ok=True);reader=iter_records(getattr(a,dataset));markers=[]
    for logical in range(num_batches):
      records=list(itertools.islice(reader,a.logical_batch_size));assert len(records)==a.logical_batch_size,(dataset,logical,'early EOF')
      # Hash original serialized records to preserve exact order and source content.
      digest=hashlib.sha256(b''.join(e.SerializeToString(deterministic=True) for e in records)).hexdigest()
      marker=folder/f'batch-{logical:04d}.json';arrays=folder/f'batch-{logical:04d}.npz'
      ident=dict(**identity,dataset=dataset,logical_batch=logical,records_sha256=digest)
      if marker.exists():
        old=json.loads(marker.read_text())
        allowed_sources={source,a.resume_source_commit}
        assert old.get('source_commit') in allowed_sources,'Resume source mismatch'
        assert all(old.get(k)==v for k,v in ident.items() if k!='source_commit'),'Resume identity mismatch'
        assert arrays.exists() and hashlib.sha256(arrays.read_bytes()).hexdigest()==old['arrays_sha256']
        markers.append(old);continue
      sums=[];counts=[];begin=time.monotonic()
      for start in range(0,len(records),a.batch_size):
        b=make_batch(records[start:start+a.batch_size],dataset);loss=infer(b)
        sums.extend((loss*b['mask']).sum(1,dtype=np.float64));counts.extend(b['mask'].sum(1))
      stats=aggregate(sums,counts)
      np.savez_compressed(arrays,nll_sum=np.asarray(sums,np.float64),scored_tokens=np.asarray(counts,np.int64))
      value=dict(**ident,**stats,arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),seconds=time.monotonic()-begin)
      marker.write_text(json.dumps(value,indent=2)+'\n');markers.append(value)
      if a.gcs_output and leader:
        # Data before commit marker; recovery must verify both.
        subprocess.run(['gsutil','-q','cp',str(arrays),a.gcs_output+'/'+dataset+'/'+arrays.name],check=True)
        subprocess.run(['gsutil','-q','cp',str(marker),a.gcs_output+'/'+dataset+'/'+marker.name],check=True)
      barrier(f'{dataset}-{logical}-uploaded')
      print('BATCH_COMPLETE',dataset,logical,json.dumps(stats),flush=True)
    sums=[];counts=[]
    for i in range(num_batches):
      with np.load(folder/f'batch-{i:04d}.npz') as z:sums.extend(z['nll_sum']);counts.extend(z['scored_tokens'])
    result=dict(**identity,dataset=dataset,logical_batches=num_batches,**aggregate(sums,counts),
        paxml_batch_avg_xent=float(sum(m['example_mean']*m['scored_tokens'] for m in markers)/sum(m['scored_tokens'] for m in markers)))
    (out/f'{dataset}-complete.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.gcs_output and leader:subprocess.run(['gsutil','-q','cp',str(out/f'{dataset}-complete.json'),a.gcs_output+f'/{dataset}-complete.json'],check=True)
    barrier(dataset+'-complete')
    print('DATASET_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()

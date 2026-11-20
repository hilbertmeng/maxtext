"""Full official Pile validation, T4096. Run against each model's sealed training checkout.

Only the evaluator/data modules are overlaid; no model implementation is replaced.
All processes evaluate the same global batches; only process0 writes durable chunks.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pile_validation_data import atomic_json,file_integrity,read_records
SOURCE_URI='gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_t4096_with_eos.tfrecord'
SOURCE_SHA256='f53b16029052cc35fe65adcd4694175db9cbcf72e29c5b31fd43c6c7b55cf8f6'
SOURCE_RECORDS=83240
def prepare_data(source, directory, sequence_length=2048, expected_sha256=SOURCE_SHA256,
                 expected_records=SOURCE_RECORDS):
  """Preserve original record boundaries, EOS and the short final record."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  ready = directory / 'data.json'
  if ready.exists():
    meta = json.loads(ready.read_text())
    if (meta['source_integrity']['sha256'] != expected_sha256 or
        meta['records'] != expected_records or meta['sequence_length'] != sequence_length):
      raise ValueError('Cached validation source/shape differs')
    for name in ('tokens.npy', 'lengths.npy'):
      if file_integrity(directory / name) != meta['arrays'][name]:
        raise ValueError('Cached array checksum differs')
    return meta
  integrity = file_integrity(source)
  if integrity['sha256'] != expected_sha256:
    raise ValueError('Validation TFRecord differs from the audited original')
  tokens = np.lib.format.open_memmap(directory / 'tokens.npy', mode='w+', dtype='<u2',
                                    shape=(expected_records, sequence_length + 1))
  lengths = np.lib.format.open_memmap(directory / 'lengths.npy', mode='w+', dtype='<u2',
                                     shape=(expected_records,))
  tokens[:] = 0
  count = stored = 0
  for i, row in enumerate(read_records(source)):
    if i >= expected_records or not 2 <= len(row) <= sequence_length + 1:
      raise ValueError('Unexpected validation record count or length')
    tokens[i, :len(row)] = row
    lengths[i] = len(row)
    stored += len(row)
    count += 1
  if count != expected_records:
    raise ValueError('Validation record count differs')
  tokens.flush()
  lengths.flush()
  meta = {
      'source_uri': SOURCE_URI, 'source_integrity': integrity,
      'records': count, 'stored_tokens': stored,
      'target_tokens_before_eos_mask': stored - count,
      'sequence_length': sequence_length, 'record_order': 'original; no rewindow or shuffle',
      'arrays': {name: file_integrity(directory / name) for name in ('tokens.npy','lengths.npy')},
  }
  atomic_json(ready, meta)
  return meta


def make_batch(tokens, lengths, first, batch_size, sequence_length):
  count = min(batch_size, len(lengths) - first)
  ids = np.zeros((batch_size, sequence_length + 1), dtype=np.int32)
  valid_lengths = np.zeros(batch_size, dtype=np.int32)
  ids[:count] = tokens[first:first+count]
  valid_lengths[:count] = np.asarray(lengths[first:first+count], np.int32) - 1
  positions = np.broadcast_to(np.arange(sequence_length, dtype=np.int32), (batch_size, sequence_length))
  mask = (positions < valid_lengths[:, None]).astype(np.int32)
  return {
      'inputs': ids[:, :-1], 'targets': ids[:, 1:],
      'inputs_position': positions, 'inputs_segmentation': mask,
      'targets_segmentation': mask,
  }, count



def aggregate(xent,batch):
  valid=batch['targets_segmentation'].astype(bool)
  if not np.isfinite(xent[valid]).all() or (xent[valid]<-1e-5).any():
    raise FloatingPointError('Invalid cross entropy on scored targets')
  result={}
  for policy,mask in [('including_eos',valid)]:
    result['sum_'+policy]=np.where(mask,xent,0).sum(axis=1,dtype=np.float64)
    result['count_'+policy]=mask.sum(axis=1,dtype=np.int64)
  return result


def main():
  p=argparse.ArgumentParser()
  for name in ['job','source','output']:p.add_argument('--'+name,required=True)
  p.add_argument('--batch-size',type=int,default=32)
  p.add_argument('--chunk-batches',type=int,default=32)
  p.add_argument('--max-batches',type=int,default=0)
  a=p.parse_args();job=json.loads(Path(a.job).read_text())
  import jax
  jax.distributed.initialize()
  import jax.numpy as jnp
  from jax.experimental import multihost_utils
  from flax.linen import partitioning as nn_partitioning
  from jax.sharding import Mesh,NamedSharding,PartitionSpec
  import exp,pyconfig,max_utils
  from layers.models import Transformer
  from layers import quantizations
  leader=jax.process_index()==0
  assert jax.device_count()==16 and jax.process_count()==4, (jax.device_count(),jax.process_count())
  def stable_ce(logits,targets,z_loss=0.):
    z=logits.astype(jnp.float32)
    ce=-jnp.sum(targets.astype(jnp.float32)*jax.nn.log_softmax(z,axis=-1),axis=-1)
    aux=jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,axis=-1)**2
    return ce+aux,aux
  max_utils.cross_entropy_with_logits=stable_ce
  actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  if actual!=job['code_commit']:raise ValueError('Wrong training runtime')
  out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
  attrs=dict(only_eval=True,skip_jax_distributed_system=True,max_target_length=4096,
      per_device_batch_size=a.batch_size/jax.device_count(),eval_per_device_batch_size=a.batch_size/jax.device_count(),
      ici_data_parallelism=-1,ici_fsdp_parallelism=1,
      enable_checkpointing=True,async_checkpointing=False,load_parameters_path=job['checkpoint'],
      load_full_state_path='',base_output_directory=str(out/'config')+'/',
      tensorboard_dir=str(out/'tb'),jax_cache_dir='',log_config=False,float32_logits=True)
  exp.ScalingPileEval=type('ScalingPileEval',(getattr(exp,job['run_name']),),attrs)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class='ScalingPileEval',run_name='eval')
  raw=dict(cfg.get_keys())
  for key in raw:
    if key in ['record_internal_nn_metrics','record_training_health_metrics','record_logits_health_metrics','bam_diagnostics'] or key.startswith('bam_record_'):
      raw[key]=False
  cfg=pyconfig.HyperParameters(SimpleNamespace(keys=raw))
  assert cfg.only_eval and cfg.load_parameters_path==job['checkpoint'] and not cfg.load_full_state_path
  meta=prepare_data(a.source,out/'data',4096)
  assert meta['target_tokens_before_eos_mask']==340948067
  tokens=np.load(out/'data/tokens.npy',mmap_mode='r');lengths=np.load(out/'data/lengths.npy',mmap_mode='r')
  mesh=Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  model=Transformer(cfg,mesh,quant=quantizations.configure_quantization(cfg))
  rng=jax.random.PRNGKey(cfg.init_weights_seed)
  # This established decode helper constructs/restores parameters only, no optimizer/writer.
  weights=max_utils.setup_decode_state(model,cfg,rng,mesh,None)[0].params
  count=int(max_utils.calculate_num_params_from_pytree(weights))
  identity=dict(run=job['run_name'],checkpoint=job['checkpoint'],training_commit=actual,
      evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      source_sha256=SOURCE_SHA256,records=SOURCE_RECORDS,sequence_length=4096,batch_size=a.batch_size,
      precision='native BF16 model; FP32 logits and CE; z-loss0',parameter_count=count,
      effective_config=json.loads(json.dumps(raw,default=str)))
  fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  gcs=job['gcs_output']
  from google.cloud import storage
  bucket,prefix=gcs[5:].split('/',1);dest=storage.Client(project='newproject-1-451205').bucket(bucket)
  def upload(path):
    if leader:dest.blob(prefix+'/'+path.name).upload_from_filename(str(path),checksum='crc32c',timeout=120)
  def barrier(name):multihost_utils.sync_global_devices(name)
  sharding=NamedSharding(mesh,PartitionSpec('data',None))
  def f(params,b):
    return model.apply(params,b['inputs'],b['inputs_position'],decoder_target_tokens=b['targets'],
        decoder_target_mask=b['targets_segmentation'],decoder_segment_ids=b['inputs_segmentation'],
        enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})[0]
  compiled=jax.jit(f,out_shardings=sharding)
  def infer(b):
    with mesh,nn_partitioning.axis_rules(cfg.logical_axis_rules):
      inputs={k:jax.make_array_from_callback(v.shape,sharding,lambda idx,v=v:v[idx]) for k,v in b.items()}
      return np.asarray(multihost_utils.process_allgather(compiled(weights,inputs),tiled=True),np.float32)
  b,_=make_batch(tokens,lengths,0,a.batch_size,4096);begin=time.monotonic()
  x=infer(b);repeat=infer(b);np.testing.assert_array_equal(x,repeat)
  changed={k:v.copy() for k,v in b.items()};changed['inputs'][:,2048:]=(changed['inputs'][:,2048:]+71)%50277
  y=infer(changed);future=float(np.max(np.abs(x[:,:2048]-y[:,:2048])));assert future<1e-5,future
  # Include the real short tail and padded batch rows in the numerical mask check.
  tail,_=make_batch(tokens,lengths,len(lengths)-1,a.batch_size,4096);tx=infer(tail)
  changed={k:v.copy() for k,v in tail.items()};changed['inputs'][tail['inputs_segmentation']==0]=71
  ty=infer(changed);padding=float(np.max(np.where(tail['targets_segmentation']!=0,np.abs(tx-ty),0)));assert padding<1e-5,padding
  smoke=dict(identity=identity,repeat_exact=True,future_prefix_max_error=future,padding_max_error=padding,
      seconds=time.monotonic()-begin,devices=16,processes=4)
  if leader:atomic_json(out/'smoke.json',smoke);upload(out/'smoke.json')
  print('SMOKE_PASS '+json.dumps(dict(run=job['run_name'],params=count,future=future,padding=padding)),flush=True)
  barrier('smoke')
  totals={k:0.0 if k.startswith('sum_') else 0 for k in aggregate(x,b)};first=0;chunk_index=0
  progress=out/'progress.json'
  if progress.exists():
    saved=json.loads(progress.read_text());assert saved['fingerprint']==fingerprint,'Resume identity changed'
    first=saved['records'];chunk_index=saved['chunks'];totals=saved['totals']
  batch_count=0;chunks=[];started=time.monotonic()
  def save():
    nonlocal chunk_index,chunks
    if leader:
      path=out/f'sequences_{chunk_index:05d}.npz'
      arrays={k:np.concatenate([row[k] for row in chunks]) for k in totals}
      np.savez_compressed(path,first_record=np.array(first-len(arrays['count_including_eos'])),**arrays)
      upload(path)
      chunk_sha=hashlib.sha256(path.read_bytes()).hexdigest()
      atomic_json(out/(path.stem+'.json'),dict(sha256=chunk_sha,filename=path.name,checkpoint=job['checkpoint'],fingerprint=fingerprint))
      upload(out/(path.stem+'.json'))
      atomic_json(progress,dict(fingerprint=fingerprint,identity=identity,records=first,chunks=chunk_index+1,totals=totals))
      upload(progress)
    chunk_index+=1;chunks=[];barrier('save-'+str(chunk_index))
  while first<SOURCE_RECORDS and (not a.max_batches or batch_count<a.max_batches):
    b,n=make_batch(tokens,lengths,first,a.batch_size,4096);x=infer(b)
    row={k:v[:n] for k,v in aggregate(x,b).items()}
    for k,v in row.items():totals[k]+=float(v.sum(dtype=np.float64)) if k.startswith('sum_') else int(v.sum())
    chunks.append(row);first+=n;batch_count+=1
    if len(chunks)>=a.chunk_batches:save()
    if batch_count<=2 or batch_count%100==0:print('PROGRESS '+json.dumps(dict(records=first,total=SOURCE_RECORDS,loss=totals['sum_including_eos']/totals['count_including_eos'],seconds=time.monotonic()-started)),flush=True)
  if chunks:save()
  complete=first==SOURCE_RECORDS
  if complete:
    assert totals['count_including_eos']==340948067
  result=dict(complete=complete,identity=identity,records_evaluated=first,sequence_chunks=chunk_index,
      totals=totals,policies={p:dict(loss=totals['sum_'+p]/totals['count_'+p],tokens=totals['count_'+p]) for p in ['including_eos']},
      checkpoint_writer_created=False,checkpoint_manager_created=False)
  if leader:atomic_json(out/'result.json',result);upload(out/'result.json')
  barrier('finished');print('DONE '+json.dumps(dict(complete=complete,records=first)),flush=True)

if __name__=='__main__':main()

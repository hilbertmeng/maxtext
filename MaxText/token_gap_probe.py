"""Frozen, resumable 21k per-token CE and identical-target context experiments."""
import argparse,ast,hashlib,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import checkpointing
from no_self_probe import setup,save_npz,publish_pair,read_remote
from token_gap_data import padded_batch,context_window
import token_gap_ce


def main():
  p=argparse.ArgumentParser()
  for k in ['manifest','family','config','phase','output','gcs-output']:p.add_argument('--'+k,required=True)
  a=p.parse_args();assert a.phase in ['smoke','base','context']
  m=json.loads(Path(a.manifest).read_text());info=m['models'][a.family]
  source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert source==m['commit']
  bsize=m['batch_size'];args=SimpleNamespace(experiment=info['experiment'],batch_size=bsize,length=2048,
      seed=m['seed'],label='gap21k_'+a.family,training_config=a.config)
  Path('/tmp/bam_stream_probe',args.label).mkdir(parents=True,exist_ok=True)
  cfg,critical,checked,mesh,model,rng,abstract=setup(args)
  extra={'dense_conn','dynamic_dense_type','dynamic_mlp_dim','scan_layers','scan_use_mudd','fused_qkv','fused_mlp'}
  for line in Path(a.config).read_text().splitlines():
    key,sep,raw=line.partition(': ')
    if not sep or key not in cfg.get_keys() or not(key in extra or key.startswith('dynamic_dense_')):continue
    try:want=ast.literal_eval(raw)
    except (ValueError,SyntaxError):want=raw
    actual=getattr(cfg,key);actual=list(actual) if isinstance(actual,tuple) else actual;want=list(want) if isinstance(want,tuple) else want
    assert actual==want,('Model config mismatch',key,actual,want);checked.append(key)
  assert bool(cfg.bam_enabled)==info['bam'] and cfg.num_decoder_layers==32 and cfg.num_query_heads==32 and cfg.head_dim==80
  if info['bam']:assert cfg.bam_attention_negative_diagonal and cfg.bam_partial_rope_nope_dim==80
  data={}
  for name,d in m['datasets'].items():
    path=Path(d['local']);assert hashlib.sha256(path.read_bytes()).hexdigest()==d['sha256']
    with np.load(path) as z:data[name]={k:z[k] for k in z.files}
    assert data[name]['tokens'].shape==data[name]['labels'].shape==(d['sequences'],2049)
    assert len(data[name]['lengths'])==d['sequences']
  token_gap_ce.install()
  def forward(weights,b):
    value=model.apply(weights,b['inputs'],b['positions'],decoder_target_tokens=b['targets'],decoder_target_mask=b['mask'],
        decoder_segment_ids=b['segments'],enable_dropout=False,rngs={'params':rng,'dropout':rng,'aqt':rng})
    return value[0],value[2]
  fn=jax.jit(forward)
  def run(weights,b):
    b={k:jnp.asarray(v) for k,v in b.items()}
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,pred=jax.device_get(fn(weights,b))
    loss=np.asarray(loss,np.float32);pred=np.asarray(pred,np.uint16)
    assert loss.shape==pred.shape==b['inputs'].shape and np.isfinite(loss).all() and loss.min()>=-1e-5
    return loss,pred
  def batch(d,rows,length):return padded_batch(d['tokens'],d['labels'],d['lengths'],rows,bsize,length)
  quality={}
  if a.phase=='smoke' and info.get('reference'):
    ref=info['reference'];path=Path(ref['local'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    with np.load(path) as z:old={k:z[k] for k in z.files}
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):weights=checkpointing.load_params_from_path(ref['checkpoint'],abstract.params)
    values=[]
    for start in range(0,len(old['inputs']),bsize):
      inputs=old['inputs'][start:start+bsize];targets=old['targets'][start:start+bsize];mask=old['mask'][start:start+bsize]
      assert len(inputs)==bsize
      b=dict(inputs=inputs,targets=targets,mask=mask,segments=np.ones_like(inputs),positions=np.broadcast_to(np.arange(2048,dtype=np.int32),inputs.shape))
      values.append(run(weights,b)[0])
    current=np.concatenate(values);error=float(np.max(np.abs(current.mean(1)-old['loss'].mean(1))))
    assert error<.005,('Historical FP32 CE mismatch',a.family,error)
    quality['historical_precise_CE_max_sequence_difference']=error;del weights
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):weights=checkpointing.load_params_from_path(info['checkpoint'],abstract.params)
  count=sum(x.size for x in jax.tree.leaves(weights));assert count==info['parameter_count'],(count,info['parameter_count'])
  common=dict(source_commit=source,training_commit=info['training_commit'],experiment=info['experiment'],checkpoint=info['checkpoint'],
      family=a.family,step=21000,parameter_count=count,loss_statistics_dtype='float32',batch_size=bsize,verified_config_keys=checked)
  out=Path(a.output)/a.family/a.phase;out.mkdir(parents=True,exist_ok=True);dest=a.gcs_output+'/'+a.family+'/'+a.phase
  if a.phase=='smoke':
    d=data['pile'];b,_=batch(d,np.arange(bsize),2048);loss,pred=run(weights,b);again,pred2=run(weights,b)
    np.testing.assert_array_equal(loss,again);np.testing.assert_array_equal(pred,pred2)
    perturbed={k:v.copy() for k,v in b.items()};perturbed['inputs'][:,1024:]=(perturbed['inputs'][:,1024:]+71)%50277
    changed,_=run(weights,perturbed);future_error=float(np.max(np.abs(changed[:,:1024]-loss[:,:1024])))
    assert future_error<1e-5,('Future-token leak',future_error)
    quality['future_perturbation_prefix_max_difference']=future_error
    d=data['flan'];quality['padding_and_bucket_checks']=[]
    for length in [128,256,512,1024]:
      rows=np.flatnonzero((d['lengths']-1<=length)&(d['lengths']>1))[:bsize];assert len(rows)==bsize
      full,_=batch(d,rows,2048);short,_=batch(d,rows,length);original,_=run(weights,full);reduced,_=run(weights,short)
      mask=short['mask'].astype(bool);diff=(reduced-original[:,:length])[mask]
      q=dict(length=length,tokens=int(mask.sum()),mean_difference=float(diff.mean()),max_abs=float(np.max(np.abs(diff))),abs_p95=float(np.quantile(np.abs(diff),.95)))
      assert abs(q['mean_difference'])<.005 and q['abs_p95']<.05 and q['max_abs']<.2,('Right-padding/shape sensitivity',q)
      quality['padding_and_bucket_checks'].append(q)
    d=data['pile'];timings=[]
    for start in [bsize,2*bsize,3*bsize]:
      b,_=batch(d,np.arange(start,start+bsize),2048);begin=time.monotonic();run(weights,b);timings.append(time.monotonic()-begin)
    quality.update(repeat_exact=True,seconds_per_batch=timings,tokens_per_second=float(bsize*2048/np.median(timings)))
    marker=out/'complete.json';marker.write_text(json.dumps(dict(**common,quality=quality),indent=2)+'\n')
    subprocess.run(['gsutil','-q','cp',str(marker),dest+'/complete.json'],check=True)
    print('TOKEN_GAP_SMOKE_COMPLETE',a.family,json.dumps(quality),flush=True);return
  if a.phase=='base':
    for name,d in data.items():
      folder=out/name;folder.mkdir(parents=True,exist_ok=True);destination=dest+'/'+name
      existing=subprocess.run(['gsutil','ls',destination+'/chunk_*.json'],capture_output=True,text=True,timeout=60)
      if existing.returncode and not any(s in existing.stderr for s in ['matched no objects','No URLs matched','NotFound','404']):raise RuntimeError(existing.stderr[-1000:])
      markers=set(existing.stdout.splitlines());n=len(d['tokens']);all_chunks=[]
      for start in range(0,n,m['chunk_size']):
        stop=min(start+m['chunk_size'],n);stem=f'chunk_{start:06d}';marker_url=destination+'/'+stem+'.json'
        identity=dict(**common,dataset=name,cohort_sha256=m['datasets'][name]['sha256'],start=start,stop=stop)
        if marker_url in markers:
          previous=read_remote(marker_url)
          for k,v in identity.items():assert previous[k]==v,('Resume identity mismatch',k)
          all_chunks.append(previous);print('TOKEN_GAP_RESUME',a.family,name,start,flush=True);continue
        begin=time.monotonic();loss=np.zeros((stop-start,2048),np.float32);prediction=np.zeros((stop-start,2048),np.uint16)
        rows=np.arange(start,stop)
        buckets=np.full(len(rows),2048,np.int32) if name=='pile' else np.maximum(128,2**np.ceil(np.log2(np.maximum(d['lengths'][rows]-1,1))).astype(np.int32))
        assert buckets.max()<=2048
        for length in sorted(np.unique(buckets)):
          group=rows[buckets==length]
          for offset in range(0,len(group),bsize):
            selected=group[offset:offset+bsize];b,real=batch(d,selected,int(length));z,pred=run(weights,b)
            loss[selected-start,:length]=z[:real];prediction[selected-start,:length]=pred[:real]
        path=folder/(stem+'.npz');save_npz(path,loss=loss,prediction=prediction)
        mask=d['labels'][start:stop,1:]>=0;targets=int(mask.sum());mean=float(loss[mask].mean())
        metadata=dict(**identity,seconds=time.monotonic()-begin,target_tokens=targets,mean_CE=mean,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        publish_pair(path,folder/(stem+'.json'),metadata,destination);all_chunks.append(metadata)
        print('TOKEN_GAP_BASE_CHUNK',a.family,name,start,stop,'CE',mean,'seconds',metadata['seconds'],flush=True)
      marker=folder/'complete.json';marker.write_text(json.dumps(dict(**common,dataset=name,cohort_sha256=m['datasets'][name]['sha256'],sequences=n,chunks=len(all_chunks),target_tokens=sum(q['target_tokens'] for q in all_chunks)),indent=2)+'\n')
      subprocess.run(['gsutil','-q','cp',str(marker),destination+'/complete.json'],check=True)
    print('TOKEN_GAP_BASE_COMPLETE',a.family,flush=True);return
  plan=m['context_plan'];path=Path(plan['local']);assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['sha256']
  with np.load(path) as z:queries={k:z[k] for k in z.files}
  contexts=[1024,512,256,128];losses=np.empty((len(queries['row']),4),np.float32);predictions=np.empty_like(losses,dtype=np.uint16)
  for ci,c in enumerate(contexts):
    for start in range(0,len(queries['row']),bsize):
      stop=min(start+bsize,len(queries['row']));batch_items=[];scores=[]
      for q in range(start,stop):
        dataset=str(queries['dataset'][q]);row=int(queries['row'][q]);j=int(queries['target_index'][q]);d=data[dataset]
        assert d['labels'][row,j]>=0 and d['tokens'][row,j]==queries['target_token'][q] and j>=1024
        x,y,n,score,pos=context_window(d['tokens'][row,:d['lengths'][row]],j,c,pad_to=1024)
        active=np.arange(1024)<n;batch_items.append(dict(inputs=x,targets=y,positions=pos,segments=active.astype(np.int32),mask=active.astype(np.int32)));scores.append(score)
      real=len(batch_items)
      while len(batch_items)<bsize:batch_items.append(batch_items[0]);scores.append(scores[0])
      b={k:np.stack([v[k] for v in batch_items]) for k in batch_items[0]};z,pred=run(weights,b)
      losses[start:stop,ci]=z[np.arange(real),np.asarray(scores[:real])];predictions[start:stop,ci]=pred[np.arange(real),np.asarray(scores[:real])]
    print('TOKEN_GAP_CONTEXT',a.family,c,float(losses[:,ci].mean()),flush=True)
  path=out/'context_losses.npz';save_npz(path,loss=losses,prediction=predictions,contexts=np.asarray(contexts))
  publish_pair(path,out/'context_losses.json',dict(**common,plan_sha256=plan['sha256'],queries=len(losses),contexts=contexts,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),dest)
  marker=out/'complete.json';marker.write_text(json.dumps(dict(**common,queries=len(losses),plan_sha256=plan['sha256']),indent=2)+'\n');subprocess.run(['gsutil','-q','cp',str(marker),dest+'/complete.json'],check=True)
  print('TOKEN_GAP_CONTEXT_COMPLETE',a.family,flush=True)


if __name__=='__main__':main()

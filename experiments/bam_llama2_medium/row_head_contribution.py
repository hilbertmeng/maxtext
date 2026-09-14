"""Exact layer/path/head row knockout with validated scalar/async/device-loop dispatch.

One restored checkpoint, immutable parameters, batch1 scalar forward semantics.
The experiment masks only the final BAM row output of one MHA head.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib,hashlib,json,os,subprocess,time
import numpy as np
import row_contribution as base
from flax import linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp
from absl import app

PATHS=base.PATHS
SHAPE=(24,4,16)
TRAINING={
 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow':'77401da6f83a5aa6ddd61994e028c3c694221518',
 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis':'c664e8211c299c9c780ce1667553cf995130c453'}

def mask_row_head(read,keep,bam_k):
 assert read.shape[-2]==16 and read.shape[-1]>bam_k
 assert keep.shape==(16,)
 return jnp.concatenate((read[...,:bam_k],read[...,bam_k:]*keep[...,None].astype(read.dtype)),axis=-1)

@contextlib.contextmanager
def head_interventions(keep):
 layers=[]
 def intercept(next_fun,args,kw,ctx):
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='__call__':
   assert kw.get('layer_index') is not None;layers.append(kw['layer_index'])
   try:return next_fun(*args,**kw)
   finally:layers.pop()
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='_read_local':
   name=args[0] if args else kw['name'];path={'q':0,'k':1,'v':2}[name]
   return mask_row_head(next_fun(*args,**kw),keep[layers[-1],path],ctx.module.bam_k)
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='_read_fetched_m':
   assert not kw.get('ungated',False)
   assert ctx.module._fetched_read_num_heads==ctx.module.num_query_heads==16
   read,gate=next_fun(*args,**kw)
   return mask_row_head(read,keep[layers[-1],3],ctx.module.bam_k),gate
  return next_fun(*args,**kw)
 with nn.intercept_methods(intercept):yield

def forward(model,params,batch,rng,keep):
 with head_interventions(keep):return base.forward(model,params,batch,rng)

def scenarios():
 out=[dict(name='native',kind='control',coalition=0)]
 out += [dict(name=f'all_{p}',kind='control',path=p,coalition=1<<i) for i,p in enumerate(PATHS)]
 out += [dict(name=f'head_{l:02d}_{p}_{h:02d}',kind='head',layer=l,path=p,head=h)
         for l in range(1,24) for p in PATHS if not(p=='V' and l%3==2) for h in range(16)]
 assert len(out)==1349
 return out

def mask_for(s):
 keep=np.ones(SHAPE,bool)
 if s['kind']=='control':
  for i,p in enumerate(PATHS):
   if s['coalition']&(1<<i):keep[:,i,:]=False
 elif s['kind']=='head':keep[s['layer'],PATHS.index(s['path']),s['head']]=False
 elif s['kind']=='group':
  for l,p,h in s['remove']:keep[l,PATHS.index(p),h]=False
 else:raise ValueError(s)
 return keep

def run(config):
 out=Path(os.environ['HEAD_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 reference_path=Path(os.environ['HEAD_REFERENCE']);reference=np.load(reference_path)
 with np.load(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz')) as f:cohort={k:np.asarray(f[k]) for k in base.KEYS}
 hashes=base.digest_cohort(cohort)
 shard=int(os.environ.get('HEAD_SHARD','0'));shards=int(os.environ.get('HEAD_SHARDS','1'))
 stage=os.environ.get('HEAD_STAGE','heads');size=int(os.environ.get('HEAD_CHUNK','16'))
 if os.environ.get('HEAD_SCENARIOS'):
  all_scenarios=json.loads(Path(os.environ['HEAD_SCENARIOS']).read_text())
 else:all_scenarios=scenarios()
 use=[s for i,s in enumerate(all_scenarios) if s['kind']=='control' or i%shards==shard]
 keeps=np.stack([mask_for(s) for s in use]);assert len(use)>=5
 start=int(os.environ.get('HEAD_START','0'));stop=int(os.environ.get('HEAD_STOP','64'));assert 0<=start<stop<=64
 assert list(reference['sequence_hash'])==[h['inputs'] for h in hashes[:64]]
 metadata=dict(model=base.BASE,checkpoint=config.load_parameters_path,training_commit=TRAINING[base.BASE],
  runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
  scenarios_sha256=hashlib.sha256(json.dumps(use,sort_keys=True).encode()).hexdigest(),
  sequence_hashes=hashes[:64],shape=SHAPE,shard=shard,shards=shards,start=start,stop=stop,stage=stage,chunk=size,
  method='Exact post-gate BAM row-head output deletion, other heads/columns/paths unchanged; ordinary downstream forward',
  skipped='layer0 all paths known zero; F LocalV nonexistent',
  overrides={k:getattr(config,k) for k in ('only_eval','record_internal_nn_metrics','record_training_health_metrics','bam_record_local_routing_metrics','bam_record_fetched_read_health_metrics')})
 def save_metadata():(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 save_metadata();(out/'scenarios.json').write_text(json.dumps(use,indent=2))
 rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
 state,_,_,_=base.max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
 one=jax.jit(lambda p,b,k:forward(model,p,b,rng,k))
 ordinary=jax.jit(lambda p,b:base.forward(model,p,b,rng))
 loop=jax.jit(lambda p,b,ks:jax.lax.map(lambda k:forward(model,p,b,rng,k),ks))
 mode=None;device_keeps=jnp.asarray(keeps)
 def dispatch(batch,part,mode):
  if mode=='loop':return np.asarray(loop(state.params,batch,part))
  if mode=='async':return np.stack(jax.device_get([one(state.params,batch,part[i]) for i in range(len(part))]))
  return np.stack([np.asarray(one(state.params,batch,part[i])) for i in range(len(part))])
 with mesh,partitioning.axis_rules(config.logical_axis_rules):
  for index in range(start,stop):
   path=out/f'{stage}_{index:03d}.npz'
   if path.exists():
    with np.load(path) as previous:assert str(previous['sequence_hash'])==hashes[index]['inputs']
    continue
   begun=time.perf_counter();batch={k:jnp.asarray(v[index:index+1]) for k,v in cohort.items()}
   native=np.asarray(one(state.params,batch,jnp.ones(SHAPE,bool)))
   if mode is None:
    ordinary_value=np.asarray(ordinary(state.params,batch))
    print('BASELINE_CHECK',index,'head',native,'ordinary',ordinary_value,'reference',reference['loss'][index,0],flush=True)
    np.testing.assert_allclose(ordinary_value,reference['loss'][index,0],atol=1e-6,rtol=0)
   np.testing.assert_allclose(native,reference['loss'][index,0],atol=1e-6,rtol=0)
   if mode is None:
    np.testing.assert_allclose(native,np.asarray(ordinary(state.params,batch)),atol=1e-6,rtol=0)
    noop=np.ones(SHAPE,bool);noop[0,:,:]=False;noop[2::3,2,:]=False
    np.testing.assert_array_equal(np.asarray(one(state.params,batch,jnp.asarray(noop))),native)
    check=device_keeps[:min(size,len(use))]
    if len(check)<size:check=jnp.concatenate([check,jnp.ones((size-len(check),)+SHAPE,bool)])
    serial=dispatch(batch,check,'scalar');np.testing.assert_allclose(serial[:5,0],reference['loss'][index],atol=1e-6,rtol=0)
    timings={};gaps={}
    # Device loop lowering changed numerical results in the full model; opt in only for revalidation.
    for candidate in ['scalar','async'] + (['loop'] if os.environ.get('HEAD_TRY_LOOP','0')=='1' else []):
     values=dispatch(batch,check,candidate);gap=float(np.max(np.abs(values-serial)));gaps[candidate]=gap
     if not np.allclose(values,serial,atol=1e-6,rtol=0):print('DISPATCH_REJECT',candidate,gap,flush=True);continue
     begun_t=time.perf_counter()
     for _ in range(3):dispatch(batch,check,candidate)
     timings[candidate]=3*size/(time.perf_counter()-begun_t)
    mode=max(timings,key=timings.get)
    metadata.update(effective_dispatch=mode,benchmark_variants_per_second=timings,dispatch_max_abs_gap=gaps)
    save_metadata();print('FIRST_STEP HEAD_NOOP_OK',index,mode,timings,flush=True)
   values=[]
   for begin in range(0,len(use),size):
    part=device_keeps[begin:begin+size];n=len(part)
    if n<size:part=jnp.concatenate([part,jnp.ones((size-n,)+SHAPE,bool)])
    predicted=dispatch(batch,part,mode)
    if begin==0:
     # Every sequence checks all four whole-path controls against the original study,
     # plus one individual-head intervention against the scalar executable.
     np.testing.assert_allclose(predicted[:5,0],reference['loss'][index],atol=1e-6,rtol=0)
     if n>5:np.testing.assert_allclose(predicted[5],np.asarray(one(state.params,batch,part[5])),atol=1e-6,rtol=0)
    values.extend(predicted[:n])
   loss=np.stack(values);assert np.isfinite(loss).all()
   temp=out/f'.pending_{stage}_{index:03d}.npz'
   np.savez_compressed(temp,loss=loss,baseline=native,gap=loss-native,sequence_hash=hashes[index]['inputs'],tokens=int(np.sum(cohort['targets_segmentation'][index]!=0)))
   temp.replace(path)
   print(f'HEAD_DONE sample={index} scenarios={len(use)} seconds={time.perf_counter()-begun:.2f} mode={mode}',flush=True)
 if writer:writer.flush()
 print('HEAD_STAGE_DONE',stage,shard,start,stop,flush=True)
if __name__=='__main__':app.run(lambda argv:run(base.pyconfig.initialize(argv)))

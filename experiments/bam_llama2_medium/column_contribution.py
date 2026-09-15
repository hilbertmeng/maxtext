"""Paired column necessity and row/column interactions, without head ablations.

The side axis is [column, row]. Production source and checkpoint stay immutable.
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

SHAPE=(24,4,2)
PATHS=base.PATHS
TRAINING={'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow':'77401da6f83a5aa6ddd61994e028c3c694221518','BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis':'c664e8211c299c9c780ce1667553cf995130c453'}

def scale_sides(read,scale,bam_k):
 assert read.shape[-1]>bam_k and scale.shape==(2,)
 return jnp.concatenate((read[...,:bam_k]*scale[0].astype(read.dtype),read[...,bam_k:]*scale[1].astype(read.dtype)),axis=-1)

@contextlib.contextmanager
def side_interventions(scales):
 layers=[]
 def intercept(next_fun,args,kw,ctx):
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='__call__':
   assert kw.get('layer_index') is not None;layers.append(kw['layer_index'])
   try:return next_fun(*args,**kw)
   finally:layers.pop()
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='_read_local':
   name=args[0] if args else kw['name'];p={'q':0,'k':1,'v':2}[name]
   return scale_sides(next_fun(*args,**kw),scales[layers[-1],p],ctx.module.bam_k)
  if isinstance(ctx.module,base.attentions.BamAttention) and ctx.method_name=='_read_fetched_m':
   assert not kw.get('ungated',False)
   read,gate=next_fun(*args,**kw)
   return scale_sides(read,scales[layers[-1],3],ctx.module.bam_k),gate
  return next_fun(*args,**kw)
 with nn.intercept_methods(intercept):yield

def forward(model,params,batch,rng,scales):
 with side_interventions(scales):return base.forward(model,params,batch,rng)

def odepth_scenarios():
 out=[]
 for name,start,stop in [('native',0,0),('O_all_off',0,24),('O_first_half',0,12),('O_last_half',12,24),('O_first_third',0,8),('O_middle_third',8,16),('O_last_third',16,24)]:
  a=np.ones((24,4),np.float32);a[start:stop,3]=0.
  out.append(dict(name=name,kind='odepth',start=start,stop=stop,scales=a.tolist()))
 return out

def scenarios():
 out=[]
 def add(side,stage,s):
  a=np.ones(SHAPE,np.float32)
  for axis in ([0,1] if side=='both' else [0] if side=='column' else [1]):a[:,:,axis]=s['scales']
  out.append(dict(id=f'{side}/{stage}/{s["name"]}',side=side,stage=stage,**{k:v for k,v in s.items() if k!='scales'},scales=a.tolist()))
 original=base.scenarios()
 for s in original[:16]:add('row','all',s)
 for stage,ss in [('all',original),('targeted',base.targeted_scenarios()),('depth',base.depth_scenarios()),('vdepth',base.vdepth_scenarios()),('qktype',base.qktype_scenarios()),('odepth',odepth_scenarios())]:
  for s in ss:add('column',stage,s)
 # Same-scope paired controls allow exact interaction = both - row - column.
 for s in original:
  if s['kind'] in ('coalition','layer','unit'):add('both','all',s)
 for stage,ss in [('targeted',base.targeted_scenarios()),('depth',base.depth_scenarios()),('vdepth',base.vdepth_scenarios()),('qktype',base.qktype_scenarios()),('odepth',odepth_scenarios())]:
  for s in ss:add('both',stage,s)
 # O depth had no earlier row experiment; measure it now for a matched pair.
 for s in odepth_scenarios():add('row','odepth',s)
 assert len({s['id'] for s in out})==len(out)
 return out

def run(config):
 out=Path(os.environ['COLUMN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 cohort_path=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
 with np.load(cohort_path) as f:cohort={k:np.asarray(f[k]) for k in base.KEYS}
 reference_path=Path(os.environ['COLUMN_REFERENCE']);reference=np.load(reference_path)
 hashes=base.digest_cohort(cohort)[:64];assert list(reference['sequence_hash'])==[h['inputs'] for h in hashes]
 use=scenarios();unique=[];index={};mapping=[]
 for s in use:
  a=np.asarray(s['scales'],np.float32);a[0,:,:]=1.;a[2::3,2,:]=1. # independently verify these exact no-ops below
  key=a.tobytes()
  if key not in index:index[key]=len(unique);unique.append(a)
  mapping.append(index[key])
 scales=jnp.asarray(np.stack(unique));start=int(os.environ.get('COLUMN_START','0'));stop=int(os.environ.get('COLUMN_STOP','64'));assert 0<=start<stop<=64
 metadata=dict(model=base.BASE,checkpoint=config.load_parameters_path,training_commit=TRAINING[base.BASE],runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),sequence_hashes=hashes,start=start,stop=stop,scenarios=len(use),unique_forwards=len(unique),scenario_mapping=mapping,side_order=['column','row'],shape=SHAPE,only_eval=config.only_eval,method='post-gate/mixing side output scaling, all positions, full downstream recomputation',shape_config={k:getattr(config,k) for k in ('emb_dim','num_decoder_layers','num_query_heads','head_dim','bam_k','bam_v','bam_local_q_rank','bam_local_v_rank','bam_local_qk_share_basis')})
 (out/'scenarios.json').write_text(json.dumps(use,indent=2))
 def save():(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 save()
 rng,writer,manager,mesh,model,_,tx=base.train.setup_mesh_and_model(config)
 state,_,_,_=base.max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
 one=jax.jit(lambda p,b,s:forward(model,p,b,rng,s));ordinary=jax.jit(lambda p,b:base.forward(model,p,b,rng));mode=None
 def dispatch(batch,part,mode):
  if mode=='async':return np.stack(jax.device_get([one(state.params,batch,s) for s in part]))
  return np.stack([np.asarray(one(state.params,batch,s)) for s in part])
 with mesh,partitioning.axis_rules(config.logical_axis_rules):
  for i in range(start,stop):
   file=out/f'column_{i:03d}.npz'
   if file.exists():
    with np.load(file) as old:assert str(old['sequence_hash'])==hashes[i]['inputs']
    continue
   begun=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()};native=np.asarray(one(state.params,batch,jnp.ones(SHAPE,np.float32)))
   np.testing.assert_allclose(native,reference['loss'][i,0],atol=1e-6,rtol=0)
   if mode is None:
    orig=np.asarray(ordinary(state.params,batch));print('NATIVE_CHECK',native,orig,reference['loss'][i,0],flush=True);np.testing.assert_allclose(native,orig,atol=1e-6,rtol=0)
    noop=np.ones(SHAPE,np.float32);noop[0,:,:]=0.;noop[2::3,2,:]=0.;np.testing.assert_array_equal(np.asarray(one(state.params,batch,jnp.asarray(noop))),native)
    check=jnp.concatenate((scales[:16],scales[16:24]));serial=dispatch(batch,check,'scalar');timings={}
    for candidate in ('scalar','async'):
     values=dispatch(batch,check,candidate);np.testing.assert_array_equal(values,serial)
     begin=time.perf_counter()
     for _ in range(3):dispatch(batch,check,candidate)
     timings[candidate]=3*len(check)/(time.perf_counter()-begin)
    mode=max(timings,key=timings.get);metadata.update(dispatch=mode,benchmarks=timings);save();print('FIRST_STEP COLUMN_NOOP_OK',len(unique),mode,timings,flush=True)
   vals=[]
   for j in range(0,len(scales),16):vals.extend(dispatch(batch,scales[j:j+16],mode))
   unique_loss=np.stack(vals);loss=unique_loss[mapping];assert np.isfinite(loss).all()
   np.testing.assert_allclose(loss[:16,0],reference['loss'][i],atol=1e-6,rtol=0)
   # Per-sequence column intervention agrees with the serial executable as well.
   np.testing.assert_array_equal(loss[17],np.asarray(one(state.params,batch,jnp.asarray(use[17]['scales'],np.float32))))
   temp=out/f'.pending_column_{i:03d}.npz';np.savez_compressed(temp,loss=loss,baseline=native,sequence_hash=hashes[i]['inputs'],tokens=int(np.sum(cohort['targets_segmentation'][i]!=0)));temp.replace(file)
   print(f'COLUMN_DONE sample={i} unique={len(unique)} scenarios={len(use)} seconds={time.perf_counter()-begun:.2f}',flush=True)
 if writer:writer.flush()
 print('COLUMN_STAGE_DONE',start,stop,flush=True)
if __name__=='__main__':app.run(lambda argv:run(base.pyconfig.initialize(argv)))

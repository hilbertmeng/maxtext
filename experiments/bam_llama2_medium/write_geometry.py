"""Read-only AllLocal four-component / write-gate capture at the trained runtime."""
import contextlib,hashlib,json,os,sys,time,subprocess,concurrent.futures
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'MaxText'))
from absl import app
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning
from flax.traverse_util import flatten_dict
import exp,pyconfig,train,max_utils
from layers import attentions
BASE='BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8IndependentGatesK48QK48MLPPerLayerAllLocal'
KEYS=('inputs','targets','inputs_position','inputs_segmentation','targets_segmentation')
GRAM=[(i,j) for i in range(4) for j in range(i,4)]
FIELDS=[f'gram_{i}{j}' for i,j in GRAM]+['write_gate','self_alpha','key_cos','content_cos','old_at_unit_p_norm','o_write_at_unit_p_norm','write_ratio','relative_parallel','p_norm','r_norm','v_local_norm','o_local_norm','sum_norm','reconstruction_relative','key_content_relative_error','read_v_gate','read_o_gate']
class GeometryProbe(getattr(exp,BASE)):
 only_eval=True
 per_device_batch_size=1
 eval_per_device_batch_size=1
 load_parameters_path=f'gs://newproject-1-llm_projects_us-east5/log/{BASE}/checkpoints/13500/items'
 record_internal_nn_metrics=False
 record_training_health_metrics=False
for cls in getattr(exp,BASE).__mro__:
 for k in vars(cls):
  if k.startswith('bam_record_'):setattr(GeometryProbe,k,False)
exp.GeometryProbe=GeometryProbe

def dot(x,y):return jnp.sum(x.astype(jnp.float32)*y.astype(jnp.float32),axis=-1)
def norm(x):return jnp.sqrt(jnp.maximum(dot(x,x),0.))
def ratio(a,b):return jnp.where(b>1e-20,a/jnp.maximum(b,1e-20),jnp.nan)
def cosine(a,b):return jnp.clip(ratio(dot(a,b),norm(a)*norm(b)),-1,1)
def features(parts,actual,gate,alpha,r,p,M,vlocal,vg,og,ungated):
 # All norms include the real current forward's total-data RMS denominator.
 a,b,c,o=parts;actual=actual.astype(jnp.float32);p=p.astype(jnp.float32)
 denom=jnp.sqrt(jnp.mean(actual**2,axis=-1)+1e-6)
 write_o=o*gate[...,None]/denom[...,None]
 pn=norm(p);old=jnp.einsum('btkv,btnv->btnk',M.astype(jnp.float32),p,precision=jax.lax.Precision.HIGHEST)/pn[...,None]
 added=write_o*pn[...,None]
 oldn=norm(old);addedn=norm(added);rc=jnp.einsum('btkv,btnv->btnk',M.astype(jnp.float32),r,precision=jax.lax.Precision.HIGHEST)
 values=[dot(parts[i],parts[j]) for i,j in GRAM]
 values += [gate,alpha,cosine(r,p),cosine(added,old),oldn,addedn,ratio(addedn,oldn),ratio(dot(added,old),oldn**2),pn,norm(r),norm(vlocal),norm(o),norm(actual),ratio(norm(sum(parts)-actual),norm(actual)),ratio(norm(rc-ungated),norm(ungated)),vg,og]
 return jnp.stack(values,axis=-1)

@contextlib.contextmanager
def capture():
 stack=[];old_op=attentions._attention_op
 def op(query,key,value,valid,**kw):
  y,alpha=old_op(query,key,value,valid,**kw)
  if stack:
   d=stack[-1];q0,s0=d['slice'];q1=q0+query.shape[1];s1=s0+key.shape[1];af=alpha.astype(jnp.float32)
   raw=d['raw'][:,s0:s1,:,:48].astype(jnp.float32);v=d['v'][:,s0:s1,:,:48].astype(jnp.float32)
   a=jnp.einsum('bnqs,bsnk->bqnk',af,raw,precision=jax.lax.Precision.HIGHEST)
   diag=(jnp.arange(q0,q1)[:,None]==jnp.arange(s0,s1)[None,:]);selfalpha=jnp.sum(jnp.where(diag[None,None],af,0),-1).transpose(0,2,1)
   b=selfalpha[...,None]*d['v'][:,q0:q1,:,:48].astype(jnp.float32)
   c=jnp.einsum('bnqs,bsnk->bqnk',jnp.where(diag[None,None],0,af),v,precision=jax.lax.Precision.HIGHEST)
   d['parts'].append((a,b,c,selfalpha))
  return y,alpha
 def hook(next_fun,args,kw,ctx):
  m=ctx.module;name=ctx.method_name
  if not isinstance(m,attentions.BamAttention):return next_fun(*args,**kw)
  if name=='__call__':
   assert not m.is_initializing();stack.append(dict(layer=kw['layer_index'],parts=[]))
   try:return next_fun(*args,**kw)
   finally:stack.pop()
  if not stack:return next_fun(*args,**kw)
  d=stack[-1]
  if name=='_standard_value_projection':
   result=next_fun(*args,**kw);d['raw']=result;return result
  if name=='_independent_local_vo':
   result=next_fun(*args,**kw);d['v'],d['o']=result
   x=args[1];d['vg']=m._read_gate_activation(m._project_read_gate_logits('W_lv_gate',x))[...,0]
   d['og']=m._read_gate_activation(m._project_read_gate_logits('W_R_gate',x,squeeze_fetch_axis=True))[...,0]
   return result
  if name=='_read_fetched_m':
   result=next_fun(*args,**kw)
   if kw.get('ungated'):
    raw=jnp.squeeze(m.W_R(args[1]),axis=-2);_,q=attentions._split_read_keys(raw,m._fetched_arm_ungated)
    d['r']=jnp.einsum('vc,btnc->btnv',m.abs_v_cache_projection.astype(jnp.float32),q.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    d['ungated']=result[0][0].astype(jnp.float32)
   return result
  if name=='_attention_block':
   d['slice']=(kw['q0'],kw['s0']);return next_fun(*args,**kw)
  if name=='_write':
   result=next_fun(*args,**kw);ohead,x,M=args
   assert m.bam_k==48 and not m._concat_write_mix and m._write_factor_norm=='rms'
   assert m._rms_epsilon==1e-6 and not m.config.bam_sqrt_n_scale and m.config.bam_lambda_decay==1.
   a,b,c,alpha=[jnp.concatenate([v[i] for v in d['parts']],axis=1) for i in range(4)]
   p=m.write_address_norm(m._write_address(x))
   f=features((a,b,c,d['o'][...,:48].astype(jnp.float32)),ohead[...,:48],result[1],alpha,d['r'],p,M,d['v'][...,:48],d['vg'],d['og'],d['ungated'])
   m.sow('intermediates','geometry',f);m.sow('intermediates','geometry_layer',d['layer'])
   return result
  return next_fun(*args,**kw)
 attentions._attention_op=op
 try:
  with nn.intercept_methods(hook):yield
 finally:attentions._attention_op=old_op

def run(config):
 out=Path(os.environ['GEOMETRY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 cohortpath=Path(os.environ.get('ROW_COHORT','/tmp/pile_eval_cohort.npz'))
 with np.load(cohortpath) as f:cohort={k:f[k] for k in KEYS}
 n=int(os.environ.get('GEOMETRY_N','64'));assert n in [64,128]
 meta=dict(model=BASE,checkpoint=config.load_parameters_path,training_commit='ba299404f17b366a4e41bce5fef9306f5dfe8a17',runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),n=n,fields=FIELDS,cohort_sha256=hashlib.sha256(cohortpath.read_bytes()).hexdigest(),sequence_hashes=[{k:hashlib.sha256(np.ascontiguousarray(v[i]).tobytes()).hexdigest() for k,v in cohort.items()} for i in range(n)],shape=[24,2048,16,len(FIELDS)])
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));np.save(out/'valid.npy',cohort['inputs_segmentation'][:n]!=0);np.save(out/'positions.npy',cohort['inputs_position'][:n]);np.save(out/'tokens.npy',cohort['inputs'][:n])
 rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(config)
 state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,config,rng,mesh,manager)
 def forward(p,b,collect):
  with capture() if collect else contextlib.nullcontext():
   output,inter=model.apply(p,b['inputs'],b['inputs_position'],decoder_segment_ids=b['inputs_segmentation'],decoder_target_mask=b['targets_segmentation'],decoder_target_tokens=b['targets'],enable_dropout=False,rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
  mask=b['targets_segmentation']!=0;loss=(output[0]*mask).sum()/jnp.maximum(mask.sum(),1)
  if not collect:return loss
  flat=flatten_dict(inter);fs=[];ids=[]
  for k,v in flat.items():
   if k[-1]=='geometry':
    arr=v[0];ls=flat[k[:-1]+('geometry_layer',)][0];print('CAPTURE_LAYOUT',k,arr.shape,ls.shape,flush=True)
    fs.append(arr.reshape((-1,2048,16,len(FIELDS))));ids.append(ls.reshape(-1))
  fs=jnp.concatenate(fs);ids=jnp.concatenate(ids);return loss,fs[jnp.argsort(ids)],jnp.sort(ids)
 ordinary=jax.jit(lambda p,b:forward(p,b,False));collect=jax.jit(lambda p,b:forward(p,b,True));checks=[]
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=4);pending=[]
 def save_sample(i,f,check):
  temp=out/f'.pending_{i:03d}.npy';np.save(temp,f);temp.replace(out/f'sample_{i:03d}.npy')
  qi=[FIELDS.index(k) for k in ['write_gate','reconstruction_relative','write_ratio','relative_parallel']]
  stats=np.nanquantile(f[...,qi],[.01,.1,.5,.9,.99],axis=(1,2))
  np.save(out/f'quick_{i:03d}.npy',stats)
  (out/f'fidelity_{i:03d}.json').write_text(json.dumps(check))
 with mesh,partitioning.axis_rules(config.logical_axis_rules):
  for i in range(n):
   dest=out/f'sample_{i:03d}.npy'
   if dest.exists():continue
   t=time.perf_counter();batch={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()}
   loss,f,ids=jax.device_get(collect(state.params,batch));np.testing.assert_array_equal(ids,np.arange(24));assert f.shape==tuple(meta['shape'])
   native=float(ordinary(state.params,batch))
   # Keep compiler fidelity explicit; never confuse drift with intervention damage.
   checks.append(dict(sequence=i,ordinary=native,capture=float(loss),difference=float(loss)-native))
   assert abs(float(loss)-native)<.01,checks[-1]
   assert np.isfinite(f[...,:12]).all()
   pending.append(pool.submit(save_sample,i,f,checks[-1]))
   if len(pending)>=4:pending.pop(0).result()
   if i==0:print('FIRST_STEP GEOMETRY_CAPTURE_OK',checks[-1],flush=True)
   print('GEOMETRY_DONE',i,'seconds',time.perf_counter()-t,flush=True)
 for job in pending:job.result()
 pool.shutdown()
 if writer:writer.flush()
 print('GEOMETRY_CAPTURE_DONE',n,flush=True)
if __name__=='__main__':app.run(lambda argv:run(pyconfig.initialize(argv)))

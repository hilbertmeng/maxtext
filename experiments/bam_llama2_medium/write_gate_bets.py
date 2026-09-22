"""Same-sequence single-head write-gate reduction, read-gate conditional."""
import contextlib,json,os,time,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import write_geometry as w
from write_geometry import jax,jnp,nn,partitioning,train,max_utils,attentions

@contextlib.contextmanager
def intervene(target_layer,target_head,threshold,shift):
 stack=[]
 def hook(next_fun,args,kw,ctx):
  m=ctx.module
  if not isinstance(m,attentions.BamAttention):return next_fun(*args,**kw)
  if ctx.method_name=='__call__':
   stack.append(kw['layer_index'])
   try:return next_fun(*args,**kw)
   finally:stack.pop()
  if ctx.method_name!='_write' or not stack:return next_fun(*args,**kw)
  o,x,M=args;cfg=m.config
  assert not cfg.bam_sqrt_n_scale and cfg.bam_lambda_decay==1.
  u=m._write_data(o,x);p=m._write_address(x)
  bias=jnp.asarray(m.gw_b0,m.dtype) if m._force_activation_dtype else m.gw_b0
  gate=jax.nn.sigmoid(m.W_gw(x)+bias)
  rg=m._read_gate_activation(m._project_read_gate_logits('W_R_gate',x,squeeze_fetch_axis=True))[...,0]
  selected=(stack[-1]==target_layer)&(jnp.arange(gate.shape[-1])==target_head)&(rg>=threshold)
  gf=gate.astype(jnp.float32)
  step=jax.lax.stop_gradient(jnp.minimum(.01,jnp.minimum(.1*gf,.1*(1-gf))))
  shifted=gf+shift*step
  # No shift!=0 branch: that would incorrectly erase the derivative at baseline.
  changed=jnp.where(selected,shifted.astype(gate.dtype),gate)
  un=m.write_data_norm(u) if m._write_data_rms else u;pn=m.write_address_norm(p)
  gu=changed[...,None]*un
  if m._write_outer_implementation=='dot':
   address_axes='nv' if m._write_address_mode=='static' else 'btnv';dm=jnp.einsum('btnk,'+address_axes+'->btkv',gu,pn)
  else:
   assert m._write_outer_implementation=='mul_reduce';dm=jnp.sum(gu[...,None]*pn[...,None,:],axis=-3)
  result=attentions._update_bam_matrix(M,dm,cfg.bam_lambda_decay)
  m.sow('intermediates','bet_stats',jnp.stack([selected.sum().astype(jnp.float32),jnp.where(selected,gate,0).sum().astype(jnp.float32),jnp.where(selected,gate-changed,0).sum().astype(jnp.float32),(selected&((shifted<0)|(shifted>1))).sum().astype(jnp.float32),jnp.where(selected,step,0).sum(),(selected&(gate==changed)).sum().astype(jnp.float32)]))
  return result,changed
 with nn.intercept_methods(hook):yield

def run(cfg):
 out=Path(os.environ['BET_OUTPUT']);out.mkdir(parents=True,exist_ok=True);protocol=json.loads(Path(os.environ['BET_PROTOCOL']).read_text());cohortpath=Path('/tmp/pile_eval_cohort.npz')
 with np.load(cohortpath) as f:cohort={k:f[k] for k in w.KEYS}
 meta=dict(model=w.BASE,checkpoint=cfg.load_parameters_path,protocol=protocol,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cohort_sha256=hashlib.sha256(cohortpath.read_bytes()).hexdigest())
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(cfg);state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,cfg,rng,mesh,manager)
 def forward(p,b,l,h,t,f,enabled=True):
  with intervene(l,h,t,f) if enabled else contextlib.nullcontext():
   result,inter=model.apply(p,b['inputs'],b['inputs_position'],decoder_segment_ids=b['inputs_segmentation'],decoder_target_mask=b['targets_segmentation'],decoder_target_tokens=b['targets'],enable_dropout=False,rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
  mask=b['targets_segmentation']!=0;loss=(result[0]*mask).sum()/jnp.maximum(mask.sum(),1)
  if not enabled:return loss
  flat=w.flatten_dict(inter);stats=sum(jnp.asarray(v[0]).reshape(-1,6).sum(0) for k,v in flat.items() if k[-1]=='bet_stats')
  return loss,stats
 fn=jax.jit(forward);gradfn=jax.jit(jax.value_and_grad(forward,argnums=5,has_aux=True));native=jax.jit(lambda p,b:forward(p,b,-1,0,0.,0.,False))
 arms=[dict(id='baseline',layer=-1,head=0,threshold=0.,shift=0.)]
 for h in protocol['heads']:
  for shift in protocol['shifts']:arms.append({**h,'shift':shift,'id':h['id']+f'_shift{shift}'})
 # Terminal write must have no downstream effect; factor-one checks copied write path.
 arms += [dict(id='terminal_control',layer=23,head=0,threshold=0.,shift=-1.),dict(id='zero_shift_control',layer=12,head=0,threshold=0.,shift=0.)]
 (out/'arms.json').write_text(json.dumps(arms,indent=2));start=time.perf_counter()
 with mesh,partitioning.axis_rules(cfg.logical_axis_rules):
  for i in range(32,128):
   dest=out/f'seq_{i:03d}.json'
   if dest.exists():continue
   b={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()};ordinary=float(native(state.params,b));rows=[];gradients=[]
   for head in protocol['heads']:
    (gloss,gstats),derivative=jax.device_get(gradfn(state.params,b,jnp.asarray(head['layer']),jnp.asarray(head['head']),jnp.asarray(head['threshold']),jnp.asarray(0.)))
    assert abs(float(gloss)-ordinary)<1e-7,(head['id'],gloss,ordinary)
    gradients.append(dict(id=head['id'],derivative=float(derivative),selected=float(gstats[0]),step_sum=float(gstats[4])))
   if i==32:print('FIRST_STEP GRADIENTS_DONE',len(gradients),'elapsed',time.perf_counter()-start,flush=True)
   # Rotate execution order after baseline to prevent dose/time confounding.
   order=[0]+list(np.random.default_rng(i).permutation(np.arange(1,len(arms))))
   for j in order:
    a=arms[j];loss,stats=jax.device_get(fn(state.params,b,jnp.asarray(a['layer']),jnp.asarray(a['head']),jnp.asarray(a['threshold']),jnp.asarray(a['shift'])))
    rows.append(dict(arm=j,id=a['id'],loss=float(loss),delta=float(loss)-ordinary,selected=float(stats[0]),gate_before=float(stats[1]),gate_removed=float(stats[2]),clipped=float(stats[3]),step_sum=float(stats[4]),rounded_unchanged=float(stats[5])))
   rows.sort(key=lambda x:x['arm']);assert abs(rows[0]['delta'])<1e-7,rows[0]
   assert abs(rows[-1]['delta'])<1e-7 and abs(rows[-2]['delta'])<1e-7,rows[-2:]
   dest.write_text(json.dumps(dict(sequence=i,native_loss=ordinary,gradients=gradients,rows=rows),indent=2))
   print('BET_SEQUENCE_DONE',i,'arms',len(arms),'elapsed',time.perf_counter()-start,flush=True)
 if writer:writer.flush()
 (out/'DONE').write_text('96 paired sequences complete\n');print('BETS_DONE',flush=True)
if __name__=='__main__':w.app.run(lambda argv:run(w.pyconfig.initialize(argv)))

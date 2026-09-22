"""Isolate LocalO -> M: frozen current-write denominator, unchanged residual.

delta=alpha-1; only add delta * g * o/RMS(original write data) outer p.
Per-position gradients allow arbitrary baseline gate partitions without reruns.
"""
import contextlib,hashlib,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import write_geometry as w
from write_geometry import jax,jnp,nn,partitioning,train,max_utils,attentions

@contextlib.contextmanager
def intervene(delta=None,layer=-1,head=0,state=0,shift=0.,threshold=.1):
    stack=[]
    def hook(next_fun,args,kw,ctx):
        m=ctx.module;name=ctx.method_name
        if not isinstance(m,attentions.BamAttention):return next_fun(*args,**kw)
        if name=='__call__':
            stack.append(dict(layer=kw['layer_index']))
            try:return next_fun(*args,**kw)
            finally:stack.pop()
        if not stack:return next_fun(*args,**kw)
        d=stack[-1]
        if name=='_independent_local_vo':
            out=next_fun(*args,**kw);d['localo']=out[1];return out
        if name!='_write':return next_fun(*args,**kw)
        out,gate=next_fun(*args,**kw)
        ohead,x,_=args
        assert m._write_factor_norm=='rms' and not m._concat_write_mix
        assert not m.config.bam_sqrt_n_scale and m.config.bam_lambda_decay==1.
        assert m._write_data_rms and 'localo' in d
        u=m._write_data(ohead,x).astype(jnp.float32)
        # Denominator is computed before modifying the write. Do not normalize
        # the modified sum: that would also change the other three components.
        denom=jnp.sqrt(jnp.mean(u*u,axis=-1,keepdims=True)+m._rms_epsilon)
        localo=d['localo'][...,:m.bam_k].astype(jnp.float32)
        piece=gate.astype(jnp.float32)[...,None]*localo/denom
        p=m.write_address_norm(m._write_address(x)).astype(jnp.float32)
        rg=m._read_gate_activation(m._project_read_gate_logits('W_R_gate',x,squeeze_fetch_axis=True))[...,0].astype(jnp.float32)
        wg=gate.astype(jnp.float32)
        if delta is not None:
            change=delta[jnp.asarray(d['layer'],jnp.int32)]
        else:
            rh=rg>=threshold;wh=wg>=threshold
            category=jnp.where(rh,jnp.where(wh,3,0),jnp.where(wh,1,2))
            take=(d['layer']==layer)&(jnp.arange(wg.shape[-1])==head)&(category==state)
            change=jnp.where(take,shift,0.)
        correction=jnp.einsum('btnk,btnv->btkv',change[...,None]*piece,p,precision=jax.lax.Precision.HIGHEST)
        m.sow('intermediates','localo_write_context',jnp.stack([wg,rg,jnp.broadcast_to(jnp.asarray(d['layer'],jnp.float32),wg.shape)],-1))
        m.sow('intermediates','localo_write_piece',piece)
        m.sow('intermediates','localo_write_address',p)
        return out+correction.astype(out.dtype),gate
    with nn.intercept_methods(hook):yield

def run(cfg):
    out=Path(os.environ['BET_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    path=Path('/tmp/pile_eval_cohort.npz')
    with np.load(path) as f:cohort={k:f[k] for k in w.KEYS}
    start=int(os.environ.get('BET_START','32'));stop=int(os.environ.get('BET_STOP','128'));mode=os.environ.get('LOCALO_MODE','gradient')
    assert 32<=start<stop<=128 and str(cfg.dtype)=='float32'
    meta=dict(model=w.BASE,checkpoint=cfg.load_parameters_path,dtype=str(cfg.dtype),matmul_precision=str(cfg.matmul_precision),runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cohort_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=mode,shard=[start,stop],definition='d per-sequence mean loss / d alpha_localO_write at alpha=1; original write denominator and residual path unchanged',threshold=.1,finite_heads=[[5,11],[8,7],[12,4],[15,3]],finite_shifts=[-.1,.1])
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    rng,writer,manager,mesh,model,_,tx=train.setup_mesh_and_model(cfg)
    state,_,_,_=max_utils.setup_training_state(model,SimpleNamespace(meta_dict={'checkpoint_step':None}),tx,cfg,rng,mesh,manager)
    zeros=jnp.zeros((24,1,cohort['inputs'].shape[-1],16),jnp.float32)
    def forward(params,b,delta=None,layer=-1,head=0,category=0,shift=0.,enabled=True):
        with intervene(delta,layer,head,category,shift) if enabled else contextlib.nullcontext():
            result,inter=model.apply(params,b['inputs'],b['inputs_position'],decoder_segment_ids=b['inputs_segmentation'],decoder_target_mask=b['targets_segmentation'],decoder_target_tokens=b['targets'],enable_dropout=False,rngs={'params':rng,'dropout':rng},mutable=['intermediates'])
        valid=b['targets_segmentation']!=0;tokens=jnp.where(valid,result[0],0.)
        loss=tokens.sum()/jnp.maximum(valid.sum(),1)
        if delta is not None:
            ctx=jnp.concatenate([jnp.asarray(v[0]).reshape(-1,cohort['inputs'].shape[-1],16,3) for k,v in w.flatten_dict(inter).items() if k[-1]=='localo_write_context'],axis=0)
            return loss,(tokens,ctx)
        return tokens
    native=jax.jit(lambda p,b:forward(p,b,enabled=False))
    gradfn=jax.jit(jax.value_and_grad(lambda p,b,d:forward(p,b,d),argnums=2,has_aux=True))
    finite=jax.jit(lambda p,b,l,h,c,s:forward(p,b,layer=l,head=h,category=c,shift=s))
    begin=time.perf_counter()
    with mesh,partitioning.axis_rules(cfg.logical_axis_rules):
        for i in range(start,stop):
            dest=out/f'{mode}_{i:03d}.npz'
            if dest.exists():continue
            b={k:jnp.asarray(v[i:i+1]) for k,v in cohort.items()};valid=cohort['targets_segmentation'][i]!=0;count=int(valid.sum())
            nt=np.asarray(native(state.params,b))
            if mode=='gradient':
                (_, (tokens,ctx)),grad=jax.device_get(gradfn(state.params,b,zeros));grad=grad[:,0]
                ls=ctx[:,0,0,2].astype(int);np.testing.assert_array_equal(np.sort(ls),np.arange(24));ctx=ctx[np.argsort(ls)]
                drift=float((tokens.astype(np.float64)-nt).sum()/count)
                assert abs(drift)<1e-7 and np.isfinite(grad).all(),drift
                assert np.count_nonzero(grad[23])==0
                payload=dict(gradient=grad,write_gate=ctx[...,0],read_gate=ctx[...,1],valid=valid,input_valid=cohort['inputs_segmentation'][i]!=0,positions=cohort['inputs_position'][i],native=nt,primal_delta=drift)
            else:
                assert mode=='finite';rows=[];ids=[]
                arms=[(-1,0,0,0.),(23,0,0,-.1)]+[(l,h,c,s) for l,h in meta['finite_heads'] for c in range(4) for s in meta['finite_shifts']]
                for l,h,c,s in arms:
                    tok=np.asarray(finite(state.params,b,jnp.asarray(l),jnp.asarray(h),jnp.asarray(c),jnp.asarray(s)))
                    rows.append(tok);ids.append([l,h,c,s])
                rows=np.stack(rows);deltas=(rows.astype(np.float64)-nt).sum(axis=(-1,-2))/count
                assert np.max(abs(deltas[:2]))<1e-7,deltas[:2]
                payload=dict(native=nt,arms=rows,ids=np.array(ids),deltas=deltas,valid=valid)
            temp=dest.with_suffix('.npz.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**payload)
            temp.replace(dest)
            print('LOCALO_SEQUENCE_DONE',mode,i,'elapsed',time.perf_counter()-begin,flush=True)
    if writer:writer.flush()
    (out/'DONE').write_text(f'{mode} complete [{start},{stop})\n')

if __name__=='__main__':w.app.run(lambda argv:run(w.pyconfig.initialize(argv)))

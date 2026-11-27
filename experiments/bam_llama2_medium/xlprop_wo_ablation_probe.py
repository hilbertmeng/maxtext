"""Normal-step capture-path diagnosis only; never an AV/O attribution report."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import numpy as np


def relative(x,y):
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))


def save_compiled_hlo(compiled, output):
    """Retain optimized compiler text, explicitly label unsupported runtimes."""
    try:
        raw=compiled.as_text().encode()
    except Exception as exc:
        return dict(available=False,error=type(exc).__name__+': '+str(exc))
    meta=dict(available=True,stage='compiled.as_text',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),saved=False)
    if len(raw)<=64*(1<<20):
        with gzip.open(output,'wb') as handle:handle.write(raw)
        meta['saved']=True
    return meta


def run_probe(train,captured,reference,output,sequence,target_layer):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from layers import attentions,bam_pallas
    from wo_capture_ablation import make_ablation_tap
    from wo_native_momentum import find_wo_adam_state,recover_unclipped
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg,mesh,state=captured['config'],captured['mesh'],captured['state']
    assert captured['step']==35635,'Normal unclipped AOT reference only'
    assert jax.process_count()==4 and jax.device_count()==16
    assert cfg.global_batch_size_to_train_on==128 and cfg.max_target_length==4096
    native=captured['observed']
    assert abs(native['learning/loss']-2.0112438201904297)<1e-6
    assert abs(native['learning/raw_grad_norm']/.21486762166023254-1)<1e-6
    assert cfg.opt_type=='adam_pax' and native['learning/raw_grad_norm']<cfg.gradient_clipping_threshold
    model,_,shardings=captured['functional_train'].args
    batch,rng=captured['data'],captured['rng']
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
    rows=jax.jit(lambda x:jnp.broadcast_to(jnp.arange(x.shape[0],dtype=jnp.int32)[:,None],x.shape),in_shardings=ds,out_shardings=ds)(batch['targets_segmentation'])
    wo=jax.jit(lambda t:t['params']['decoder']['layers']['block']['self_attention']['out']['kernel'][:,:6],out_shardings=rep)
    norm=lambda t:jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2) for x in jax.tree.leaves(t)))
    norm_fn=jax.jit(norm,out_shardings=rep)
    difference=jax.jit(lambda a,b:norm(jax.tree.map(lambda x,y:x.astype(jnp.float32)-y.astype(jnp.float32),a,b))/norm(b),out_shardings=rep)
    def loss_fn(params,batch,rng):
        return train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)[0]
    plain=jax.jit(jax.value_and_grad(loss_fn),in_shardings=(shardings.params,ds,None),out_shardings=(rep,shardings.params))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,base=plain(state.params,batch,rng)
    loss=float(jax.device_get(loss));raw=float(jax.device_get(norm_fn(base)))
    assert abs(loss-native['learning/loss'])<1e-3 and abs(raw/native['learning/raw_grad_norm']-1)<.01
    control=np.asarray(jax.device_get(wo(base)));np.save(output/'plain-wo.npy',control)
    before=find_wo_adam_state(state.opt_state);after=find_wo_adam_state(captured['native_result'].opt_state)
    recover=jax.jit(lambda b,a:recover_unclipped(b,a,cfg.adam_b1,cfg.adam_b2),out_shardings=(rep,rep,rep,rep))
    ng,nu,pnu,inc=jax.device_get(recover(before,after));assert int(inc)==1
    assert relative(pnu,nu)<1e-5
    native_errors=[relative(control[:,i],ng[:,i]) for i in range(6)]
    assert all(e<.01 for e in native_errors),'Baseline must match native before ablations'
    np.save(output/'native-aot-wo.npy',ng)
    report=dict(step=35635,complete=False,scientific_review_complete=False,
        purpose='Capture-path ablation only; no branch attribution',
        jax_version=jax.__version__,process_index=jax.process_index(),
        runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('wo_capture_ablation.py'),Path(__file__).with_name('wo_native_momentum.py'))},
        native_moment_consistency=relative(pnu,nu),plain_vs_native=native_errors,
        input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest()) for s in batch['inputs'].addressable_shards],modes={})
    local_rows=sorted({r for s in batch['inputs'].addressable_shards for r in range(*s.index[0].indices(128))})
    native_read,native_splash=bam_pallas.read,attentions._bam_splash_attention
    try:
        for mode in ('identity','forward','cotangent'):
            events=[];lock=threading.Lock();stack=[];active_rows=None;active_valid=None
            def writer(*args):
                layer=int(np.asarray(args[0]));rr=np.asarray(args[-2]);valid=np.asarray(args[-1])
                assert rr.ndim==2 and valid.shape==rr.shape and 0<=layer<6
                assert np.array_equal(rr,np.broadcast_to(rr[:,0,None],rr.shape))
                event=dict(layer=layer,rows=rr[:,0].tolist(),dtypes=[str(x.dtype) for x in args[1:-2]],shapes=[list(x.shape) for x in args[1:-2]])
                with lock:events.append(event)
            tap=make_ablation_tap(writer,mode,mesh,cfg.data_sharding[0],layer_limit=6)
            def read(*args,**kw):
                r=native_read(*args,**kw);stack[-1]['O']=r[3];return r
            def splash(*args,**kw):
                r=native_splash(*args,**kw);stack[-1]['A']=r;return r
            def intercept(next_fun,args,kw,ctx):
                m=ctx.module
                if isinstance(m,attentions.BamAttention) and ctx.method_name=='__call__':
                    layer=kw.get('layer_index');layer=jnp.int32(m.layer_inx) if layer is None else layer
                    stack.append(dict(layer=layer))
                    try:return next_fun(*args,**kw)
                    finally:stack.pop()
                if isinstance(m,attentions.BamAttention) and ctx.method_name=='out_projection':
                    y=next_fun(*args,**kw);v=stack[-1]
                    return tap(y,v['A'],v['O'],args[1],active_rows,active_valid,v['layer'])
                return next_fun(*args,**kw)
            def measured(params,batch,rng,rows):
                nonlocal active_rows,active_valid
                active_rows=rows;active_valid=batch['targets_segmentation']!=0
                with nn.intercept_methods(intercept):return loss_fn(params,batch,rng)
            fn=jax.jit(jax.value_and_grad(measured),in_shardings=(shardings.params,ds,None,ds),out_shardings=(rep,shardings.params))
            bam_pallas.read=read;attentions._bam_splash_attention=splash
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                compiled=fn.lower(state.params,batch,rng,rows).compile()
                hlo=save_compiled_hlo(compiled,output/(mode+'-optimized-hlo.txt.gz'))
                ml,grad=compiled(state.params,batch,rng,rows)
            jax.block_until_ready((ml,grad));jax.effects_barrier()
            actual=np.asarray(jax.device_get(wo(grad)));np.save(output/(mode+'-wo.npy'),actual)
            errors=[relative(actual[:,i],ng[:,i]) for i in range(6)]
            coverage={i:sorted({r for e in events if e['layer']==i for r in e['rows']}) for i in range(6)}
            if mode=='identity':assert not events
            else:assert all(v==local_rows for v in coverage.values()),('Incomplete callback coverage',coverage,local_rows)
            report['modes'][mode]=dict(hlo=hlo,loss=float(jax.device_get(ml)),full_vector_relative_error=float(jax.device_get(difference(grad,base))),W_O_vs_native=errors,leaf_gate_pass=all(e<.01 for e in errors),events=events,coverage=coverage)
            # Persist each completed mode even if a subsequent compilation fails.
            (output/'capture-ablation.json').write_text(json.dumps(report,indent=2)+'\n')
            bam_pallas.read=native_read;attentions._bam_splash_attention=native_splash
            fn.clear_cache();del grad,compiled
        report['complete']=True
        (output/'capture-ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        bam_pallas.read=native_read;attentions._bam_splash_attention=native_splash
        jax.effects_barrier();plain.clear_cache()

"""Native-gated frozen W_O boundary capture; global branch review is separate.

Called by the existing bounded replay at normal35635 and firstspike35636.
A complete capture report is NOT a globally reviewed AV/O decomposition.
"""
import hashlib
import os
import subprocess
import json
import shutil
import sys
from pathlib import Path
import numpy as np


def run_probe(train,captured,reference,output,sequence,target_layer):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from layers import attentions,bam_pallas
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from wo_boundary_capture import BoundaryWriter,identity_boundary_tap,load_boundary
    from wo_boundary_decomposition import partition_sums,merge_sums

    cfg,mesh,state=captured['config'],captured['mesh'],captured['state']
    model,_,shardings=captured['functional_train'].args
    step=captured['step'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assert step in (35635,35636) and jax.process_count()==4 and jax.device_count()==16
    assert cfg.global_batch_size_to_train_on==128 and cfg.max_target_length==4096
    expected={35635:(2.0112438201904297,.21486762166023254),35636:(1.924464225769043,2.1978654861450195)}[step]
    native=captured['observed']
    assert abs(native['learning/loss']-expected[0])<1e-6
    assert abs(native['learning/raw_grad_norm']/expected[1]-1)<1e-6
    assert shutil.disk_usage(output).free>=32*(1<<30),'Need32GiB free for bounded raw capture and partials'
    rawdir=output/'boundaries';assert not rawdir.exists(),'Never mix capture generations'
    writer=BoundaryWriter(rawdir,step);tap=identity_boundary_tap(writer,layer_limit=6,mesh=mesh,batch_axes=cfg.data_sharding[0])
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
    batch=captured['data'];rng=captured['rng']
    # Derive row IDs inside JIT from an explicit, batch-sharded dynamic operand.
    row_fn=jax.jit(lambda x:jnp.broadcast_to(jnp.arange(x.shape[0],dtype=jnp.int32)[:,None],x.shape),in_shardings=ds,out_shardings=ds)
    row_ids=row_fn(batch['targets_segmentation'])
    stack=[];active_rows=None;active_valid=None
    native_read=bam_pallas.read;native_splash=attentions._bam_splash_attention
    def read(*args,**kw):
        result=native_read(*args,**kw);stack[-1]['O']=result[3];return result
    def splash(*args,**kw):
        result=native_splash(*args,**kw);stack[-1]['A']=result;return result
    def intercept(next_fun,args,kw,ctx):
        module=ctx.module
        if isinstance(module,attentions.BamAttention) and ctx.method_name=='__call__':
            layer=kw.get('layer_index');layer=jnp.int32(module.layer_inx) if layer is None else layer
            stack.append(dict(layer=layer))
            try:return next_fun(*args,**kw)
            finally:stack.pop()
        if isinstance(module,attentions.BamAttention) and ctx.method_name=='out_projection':
            y=next_fun(*args,**kw);v=stack[-1];h=args[1]
            assert h.shape==v['A'].shape==v['O'].shape
            return tap(y,v['A'],v['O'],h,active_rows,active_valid,v['layer'])
        return next_fun(*args,**kw)
    def plain(params,batch,rng):
        return train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)[0]
    def measured(params,batch,rng,rows):
        nonlocal active_rows,active_valid
        active_rows=rows;active_valid=batch['targets_segmentation']!=0
        with nn.intercept_methods(intercept):return plain(params,batch,rng)
    plain_fn=jax.jit(jax.value_and_grad(plain),in_shardings=(shardings.params,ds,None),out_shardings=(rep,shardings.params))
    capture_fn=jax.jit(jax.value_and_grad(measured),in_shardings=(shardings.params,ds,None,ds),out_shardings=(rep,shardings.params))
    def norm(tree):return jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2)for x in jax.tree.leaves(tree)))
    norm_fn=jax.jit(norm,out_shardings=rep)
    difference=jax.jit(lambda a,b:norm(jax.tree.map(lambda x,y:x.astype(jnp.float32)-y.astype(jnp.float32),a,b))/norm(b),out_shardings=rep)
    wo_fn=jax.jit(lambda tree:tree['params']['decoder']['layers']['block']['self_attention']['out']['kernel'][:,:6],out_shardings=rep)
    try:
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):loss,base=plain_fn(state.params,batch,rng)
        loss=float(jax.device_get(loss));raw=float(jax.device_get(norm_fn(base)))
        assert abs(loss-expected[0])<1e-3 and abs(raw/expected[1]-1)<.01,('plain/native',loss,raw)
        bam_pallas.read=read;attentions._bam_splash_attention=splash
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):measured_loss,grad=capture_fn(state.params,batch,rng,row_ids)
        jax.block_until_ready((measured_loss,grad));jax.effects_barrier()
        assert not writer.errors,writer.errors
        measured_loss=float(jax.device_get(measured_loss));vector_error=float(jax.device_get(difference(grad,base)))
        assert abs(measured_loss-loss)<1e-6 and vector_error<.01,('capture/plain',measured_loss,loss,vector_error)
        actual=np.asarray(jax.device_get(wo_fn(grad)),np.float32)
        control=np.asarray(jax.device_get(wo_fn(base)),np.float32)
        np.save(output/'actual-wo.npy',actual,allow_pickle=False)
        np.save(output/'plain-wo.npy',control,allow_pickle=False)
        def relative(x,y):return float(np.linalg.norm(x.astype(np.float64)-y)/max(np.linalg.norm(y.astype(np.float64)),1e-30))
        errors=[relative(actual[:,i],control[:,i])for i in range(6)]
        diagnostic=dict(step=step,complete=False,scientific_review_complete=False,W_O_leaf_relative_errors=errors)
        if step==35635 and 'native_result' in captured:
            from wo_native_momentum import find_wo_adam_state,recover_unclipped
            assert cfg.opt_type=='adam_pax' and native['learning/raw_grad_norm']<cfg.gradient_clipping_threshold
            before=find_wo_adam_state(state.opt_state);after=find_wo_adam_state(captured['native_result'].opt_state)
            recover=jax.jit(lambda b,a:recover_unclipped(b,a,cfg.adam_b1,cfg.adam_b2),out_shardings=(rep,rep,rep,rep))
            ng,nu,pnu,inc=jax.device_get(recover(before,after));assert int(inc)==1
            nu_error=relative(np.asarray(pnu),np.asarray(nu));assert nu_error<1e-5,('Native moment consistency',nu_error)
            ng=np.asarray(ng,np.float32);np.save(output/'native-aot-wo.npy',ng,allow_pickle=False)
            diagnostic.update(native_moment_consistency=nu_error,plain_vs_native_W_O=[relative(control[:,i],ng[:,i])for i in range(6)],capture_vs_native_W_O=[relative(actual[:,i],ng[:,i])for i in range(6)])
        (output/'fidelity-diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
        assert all(x<.01 for x in errors),('W_O capture/plain',errors)
        local_rows=sorted({row for shard in batch['inputs'].addressable_shards for row in range(*shard.index[0].indices(128))})
        coverage={i:[]for i in range(6)};records=[]
        paths=sorted(rawdir.glob('layer-*'))
        for path in paths:
            meta,arrays=load_boundary(path);assert meta['step']==step and meta['layer'] in coverage
            coverage[meta['layer']].extend(arrays['global_rows'].tolist());records.append(dict(partition=path.name,metadata=meta))
        for i,rows in coverage.items():assert sorted(rows)==local_rows and len(rows)==len(set(rows)),('capture coverage',i,rows,local_rows)
        (output/'boundary-manifests.json').write_text(json.dumps(records,indent=2)+'\n')
        # Release device gradient buffers before host-side matrix work.
        del grad,base
        for layer in range(6):
            selected=[p for p in paths if p.name.startswith(f'layer-{layer:02d}-')]
            partials=[]
            for path in selected:
                _,v=load_boundary(path)
                partials.append(partition_sums(v['AV'],v['localO'],v['H'],v['G'],v['global_rows'],v['valid']))
            sums=merge_sums(partials,local_rows)
            target=output/f'layer-{layer:02d}-partial.npz';temp=output/f'.layer-{layer:02d}-partial.tmp'
            with temp.open('wb') as f:np.savez(f,**sums)
            temp.replace(target)
            # Raw capture is redundant after verified partials+hash manifests.
            with np.load(target,allow_pickle=False) as saved:
                assert np.array_equal(saved['global_rows'],sums['global_rows']) and np.array_equal(saved['parts'],sums['parts'])
            del sums,partials,v
            for path in selected:shutil.rmtree(path)
        report=dict(step=step,native_control=native,plain_loss=loss,plain_raw=raw,instrumented_loss=measured_loss,instrumented_vector_relative_error=vector_error,
                    W_O_leaf_relative_errors=errors,local_rows=local_rows,capture_complete=True,complete=True,scientific_review_complete=False,
                    input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest())for s in batch['inputs'].addressable_shards],
                    scope='Frozen actual sharedG boundaries, local partial matrices; requires global4worker merge, nativeW_O/branchclosure/forward/input16 review before interpretation.')
        (output/'wo-split.json').write_text(json.dumps(report,indent=2)+'\n')
        if jax.process_index()==0:
            subprocess.run(['gsutil','-q','cp',str(output/'wo-split.json'),os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'].rstrip('/')+f'/step{step}/wo-split.json'],check=True,timeout=120)
        print('WO_BOUNDARY_CAPTURE_COMPLETE',step,flush=True)
    finally:
        bam_pallas.read=native_read;attentions._bam_splash_attention=native_splash
        jax.effects_barrier();plain_fn.clear_cache();capture_fn.clear_cache()

"""Native-gated G-only backward plus separate forward boundary capture; global branch review is separate.

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
    from wo_split_capture import SplitWriter,paired_snapshots,save_failed_hg_diagnostic
    from wo_capture_ablation import make_ablation_tap
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
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(*cfg.data_sharding))
    batch=captured['data'];rng=captured['rng']
    def batch_fingerprint():
        entries=[(name,str(shard.index),hashlib.sha256(np.asarray(shard.data).tobytes()).hexdigest())
                 for name,value in sorted(batch.items()) for shard in value.addressable_shards]
        return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    context=dict(run=str(output.parent),state=f'd54:AOT-original-trajectory:CP35599:preupdate{step}',
                 input=batch_fingerprint(),rng=hashlib.sha256(np.asarray(jax.device_get(jax.random.key_data(rng))).tobytes()).hexdigest())
    context['primal_wire_layout']='producer_bnth_bnht'
    writer=SplitWriter(rawdir,step,context,allow_identical_forward_repeats=True)
    gtap=make_ablation_tap(writer.backward,'cotangent',mesh,cfg.data_sharding[0],layer_limit=6)
    # Avoid a device conditional around forward host callbacks. The writer
    # discards unselected layers on the host; transfer overhead is bounded.
    ftap=make_ablation_tap(writer.forward,'forward',mesh,cfg.data_sharding[0],layer_limit=None,primal_layout='producer')
    # Derive row IDs inside JIT from an explicit, batch-sharded dynamic operand.
    row_fn=jax.jit(lambda x:jnp.broadcast_to(jnp.arange(x.shape[0],dtype=jnp.int32)[:,None],x.shape),in_shardings=ds,out_shardings=ds)
    row_ids=row_fn(batch['targets_segmentation'])
    stack=[];active_rows=None;active_valid=None;active_tap=None
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
            return active_tap(y,v['A'],v['O'],h,active_rows,active_valid,v['layer'])
        return next_fun(*args,**kw)
    def plain(params,batch,rng):
        return train.loss_fn(model,cfg,dict(batch),rng,params,is_train=True)[0]
    def measured(params,batch,rng,rows,phase):
        nonlocal active_rows,active_valid,active_tap
        active_tap=gtap if phase=='gradient' else ftap
        active_rows=rows;active_valid=batch['targets_segmentation']!=0
        with nn.intercept_methods(intercept):return plain(params,batch,rng)
    plain_fn=jax.jit(jax.value_and_grad(plain),in_shardings=(shardings.params,ds,None),out_shardings=(rep,shardings.params))
    capture_fn=jax.jit(jax.value_and_grad(lambda p,b,r,rows:measured(p,b,r,rows,'gradient')),in_shardings=(shardings.params,ds,None,ds),out_shardings=(rep,shardings.params))
    forward_fn=jax.jit(jax.value_and_grad(lambda p,b,r,rows:measured(p,b,r,rows,'forward')),in_shardings=(shardings.params,ds,None,ds),out_shardings=(rep,shardings.params))
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
        if step==35635 and 'native_result' in captured:
            assert all(x<.01 for x in diagnostic['plain_vs_native_W_O'])
            assert all(x<.01 for x in diagnostic['capture_vs_native_W_O'])
        # Separate full-gradient invocation: forward callback, no custom-VJP primal residual.
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):forward_loss,forward_grad=forward_fn(state.params,batch,rng,row_ids)
        jax.block_until_ready((forward_loss,forward_grad));jax.effects_barrier()
        forward_loss=float(jax.device_get(forward_loss))
        forward_vector_error=float(jax.device_get(difference(forward_grad,base)))
        forward_wo=np.asarray(jax.device_get(wo_fn(forward_grad)),np.float32)
        np.save(output/'forward-wo.npy',forward_wo,allow_pickle=False)
        forward_errors=[relative(forward_wo[:,i],control[:,i]) for i in range(6)]
        forward_ok=abs(forward_loss-loss)<1e-6 and forward_vector_error<.01 and all(x<.01 for x in forward_errors)
        forward_diagnostic=dict(step=step,diagnostic_only=True,complete=False,scientific_review_complete=False,
                                snapshot_context=context,plain_loss=loss,forward_loss=forward_loss,
                                input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest()) for s in batch['inputs'].addressable_shards],
                                forward_vector_relative_error=forward_vector_error,
                                forward_W_O_leaf_relative_errors=forward_errors,fidelity_passed=forward_ok)
        (output/'forward-fidelity-diagnostic.json').write_text(json.dumps(forward_diagnostic,indent=2)+'\n')
        assert not writer.errors,writer.errors
        assert batch_fingerprint()==context['input'],'Batch changed between executions'
        local_rows=sorted({row for shard in batch['inputs'].addressable_shards for row in range(*shard.index[0].indices(128))})
        pairs=paired_snapshots(rawdir,step,context,range(6),local_rows)
        records=[]
        for meta,arrays in pairs:
            key=f"layer-{meta['layer']:02d}-rows-"+hashlib.sha256(arrays['global_rows'].tobytes()).hexdigest()[:16]
            gm=json.loads((rawdir/'cotangent'/key/'complete.json').read_text())
            records.append(dict(partition=key,metadata=meta,cotangent_metadata=gm))
        (output/'boundary-manifests.json').write_text(json.dumps(records,indent=2)+'\n')
        if not forward_ok:
            # Preserve a bounded H-only reconstruction with the faithful G-only
            # cotangent before rejecting the run; never produce scientific partials.
            del grad,forward_grad,base
            save_failed_hg_diagnostic(output,pairs,range(6),local_rows)
            raise AssertionError(('Snapshot fidelity/plain',forward_loss,loss,forward_vector_error,forward_errors))
        # Release full device gradients before local FP64 contractions.
        del grad,forward_grad,base,arrays
        for layer in range(6):
            selected=[(meta,v) for meta,v in pairs if meta['layer']==layer]
            partials=[partition_sums(v['AV'],v['localO'],v['H'],v['G'],v['global_rows'],v['valid']) for _,v in selected]
            sums=merge_sums(partials,local_rows)
            target=output/f'layer-{layer:02d}-partial.npz';temp=output/f'.layer-{layer:02d}-partial.tmp'
            with temp.open('wb') as f:np.savez(f,**sums)
            temp.replace(target)
            with np.load(target,allow_pickle=False) as saved:
                assert np.array_equal(saved['global_rows'],sums['global_rows']) and np.array_equal(saved['parts'],sums['parts'])
            pairs=[item for item in pairs if item[0]['layer']!=layer]
            keys=[x['partition'] for x in records if x['metadata']['layer']==layer]
            del sums,partials,selected
            for key in keys:
                shutil.rmtree(rawdir/'primal'/key);shutil.rmtree(rawdir/'cotangent'/key)
        report=dict(step=step,capture_method='split_execution_v1',primal_execution='value_and_grad',primal_wire_layout='producer_bnth_bnht',forward_vector_relative_error=forward_vector_error,forward_W_O_leaf_relative_errors=forward_errors,identical_forward_repeats=writer.identical_forward_repeats,forward_snapshot_loss=forward_loss,snapshot_context=context,native_control=native,plain_loss=loss,plain_raw=raw,instrumented_loss=measured_loss,instrumented_vector_relative_error=vector_error,
                    W_O_leaf_relative_errors=errors,local_rows=local_rows,capture_complete=True,complete=True,scientific_review_complete=False,
                    input_shards=[dict(index=str(s.index),sha256=hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest())for s in batch['inputs'].addressable_shards],
                    scope='G-only validated gradient graph plus independently captured forward boundaries; requires global4worker merge, nativeW_O/branchclosure/forward/input16 review before interpretation.')
        (output/'wo-split.json').write_text(json.dumps(report,indent=2)+'\n')
        if jax.process_index()==0:
            subprocess.run(['gsutil','-q','cp',str(output/'wo-split.json'),os.environ['BAM_LOSS_LOCALIZATION_UPLOAD'].rstrip('/')+f'/step{step}/wo-split.json'],check=True,timeout=120)
        print('WO_BOUNDARY_CAPTURE_COMPLETE',step,flush=True)
    finally:
        bam_pallas.read=native_read;attentions._bam_splash_attention=native_splash
        jax.effects_barrier();plain_fn.clear_cache();capture_fn.clear_cache();forward_fn.clear_cache()

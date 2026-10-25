"""Native output identity, full/chunk QK scope and exact synthetic statistics."""
import os,tempfile
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils,precise_ce
from layers.models import Transformer
import value_route_capture as hook
def main():
    # Rank and attention-statistic controls independent of the model wrappers.
    x=jnp.eye(4)[None,None];np.testing.assert_allclose(hook.diversity(x)[...,:2],4,rtol=1e-6)
    x=jnp.ones((1,1,4,8));np.testing.assert_allclose(hook.diversity(x)[...,:2],1,rtol=1e-6)
    q=k=jnp.zeros((1,128,2,4));seg=jnp.ones((1,128),jnp.int32);s=hook.self_statistics(q,k,seg)
    expected=np.mean(1/(np.arange(128)+1));np.testing.assert_allclose(s[:,:,0,0],expected,rtol=1e-6)
    np.testing.assert_allclose(s[:,:,1,0],np.mean(1/(np.arange(64,128)+1)),rtol=1e-6)
    v=jnp.ones((1,128,2,4));dyv=.25*v;dyo=.5*v;sto=.3*v;o=dyo+sto
    e,r=hook.route_statistics(q,k,v,seg,v,o,dyv,v-dyv,dyo,sto)
    np.testing.assert_allclose(e[...,hook.ENERGY_NAMES.index('AV_reconstruction_error')],0,atol=1e-10)
    a=hook.ENERGY_NAMES.index('dynamicAV');b=hook.ENERGY_NAMES.index('dynamicAV_parallel_dynamicO');np.testing.assert_allclose(e[...,a],e[...,b],rtol=1e-6)
    print('SYNTHETIC_UNIFORM_ATTENTION_AND_ORTHOGONAL_IDENTICAL_HEAD_RANKS_PASS',flush=True)
    kind=os.environ.get('TEST_KIND','xl');pallas=kind=='pallas';py=kind!='xl';B=1;T=128;n=5 if pallas else (8 if py else 7)
    parent=exp.BamPallasPythiaR768TinyPallasV8TTest if pallas else exp.BamPallasPythiaR768TinyXlaTest if py else exp.BamPallasFullMTinyXlaTest
    attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype=os.environ.get('TEST_DTYPE','float32'),weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=B,per_device_batch_size=float(B),query_chunk_size=T//2)
    exp.ValueRouteTiny=type('ValueRouteTiny',(parent,),attrs);precise_ce.install()
    with tempfile.TemporaryDirectory() as out:
        os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='ValueRouteTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(B,T),0,128);pos=jnp.broadcast_to(jnp.arange(T),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
        params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        def native():return model.apply(params,tok,pos,rngs=rng,**kw)[0]
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=jax.jit(native)()
        hook.install();hook.ACTIVE=True
        def observed():return model.apply(params,tok,pos,rngs=rng,mutable=['intermediates'],**kw)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):result,raw=jax.jit(observed)()
        difference=np.asarray(result[0])-np.asarray(baseline);print('NEUTRAL_CE',kind,'maxmean',float(np.abs(difference.mean(1)).max()),'maxtoken',float(np.abs(difference).max()),flush=True)
        assert np.abs(difference.mean(1)).max()<(.005 if cfg.dtype==jnp.bfloat16 else 1e-5)
        energy=[];ranks=[];selfs=[]
        for path,values in flatten_dict(jax.device_get(raw)).items():
            if path[-1]=='stream_route_energy':energy.extend([np.asarray(x) for x in values])
            if path[-1]=='stream_route_rank':ranks.extend([np.asarray(x) for x in values])
            if path[-1]=='stream_route_self':selfs.extend([np.asarray(x) for x in values])
        assert len(energy) and len(ranks)==len(energy)==len(selfs)
        def error(num,den):
            return max(float(np.sqrt(x[...,hook.ENERGY_NAMES.index(num)].sum()/max(x[...,hook.ENERGY_NAMES.index(den)].sum(),1e-30))) for x in energy)
        for num,den in [('V_closure_error','V'),('O_closure_error','O'),('AV_reconstruction_error','MHA')]:
            err=error(num,den);print(num,err,flush=True);assert err<(.02 if cfg.dtype==jnp.bfloat16 else 1e-5)
        for x in energy+ranks+selfs:assert np.isfinite(x).all()
        assert all(x.shape[-3]==16 for x in energy) # selected queries retained, full sequence observed
        assert all(np.min(x)>=-1e-6 and np.max(x[...,:2])<=cfg.num_query_heads+1e-3 for x in ranks)
        print('NATIVE_IDENTITY_FULL_QUERY_SCOPE_VO_CLOSURES_AV_RECONSTRUCTION_FINITE_RANKS_PASS',kind,str(cfg.dtype),flush=True)
if __name__=='__main__':main()

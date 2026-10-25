"""Operator exposure, prefix causality, native identity and bias-free W_o moments."""
import os,tempfile,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,pyconfig,max_utils,precise_ce
from layers.models import Transformer
import attention_contrast_capture as hook
def main():
    q,k,v=jax.random.normal(jax.random.key(9),(3,2,128,2,8));seg=jnp.ones((2,128),jnp.int32);tokens=jnp.ones((2,128),jnp.int32)*42;tokens=tokens.at[:,0].set(0).at[:,30].set(15).at[:,90].set(187)
    refs,stats=hook.operators(q,k,v,seg,tokens)
    np.testing.assert_allclose(stats[:,1,0],0,atol=1e-7)
    np.testing.assert_allclose(stats[:,3,1],stats[:,0,1],atol=1e-6)
    np.testing.assert_allclose(stats[:,2,0],np.mean(1/(np.arange(64,128)+1)),rtol=1e-6)
    np.testing.assert_allclose(stats[:,0,2],0,atol=1e-8)
    for mode in range(4):
        coeff=jnp.eye(4)[mode];changed=hook.apply_operator(refs[:,0],refs,coeff)
        np.testing.assert_array_equal(changed[:,:64],refs[:,0,:64]);np.testing.assert_allclose(changed[:,64:],refs[:,mode,64:],atol=1e-6)
    packed=seg.at[:,64:].set(2);alt,_=hook.operators(q,k,v,packed,tokens)
    np.testing.assert_allclose(alt[:,1,64],alt[:,0,64],atol=1e-6)
    np.testing.assert_allclose(alt[:,1,65],v[:,64],atol=1e-6) # only previous key in the new segment
    common=jnp.ones_like(v);a,_=hook.operators(q,k,common,seg,tokens)
    np.testing.assert_allclose(a,1,atol=5e-6) # FP32 dot/softmax reduction rounding across128keys
    print('TRUE_SELF_EXCLUSION_UNIFORM_SINK_MASS_PACKED_FALLBACK_AND_CONSTANT_VALUE_PASS',flush=True)
    kind=os.environ.get('TEST_KIND','xl_bam');mudd=kind.endswith('mudd');py=kind.startswith('py') or kind=='pallas';pallas=kind=='pallas';B=1;T=128;n=4 if mudd else 5 if pallas else 8 if py else 7
    if mudd:
        parent=exp.MuddPythia3BSerialMLPShortTrain if py else exp.MuddLlama2XLProp
        attrs=dict(base_emb_dim=128,base_num_query_heads=4,base_num_kv_heads=4,head_dim=32,base_mlp_dim=256,base_num_decoder_layers=n,mudd_router_hidden_expand=([1]*(n-1)+[4] if py else [1]*(n-2)+[4]),attention='dot_product')
        if py:attrs['pythia_rotary_dim']=8
    else:
        parent=exp.BamPallasPythiaR768TinyPallasV8TTest if pallas else exp.BamPallasPythiaR768TinyXlaTest if py else exp.BamPallasFullMTinyXlaTest
        attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,query_chunk_size=T//2)
    attrs.update(vocab_size=128,dtype=os.environ.get('TEST_DTYPE','float32'),weight_dtype='float32',max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=B,per_device_batch_size=float(B))
    exp.ContrastTiny=type('ContrastTiny',(parent,),attrs);precise_ce.install()
    with tempfile.TemporaryDirectory() as outdir:
        os.mkdir(outdir+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='ContrastTiny',run_name='test',base_output_directory=outdir+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(B,T),0,128);pos=jnp.broadcast_to(jnp.arange(T),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
        params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        def native(weights):return model.apply(weights,tok,pos,rngs=rng,**kw)[0]
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=jax.jit(native)(params)
        hook.install()
        def observed(weights,ctrl):
            hook.CONTROLS=ctrl;hook.CAPTURE=True;hook.GEOMETRY=True;hook.TOKENS=tok
            try:return model.apply(weights,tok,pos,rngs=rng,mutable=['intermediates'],**kw)
            finally:hook.CONTROLS=None;hook.CAPTURE=False;hook.GEOMETRY=False;hook.TOKENS=None
        fn=jax.jit(observed);zero=jnp.zeros((n,4),jnp.float32)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):out,raw=fn(params,zero)
        error=np.asarray(out[0])-np.asarray(baseline);print('NEUTRAL_NATIVE_CE',kind,float(np.max(np.abs(error.mean(1)))),float(np.max(np.abs(error))),flush=True)
        assert np.max(np.abs(error.mean(1)))<(.005 if cfg.dtype==jnp.bfloat16 else 1e-5)
        def unpack(raw):
            flat={path:np.asarray(value[0]) for path,value in flatten_dict(jax.device_get(raw['intermediates'])).items() if path[-1].startswith('stream_contrast_')};layers={}
            for path,index in flat.items():
                if path[-1]!='stream_contrast_layer_index':continue
                for i,logical in enumerate(index.reshape(-1)):
                    logical=int(logical);assert logical not in layers
                    layers[logical]={p[-1].removeprefix('stream_'):(value if index.ndim==0 else value[i]) for p,value in flat.items() if p[:-1]==path[:-1]}
            assert set(layers)==set(range(n))
            return layers
        layers=unpack(raw);assert len(layers)==n
        for values in layers.values():
            g=values['contrast_projected_gram'];e=values['contrast_errors'];assert np.isfinite(g).all()
            assert np.sqrt(e[:,0].sum()/max(e[:,1].sum(),1e-30))<.02
            assert np.sqrt(e[:,2].sum()/max(g[:,-1,-1].sum(),1e-30))<.02
            assert np.sqrt(e[:,3].sum()/max(e[:,4].sum(),1e-30))<.02
        # Full-model causality and scope, including scanned blocks and Mudd history.
        for target,mode in [(0,1),(n-1,2),(n-1,3)]:
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):changed,state=fn(params,zero.at[target,mode].set(1))
            np.testing.assert_array_equal(np.asarray(changed[0])[:,:64],np.asarray(out[0])[:,:64])
            assert np.max(np.abs(np.asarray(changed[0])-np.asarray(out[0])))>1e-8
            selected=unpack(state)[target]['contrast_exposure'];before,after=selected[:,0],selected[:,1]
            if mode==1:np.testing.assert_allclose(after[:,0],0,atol=1e-7)
            if mode==3:np.testing.assert_allclose(before[:,1],after[:,1],atol=1e-6)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,_=fn(params,zero)
        np.testing.assert_array_equal(again[0],out[0])
        # Geometry projects with W_o only. A nonzero output bias cannot leak into components.
        flat=flatten_dict(params);changed={};matched=0
        for path,value in flat.items():
            text='/'.join(path)
            if text.endswith('/self_attention/out/kernel'):value=jnp.zeros_like(value);matched+=1
            if text.endswith('/self_attention/out/bias'):value=jnp.ones_like(value)*.001
            changed[path]=value
        assert matched>0
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):_,null=fn(unflatten_dict(changed),zero)
        for values in unpack(null).values():np.testing.assert_array_equal(values['contrast_projected_gram'],0)
        print('NATIVE_IDENTITY_ALL_LAYERS_WO_RECONSTRUCTION_PREFIX_SCOPE_EXPOSURE_REPEAT_AND_BIAS_PASS',kind,str(cfg.dtype),flush=True)
if __name__=='__main__':main()

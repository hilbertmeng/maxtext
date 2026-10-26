"""Native/observed parity, read decomposition and independent-key detection."""
import os,tempfile
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import vo_similarity_capture as hook
def main():
    x=jnp.array([1.,0.,0.]);y=jnp.array([0.,1.,0.])
    for a,b,want in [(x,x,1.),(x,y,0.),(x,-x,-1.)]:
        e=hook.moments(a,b);np.testing.assert_allclose(e[2]/jnp.sqrt(e[0]*e[1]),want,atol=1e-6)
    q=k=jnp.zeros((1,128,2,4));s=hook.attention_stats(q,k,jnp.ones((1,128),jnp.int32));np.testing.assert_allclose(s[...,0],np.broadcast_to(1/(np.arange(64,128)+1)[None,:,None],s[...,0].shape),rtol=2e-6);np.testing.assert_array_equal(s[...,1],0)
    kind=os.environ.get('TEST_KIND','separate');xl=kind=='xl';pallas=os.environ.get('TEST_PALLAS','0' if xl else '1')=='1';n=7 if xl else 5;T=128
    parent=exp.BamPallasFullMTinyXlaTest if xl else exp.BamPallasPythiaR768TinyXlaTest
    attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype=os.environ.get('TEST_DTYPE','bfloat16'),weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=1,per_device_batch_size=1.,query_chunk_size=T//2)
    if kind in ['separate','shared','memory']:
        attrs.update(base_mlp_dim=64,mlp_dim_by_block=[64]*3,bam_final_local_mlp_dim=64,bam_last_mlp_dim=48,bam_memory_mlp_dim=32,bam_memory_mlp_last_write_dim=0,bam_mlp_write_every=1,bam_mlp_write_offset=0,bam_skip_final_mlp_write=True)
    if kind=='separate':attrs.update(bam_local_vo_independent_keys=True,bam_pair_scan=False,bam_local_fetch_block_size=None,bam_final_local_layer_count=1,bam_final_local_mlp_dim=None,mlp_dim_by_block=None,bam_attention_write_layers=list(range(n-1)))
    exp.VOTiny=type('VOTiny',(parent,),attrs)
    with tempfile.TemporaryDirectory() as root:
        os.mkdir(root+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='VOTiny',run_name='test',base_output_directory=root+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(1,T),0,128);pos=jnp.arange(T)[None];seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
        params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        def native(p):return model.apply(p,tok,pos,rngs=rng,**kw)[0]
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):before=jax.jit(native)(params)
        hook.install()
        def observed(p):
            hook.ACTIVE=True
            try:return model.apply(p,tok,pos,rngs=rng,mutable=['intermediates'],**kw)
            finally:hook.ACTIVE=False
        fn=jax.jit(observed)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):after,raw=fn(params)
        diff=np.asarray(after[0])-np.asarray(before);assert np.isfinite(diff).all() and np.max(np.abs(diff.mean(1)))<(.005 if cfg.dtype==jnp.bfloat16 else 1e-5)
        def unpack(raw):
            flat={p:np.asarray(v[0]) for p,v in flatten_dict(jax.device_get(raw['intermediates'])).items() if p[-1].startswith('stream_vo_')};layers={}
            for path,index in flat.items():
                if path[-1]!='stream_vo_layer_index':continue
                for point in np.ndindex(index.shape):
                    i=int(index[point]);assert i not in layers;layers[i]={p[-1]:v[point] for p,v in flat.items() if p[:-1]==path[:-1]}
            assert set(layers)==set(range(n));return layers
        layers=unpack(raw)
        for layer,row in layers.items():
            m=row['stream_vo_moments'];e=row['stream_vo_errors'];a=row['stream_vo_attention']
            assert m.shape==(1,64,cfg.num_query_heads,5,3) and a.shape==(1,64,cfg.num_query_heads,2) and np.isfinite(m).all() and np.isfinite(a).all()
            assert a.min()>=0 and a.max()<=1
            for i in [0,2]:assert np.sqrt(e[:,i].sum()/max(e[:,i+1].sum(),1e-30))<(.02 if cfg.dtype==jnp.bfloat16 else 1e-5)
            for pair in [0,4]:
                z=m[...,pair,:];valid=(z[...,0]>1e-25)&(z[...,1]>1e-25)
                if valid.any() and kind!='separate':assert np.min(z[...,2][valid]/np.sqrt(z[...,0][valid]*z[...,1][valid]))>.995
        if kind=='separate':
            flat=flatten_dict(params);changed={};count=0
            for p,value in flat.items():
                if p[-1]=='general_v_pre_bias':value=value+.25;count+=1
                changed[p]=value
            assert count>0
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):_,other=fn(unflatten_dict(changed))
            changed=unpack(other);z=changed[0]['stream_vo_moments'][...,4,:];cos=z[...,2]/np.sqrt(np.maximum(z[...,0]*z[...,1],1e-30));assert np.mean(cos)<.99
            # Only V addresses changed at the first layer; the native O read
            # at that layer has the same input M and must remain identical.
            np.testing.assert_array_equal(changed[0]['stream_vo_moments'][...,1,1],layers[0]['stream_vo_moments'][...,1,1])
            assert not np.array_equal(changed[0]['stream_vo_moments'][...,1,0],layers[0]['stream_vo_moments'][...,1,0])
            for row in changed.values():
                error=row['stream_vo_errors']
                for i in [0,2]:assert np.sqrt(error[:,i].sum()/max(error[:,i+1].sum(),1e-30))<(.02 if cfg.dtype==jnp.bfloat16 else 1e-5)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):again,_=fn(params)
        np.testing.assert_array_equal(again[0],after[0])
        print('VO_NATIVE_IDENTITY_LAYER_INDEX_COMPONENT_CLOSURE_SHARED_COLLINEARITY_INDEPENDENT_KEY_AND_REPEAT_PASS',kind,pallas,str(cfg.dtype),flush=True)
if __name__=='__main__':main()

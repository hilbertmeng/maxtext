"""Native parity, all logical layers, masks and future-prefix independence."""
import os,tempfile
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import sentence_attention_capture as capture
def main():
    kind=os.environ.get('SENTENCE_TEST_KIND','separate');T=128;N=55;capture.PREFIX=N
    if kind=='separate':
        n=5;parent=exp.BamPallasPythiaR768TinyXlaTest
        attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype='bfloat16',weight_dtype='float32',bam_pallas_core=True,bam_pallas_interpret=True,bam_pallas_body='v8t',bam_pallas_fused_inputs=True,bam_splash_attention=False,bam_m_shift_mode='read',bam_m_shift_init=.5,base_mlp_dim=64,bam_last_mlp_dim=48,bam_memory_mlp_dim=32,bam_memory_mlp_last_write_dim=0,bam_mlp_write_every=1,bam_mlp_write_offset=0,bam_skip_final_mlp_write=True,bam_local_vo_independent_keys=True,bam_pair_scan=False,bam_local_fetch_block_size=None,bam_final_local_layer_count=1,bam_final_local_mlp_dim=None,mlp_dim_by_block=None,bam_attention_write_layers=list(range(n-1)))
    else:
        n=3;parent=exp.MuddPythia3BSerialMLPShortTrain;attrs=dict(base_emb_dim=128,base_num_query_heads=4,base_num_kv_heads=4,head_dim=32,base_mlp_dim=512,base_num_decoder_layers=n,pythia_rotary_dim=8,vocab_size=128,dtype='float32',weight_dtype='float32',attention='dot_product',mudd_router_hidden_expand=[1,1,4])
    attrs.update(max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=1,per_device_batch_size=1.0)
    exp.SentenceTiny=type('SentenceTiny',(parent,),attrs)
    with tempfile.TemporaryDirectory() as root:
        os.mkdir(root+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='SentenceTiny',run_name='test',base_output_directory=root+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False,record_training_health_metrics=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(42),(1,T),0,128);pos=jnp.arange(T)[None];seg=jnp.ones_like(tok);rngs={k:jax.random.key(i) for i,k in enumerate(['params','dropout','aqt'])};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rngs,tok,pos,**kw)
        params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        def native(t):return model.apply(params,t,pos,rngs=rngs,**kw)[0]
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):before=jax.jit(native)(tok)
        capture.install()
        def observe(t):
            capture.ACTIVE=True
            try:return model.apply(params,t,pos,rngs=rngs,mutable=['intermediates'],**kw)
            finally:capture.ACTIVE=False
        fn=jax.jit(observe)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):after,raw=fn(tok)
        assert np.max(np.abs(np.asarray(after[0])-np.asarray(before)).mean(1))<.005
        flat={k:np.asarray(v[0]) for k,v in flatten_dict(raw['intermediates']).items() if k[-1].startswith('sentence_')};layers={}
        for path,index in flat.items():
            if path[-1]!='sentence_layer_index':continue
            for point in np.ndindex(index.shape):
                i=int(index[point]);assert i not in layers;layers[i]={k[-1]:v[point] for k,v in flat.items() if k[:-1]==path[:-1]}
        assert set(layers)==set(range(n))
        for row in layers.values():
            a=row['sentence_alpha'];assert a.shape==(cfg.num_query_heads,N,N);assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
            np.testing.assert_allclose(a.sum(-1),1,atol=1e-6);assert np.triu(a,1).max()==0
            q=row['sentence_query'];k=row['sentence_key'];logits=np.einsum('qhd,khd->hqk',q,k);logits=np.where(np.tri(N,dtype=bool)[None],logits,-1e30);p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);np.testing.assert_allclose(a,p,atol=3e-6)
            e=row['sentence_error'];assert np.sqrt(e[:,0]/np.maximum(e[:,1],1e-30)).max()<.02
        changed=tok.at[:,N:].set((tok[:,N:]+1)%128)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):_,future=fn(changed);again,_=fn(tok)
        other={k:np.asarray(v[0]) for k,v in flatten_dict(future['intermediates']).items() if k[-1].startswith('sentence_')}
        for k,v in flat.items():np.testing.assert_array_equal(v,other[k])
        np.testing.assert_array_equal(after[0],again[0]);print('SENTENCE_NATIVE_PARITY_ALL_LAYERS_FULL_MAPS_AV_AND_FUTURE_INVARIANCE_PASS',kind,flush=True)
if __name__=='__main__':main()

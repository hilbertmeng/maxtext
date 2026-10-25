"""Exact diagonal-logit identity, neutral native model and changed scanned scopes."""
import os,tempfile
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
import exp,pyconfig,max_utils,precise_ce
from layers.models import Transformer
from value_route_capture import attention_map
import self_bias_capture as hook
def main():
    key=jax.random.key(9);q,k,v=jax.random.normal(key,(3,2,128,2,8));seg=jnp.ones((2,128),jnp.int32)
    p=hook.probabilities(q,k,seg);a=attention_map(q,k,seg,jnp.arange(128));out=jnp.einsum('bhqs,bshk->bqhk',a,v)
    np.testing.assert_array_equal(hook.adjust(out,v,p,0.),out)
    # Explicitly modify diagonal logits, retaining causal and segment masks.
    for bias in [-1.,-.25,.25,1.]:
        logits=jnp.einsum('bqhd,bshd->bhqs',q,k)
        logits=logits+bias*jnp.eye(128)[None,None]
        valid=jnp.arange(128)[None,:]<=jnp.arange(128)[:,None]
        biased=jax.nn.softmax(jnp.where(valid[None,None],logits,-1e30),axis=-1)
        expected=jnp.einsum('bhqs,bshk->bqhk',biased,v)
        np.testing.assert_allclose(hook.adjust(out,v,p,bias),expected,atol=2e-6,rtol=2e-5)
    # Common value across keys gives exact zero intervention for every bias.
    common=jnp.ones_like(v);np.testing.assert_array_equal(hook.adjust(common,common,p,-1.),common)
    query_mask=(jnp.arange(128)>=64)[None,:,None]
    scoped=hook.adjust(out,v,p,-query_mask.astype(jnp.float32))
    np.testing.assert_array_equal(scoped[:,:64],out[:,:64])
    logits=jnp.einsum('bqhd,bshd->bhqs',q,k)-jnp.eye(128)[None,None]*(jnp.arange(128)>=64)[None,None,:,None]
    valid=jnp.arange(128)[None,:]<=jnp.arange(128)[:,None]
    expected=jnp.einsum('bhqs,bshk->bqhk',jax.nn.softmax(jnp.where(valid[None,None],logits,-1e30),axis=-1),v)
    np.testing.assert_allclose(scoped,expected,atol=2e-6,rtol=2e-5)
    print('EXPLICIT_SOFTMAX_DIAGONAL_BIAS_ALGEBRA_ZERO_AND_CONSTANT_VALUE_CONTROLS_PASS',flush=True)
    kind=os.environ.get('TEST_KIND','xl');pallas=kind=='pallas';py=kind!='xl';B=1;T=128;n=5 if pallas else (8 if py else 7)
    parent=exp.BamPallasPythiaR768TinyPallasV8TTest if pallas else exp.BamPallasPythiaR768TinyXlaTest if py else exp.BamPallasFullMTinyXlaTest
    attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype=os.environ.get('TEST_DTYPE','float32'),weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=B,per_device_batch_size=float(B),query_chunk_size=T//2)
    exp.SelfBiasTiny=type('SelfBiasTiny',(parent,),attrs);precise_ce.install()
    with tempfile.TemporaryDirectory() as outdir:
        os.mkdir(outdir+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='SelfBiasTiny',run_name='test',base_output_directory=outdir+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(B,T),0,128);pos=jnp.broadcast_to(jnp.arange(T),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
        params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
        def native():return model.apply(params,tok,pos,rngs=rng,**kw)[0]
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=jax.jit(native)()
        hook.install()
        def probe(bias):
            hook.BIASES=bias
            try:return model.apply(params,tok,pos,rngs=rng,**kw)[0]
            finally:hook.BIASES=None
        fn=jax.jit(probe);zero=jnp.zeros(n,jnp.float32)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):neutral=fn(zero);first=fn(zero.at[0].set(-1));last=fn(zero.at[-1].set(-1));again=fn(zero)
        err=np.asarray(neutral)-np.asarray(baseline);print('NATIVE_ZERO_ERROR',kind,'maxmean',float(np.abs(err.mean(1)).max()),'maxtoken',float(np.abs(err).max()),flush=True)
        assert np.abs(err.mean(1)).max()<(.005 if cfg.dtype==jnp.bfloat16 else 1e-5)
        np.testing.assert_array_equal(again,neutral)
        assert np.max(np.abs(first-neutral))>1e-8 and np.max(np.abs(last-neutral))>1e-8
        assert np.max(np.abs(first-last))>1e-8
        print('NEUTRAL_NATIVE_IDENTITY_SCANNED_FIRST_LAST_SCOPE_AND_REPEAT_PASS',kind,str(cfg.dtype),flush=True)
if __name__=='__main__':main()

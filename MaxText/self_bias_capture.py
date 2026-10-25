"""Counterfactual diagonal attention bias using the unchanged native AV output.

For e=expm1(b), o_b=o+e*p_self/(1+e*p_self)*(v_self-o).
The identity is exact for real-valued softmax attention; b=0 preserves native
outputs. No parameters or optimizer state are altered. Native M writes respond.
"""
import functools
import jax
import jax.numpy as jnp
from value_route_capture import attention_map
BIASES=None
_INSTALLED=False
_STACK=[]
def probabilities(q,k,segments,window=None,soft_cap=0.):
    t=q.shape[1];chunk=min(64,t)
    if t!=k.shape[1] or t%chunk:raise ValueError('Self bias requires full aligned Q/K sequence')
    def block(_,i):
        ids=i*chunk+jnp.arange(chunk)
        a=attention_map(q,k,segments,ids,window,soft_cap)
        p=jnp.take_along_axis(a,jnp.broadcast_to(ids[None,None,:,None],a.shape[:-1]+(1,)),-1)[...,0]
        return None,p.transpose(0,2,1)
    _,p=jax.lax.scan(block,None,jnp.arange(t//chunk))
    return p.transpose(1,0,2,3).reshape(q.shape[:3])
def adjust(out,v,p,bias):
    a=jnp.expm1(jnp.asarray(bias,jnp.float32))*p.astype(jnp.float32)
    factor=a/(1+a)
    return (out.astype(jnp.float32)+factor[...,None]*(v.astype(jnp.float32)-out.astype(jnp.float32))).astype(out.dtype)
def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,fusion
    native_block=fusion.SubDecoderLayer.__call__
    @functools.wraps(native_block)
    def layer(self,*args,**kwargs):
        if BIASES is None or self.is_initializing():return native_block(self,*args,**kwargs)
        if not self.config.bam_enabled or self.config.dense_conn:raise ValueError('BAM-only frozen-checkpoint self bias')
        idx=kwargs.get('layer_index')
        if idx is None:idx=self.layer_inx
        c=dict(layer=idx,native_depth=0,calls=0);_STACK.append(c)
        try:
            result=native_block(self,*args,**kwargs)
            if c['calls']!=1:raise ValueError(('Expected one complete attention output per block',self.scope.path,c['calls']))
            return result
        finally:_STACK.pop()
    fusion.SubDecoderLayer.__call__=layer
    def modify(q,k,v,seg,out,window,soft_cap):
        c=_STACK[-1];p=probabilities(q,k,seg,window,soft_cap)
        if out.shape!=v.shape or q.shape[1]!=out.shape[1]:raise ValueError('Self bias query/value/output alignment mismatch')
        c['calls']+=1
        # Match the observed q>=64 hotspot; do not perturb the short prefix.
        query_mask=(jnp.arange(q.shape[1])>=min(64,q.shape[1]-1))[None,:,None]
        return adjust(out,v,p,BIASES[c['layer']]*query_mask)
    original_splash=attentions._bam_splash_attention
    @functools.wraps(original_splash)
    def splash(q,k,v,seg,cfg,mesh,window=None,**kwargs):
        out=original_splash(q,k,v,seg,cfg,mesh,window=window,**kwargs)
        if BIASES is not None and _STACK and not _STACK[-1]['native_depth']:out=modify(q,k,v,seg,out,window,cfg.attn_logits_soft_cap)
        return out
    attentions._bam_splash_attention=splash
    original_chunk=attentions.BamAttention._query_chunk_op
    @functools.wraps(original_chunk)
    def chunk(self,q,k,v,seg,window,*args,**kwargs):
        active=BIASES is not None and bool(_STACK) and not _STACK[-1]['native_depth']
        if active:_STACK[-1]['native_depth']+=1
        try:out=original_chunk(self,q,k,v,seg,window,*args,**kwargs)
        finally:
            if active:_STACK[-1]['native_depth']-=1
        if active:
            if out[1] is not None:raise ValueError('Self-bias probe does not support full-M attention aggregation')
            return (modify(q,k,v,seg,out[0],window,self.config.attn_logits_soft_cap),)+out[1:]
        return out
    attentions.BamAttention._query_chunk_op=chunk
    original_attention_block=attentions.BamAttention._attention_block
    @functools.wraps(original_attention_block)
    def attention_block(self,q,k,v,seg,**kwargs):
        out=original_attention_block(self,q,k,v,seg,**kwargs)
        if BIASES is not None and _STACK and not _STACK[-1]['native_depth']:
            if kwargs['q0']!=0 or kwargs['s0']!=0 or out[1] is not None:raise ValueError('Unsupported partial fallback attention')
            out=(modify(q,k,v,seg,out[0],kwargs['window_size'],self.config.attn_logits_soft_cap),)+out[1:]
        return out
    attentions.BamAttention._attention_block=attention_block

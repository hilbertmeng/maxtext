"""Observe native Q/K/V and MHA outputs; no replacement of model operations."""
import functools,re
import jax
import jax.numpy as jnp
ACTIVE=False
PREFIX=55
_INSTALLED=False
_STACK=[]
def weights(q,k,segments=None,window=None,cap=None):
    if q.shape[2]!=k.shape[2]:k=jnp.repeat(k,q.shape[2]//k.shape[2],axis=2)
    logits=jnp.einsum('bqhd,bkhd->bhqk',q.astype(jnp.float32),k.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    if cap:logits=jnp.tanh(logits/cap)*cap
    idx=jnp.arange(q.shape[1]);src=jnp.arange(k.shape[1]);valid=(src[None]<=idx[:,None])[None,None]
    if window is not None:valid=valid&(src[None]>idx[:,None]-window)[None,None]
    if segments is not None:valid=valid&(segments[:,None,:,None]==segments[:,None,None,:])&(segments[:,None,:,None]!=0)
    return jax.nn.softmax(jnp.where(valid,logits,-1e30),-1)
def save(module,c,index):
    n=PREFIX;q,k,v,out=[c[x][:1,:n].astype(jnp.float32) for x in ['q','k','v','out']]
    if v.shape[2]!=q.shape[2]:v=jnp.repeat(v,q.shape[2]//v.shape[2],axis=2)
    seg=None if c['segments'] is None else c['segments'][:1,:n]
    alpha=weights(q,k,seg,c['window'],c['cap'])
    reconstructed=jnp.einsum('bhqk,bkhd->bqhd',alpha,v,precision=jax.lax.Precision.HIGHEST)
    error=jnp.stack([jnp.sum((reconstructed-out)**2,axis=(0,1,3)),jnp.sum(out*out,axis=(0,1,3))],axis=-1)
    for name,value in [('alpha',alpha[0]),('query',q[0]),('key',k[0]),('value',v[0]),('native_output',out[0]),('error',error),('layer_index',jnp.asarray(index,jnp.int32))]:module.sow('intermediates','sentence_'+name,value)
def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,fusion
    native_layer=fusion.SubDecoderLayer.__call__
    @functools.wraps(native_layer)
    def layer(self,*args,**kw):
        if not ACTIVE or self.is_initializing() or not getattr(self.config,'bam_enabled',False):return native_layer(self,*args,**kw)
        idx=kw.get('layer_index');idx=self.layer_inx if idx is None else idx
        c=dict(index=idx);_STACK.append(c)
        try:
            result=native_layer(self,*args,**kw)
            if not c.get('saved'):raise ValueError(('Missing sentence BAM layer',self.scope.path))
            return result
        finally:_STACK.pop()
    fusion.SubDecoderLayer.__call__=layer
    def remember(q,k,v,out,seg,window,cap):
        if ACTIVE and _STACK and q.shape[1]==k.shape[1] and q.shape[1]>=PREFIX:
            c=_STACK[-1]
            if q.shape[1]>=c.get('length',0):c.update(q=q,k=k,v=v,out=out,segments=seg,window=window,cap=cap,length=q.shape[1])
    if hasattr(attentions,'_bam_splash_attention'):
        native=attentions._bam_splash_attention
        @functools.wraps(native)
        def splash(q,k,v,seg,cfg,mesh,window=None,**kw):
            out=native(q,k,v,seg,cfg,mesh,window=window,**kw);remember(q,k,v,out,seg,window,cfg.attn_logits_soft_cap);return out
        attentions._bam_splash_attention=splash
    if hasattr(attentions.BamAttention,'_query_chunk_op'):
        native_chunk=attentions.BamAttention._query_chunk_op
        @functools.wraps(native_chunk)
        def chunk(self,q,k,v,seg,window,*args,**kw):
            out=native_chunk(self,q,k,v,seg,window,*args,**kw);remember(q,k,v,out[0],seg,window,self.config.attn_logits_soft_cap);return out
        attentions.BamAttention._query_chunk_op=chunk
    if hasattr(attentions.BamAttention,'_attention_block'):
        native_block=attentions.BamAttention._attention_block
        @functools.wraps(native_block)
        def block(self,q,k,v,seg,**kw):
            out=native_block(self,q,k,v,seg,**kw)
            if kw['q0']==kw['s0']==0:remember(q,k,v,out[0],seg,kw['window_size'],self.config.attn_logits_soft_cap)
            return out
        attentions.BamAttention._attention_block=block
    native_proj=attentions.BamAttention.out_projection
    @functools.wraps(native_proj)
    def projection(self,*args,**kw):
        result=native_proj(self,*args,**kw)
        if ACTIVE and _STACK:
            c=_STACK[-1];assert 'q' in c and not c.get('saved');save(self,c,c['index']);c['saved']=True
        return result
    attentions.BamAttention.out_projection=projection
    native_mha=attentions.AttentionOp.__call__
    @functools.wraps(native_mha)
    def mha(self,q,k,v,seg,model_mode,*args,**kw):
        out=native_mha(self,q,k,v,seg,model_mode,*args,**kw)
        if ACTIVE and not self.is_initializing():
            found=[re.fullmatch(r'layers_(\d+)',x) for x in self.scope.path];found=[x for x in found if x]
            if len(found)!=1:raise ValueError(('Mudd observation requires unscanned logical layer paths',self.scope.path))
            window=None if str(self.attention_type).lower().endswith('global') else self.sliding_window_size
            save(self,dict(q=q,k=k,v=v,out=out,segments=seg,window=window,cap=self.attn_logits_soft_cap),int(found[0][1]))
        return out
    attentions.AttentionOp.__call__=mha

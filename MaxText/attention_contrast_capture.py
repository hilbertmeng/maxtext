"""Matched attention-operator counterfactuals and common-residual output moments.

Only q>=64 is changed. Modes: reference FP32 AV, no-self, causal uniform,
and uniform among eligible keys while preserving native sink-key probabilities.
Reference replacement controls numerical differences from the native kernel.
No optimizer or parameter changes. Observations use effective native W_o weights
in FP32 linear statistics; output bias belongs to no individual component.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
CONTROLS=None
CAPTURE=False
GEOMETRY=False
TOKENS=None
_INSTALLED=False
_STACK=[]
MODE_NAMES=('reference','no_self','uniform','sink_uniform')
VECTOR_NAMES=('native_MHA','reference_AV','uniform_AV','sink_uniform_AV','no_self_AV','value_self','direct_O','total_attention')
EXPOSURE_NAMES=('diagonal_mass','eligible_mass','L1_from_reference','active_ratio','self_mass_gt_099')
def q_positions(t):return np.unique(np.linspace(min(64,t-1),t-1,min(16,t),dtype=np.int32))
def operators(q,k,v,segments,tokens,window=None,soft_cap=0.):
    b,t,h,d=q.shape;chunk=min(64,t);start=min(64,t-1)
    if t!=k.shape[1] or k.shape!=v.shape or h!=k.shape[2] or t%chunk:raise ValueError('Full aligned, equal-head Q/K/V required')
    source=jnp.arange(t);allowed=(source[None]>=2)&(tokens!=0)&(tokens!=15)&(tokens!=187)
    def block(_,i):
        ids=i*chunk+jnp.arange(chunk);logits=jnp.einsum('bqhd,bshd->bhqs',q[:,ids].astype(jnp.float32),k.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
        if soft_cap:logits=jnp.tanh(logits/soft_cap)*soft_cap
        valid=(source[None]<=ids[:,None])[None,None]
        if window is not None:valid=valid&(source[None]>ids[:,None]-window)[None,None]
        if segments is not None:valid=valid&(segments[:,None,ids,None]==segments[:,None,None,:])&(segments[:,None,ids,None]!=0)
        valid=jnp.broadcast_to(valid,(b,h,chunk,t));alpha=jax.nn.softmax(jnp.where(valid,logits,-1e30),-1)
        no_self_valid=valid&~((source[None,:]==ids[:,None])&(ids[:,None]>=start))[None,None]
        # At the first token of a packed segment there may be no other valid key.
        # Retain native support rather than manufacturing a future-token read.
        no_self_valid=jnp.where(no_self_valid.any(-1,keepdims=True),no_self_valid,valid)
        no_self=jax.nn.softmax(jnp.where(no_self_valid,logits,-1e30),-1)
        uniform=valid.astype(jnp.float32)/jnp.maximum(valid.sum(-1,keepdims=True),1)
        eligible=valid&allowed[:,None,None,:];mass=jnp.sum(alpha*eligible,-1,keepdims=True)
        sink_uniform=alpha*(~eligible)+mass*eligible/jnp.maximum(eligible.sum(-1,keepdims=True),1)
        maps=(alpha,no_self,uniform,sink_uniform)
        av=jnp.stack([jnp.einsum('bhqs,bshk->bqhk',a,v.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST) for a in maps],axis=1)
        keep=(ids>=start)[None,None];stats=[]
        for a in maps:
            diag=jnp.take_along_axis(a,jnp.broadcast_to(ids[None,None,:,None],a.shape[:-1]+(1,)),-1)[...,0]
            top=jnp.argmax(a,-1);active=jnp.take_along_axis(eligible,top[...,None],-1)[...,0]
            vals=(diag,jnp.sum(a*eligible,-1),jnp.sum(jnp.abs(a-alpha),-1),active.astype(jnp.float32),(diag>.99).astype(jnp.float32))
            stats.append(jnp.stack([jnp.sum(x*keep,axis=(1,2)) for x in vals],axis=-1))
        return None,(av,jnp.stack(stats,axis=1))
    _,(av,stats)=jax.lax.scan(block,None,jnp.arange(t//chunk))
    av=av.transpose(1,2,0,3,4,5).reshape(b,4,t,h,d)
    stats=stats.sum(0)/((t-start)*h)
    return av,stats
def apply_operator(native,refs,coefficients):
    chosen=jnp.einsum('m,bmthk->bthk',coefficients.astype(jnp.float32),refs)
    modified=native.astype(jnp.float32)*(1-coefficients.sum())+chosen
    mask=(jnp.arange(native.shape[1])>=min(64,native.shape[1]-1))[None,:,None,None]
    return jnp.where(mask,modified,native.astype(jnp.float32)).astype(native.dtype)
def _unbox(x):return x.unbox(apply_constraint=False) if hasattr(x,'unbox') else x
def gram(vectors):
    x=jnp.stack([v.astype(jnp.float32) for v in vectors],axis=1).reshape(vectors[0].shape[0],len(vectors),-1)
    return jnp.einsum('bmi,bni->bmn',x,x,precision=jax.lax.Precision.HIGHEST)
def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,fusion
    original_layer=fusion.SubDecoderLayer.__call__
    @functools.wraps(original_layer)
    def layer(self,*args,**kwargs):
        if (CONTROLS is None and not CAPTURE) or self.is_initializing():return original_layer(self,*args,**kwargs)
        idx=kwargs.get('layer_index')
        if idx is None:idx=self.layer_inx
        c=dict(layer=idx,depth=0,calls=0);_STACK.append(c)
        try:
            out=original_layer(self,*args,**kwargs)
            if c['calls']!=1 or (CAPTURE and not c.get('projected',False)):raise ValueError(('Missing/duplicate attention operation',self.scope.path,c['calls']))
            return out
        finally:_STACK.pop()
    fusion.SubDecoderLayer.__call__=layer
    def transform(q,k,v,seg,out,window,cap):
        c=_STACK[-1];c['calls']+=1
        if out.shape!=v.shape or TOKENS is None:raise ValueError('Missing aligned attention output/tokens')
        refs,stats=operators(q,k,v,seg,TOKENS,window,cap)
        coeff=jnp.zeros(4,jnp.float32) if CONTROLS is None else CONTROLS[c['layer']]
        changed=apply_operator(out,refs,coeff)
        exposure=(1-coeff.sum())*stats[:,0]+jnp.einsum('m,bme->be',coeff,stats)
        c.update(native=out,applied=changed,refs=refs,value=v,exposure=jnp.stack([stats[:,0],exposure],axis=1))
        return changed
    original_mha=attentions.AttentionOp.__call__
    @functools.wraps(original_mha)
    def mha(self,q,k,v,seg,*args,**kwargs):
        out=original_mha(self,q,k,v,seg,*args,**kwargs)
        if _STACK and not _STACK[-1]['depth']:
            window=None if str(self.attention_type).lower().endswith('global') else self.sliding_window_size
            out=transform(q,k,v,seg,out,window,self.attn_logits_soft_cap)
        return out
    attentions.AttentionOp.__call__=mha
    if hasattr(attentions,'_bam_splash_attention'):
        original_splash=attentions._bam_splash_attention
        @functools.wraps(original_splash)
        def splash(q,k,v,seg,cfg,mesh,window=None,**kwargs):
            out=original_splash(q,k,v,seg,cfg,mesh,window=window,**kwargs)
            if _STACK and not _STACK[-1]['depth']:out=transform(q,k,v,seg,out,window,cfg.attn_logits_soft_cap)
            return out
        attentions._bam_splash_attention=splash
    cls=attentions.BamAttention
    if hasattr(cls,'_query_chunk_op'):
        original_chunk=cls._query_chunk_op
        @functools.wraps(original_chunk)
        def chunk(self,q,k,v,seg,window,*args,**kwargs):
            active=bool(_STACK) and not _STACK[-1]['depth']
            if active:_STACK[-1]['depth']+=1
            try:out=original_chunk(self,q,k,v,seg,window,*args,**kwargs)
            finally:
                if active:_STACK[-1]['depth']-=1
            if active:
                if out[1] is not None:raise ValueError('Full-M attention aggregation unsupported')
                return (transform(q,k,v,seg,out[0],window,self.config.attn_logits_soft_cap),)+out[1:]
            return out
        cls._query_chunk_op=chunk
    if hasattr(cls,'_attention_block'):
        original_block=cls._attention_block
        @functools.wraps(original_block)
        def block(self,q,k,v,seg,**kwargs):
            out=original_block(self,q,k,v,seg,**kwargs)
            if _STACK and not _STACK[-1]['depth']:
                if kwargs['q0']!=0 or kwargs['s0']!=0 or out[1] is not None:raise ValueError('Unsupported fallback mask')
                out=(transform(q,k,v,seg,out[0],kwargs['window_size'],self.config.attn_logits_soft_cap),)+out[1:]
            return out
        cls._attention_block=block
    if hasattr(cls,'_record_concat_amplitude'):
        original_amp=cls._record_concat_amplitude
        @functools.wraps(original_amp)
        def amplitude(self,name,bam,standard):
            if _STACK and name=='local_o':_STACK[-1]['direct_o']=bam
            return original_amp(self,name,bam,standard)
        cls._record_concat_amplitude=amplitude
    try:from layers import bam_pallas
    except ImportError:bam_pallas=None
    if bam_pallas is not None:
        original_read=bam_pallas.read
        @functools.wraps(original_read)
        def read(*args,**kwargs):
            out=original_read(*args,**kwargs)
            if _STACK:_STACK[-1]['direct_o']=out[3]
            return out
        bam_pallas.read=read
    original_projection=attentions.Attention.out_projection
    @functools.wraps(original_projection)
    def projection(self,dim,out):
        value=original_projection(self,dim,out)
        if not _STACK or not CAPTURE:return value
        c=_STACK[-1];c['projected']=True
        if 'native' not in c:raise ValueError('Missing full attention before W_o')
        self.sow('intermediates','stream_contrast_exposure',c['exposure'])
        self.sow('intermediates','stream_contrast_layer_index',jnp.asarray(c['layer'],jnp.int32))
        if GEOMETRY:
            q=q_positions(out.shape[1]);native=c['native'];refs=c['refs'];direct=c.get('direct_o',jnp.zeros_like(native))
            vectors=[native[:,q],refs[:,0,q],refs[:,2,q],refs[:,3,q],refs[:,1,q],c['value'][:,q],direct[:,q],out[:,q]]
            kernel=_unbox(self.variables['params']['out']['kernel']).astype(self.dtype).astype(jnp.float32)
            projected=[jnp.einsum('bqhk,hkd->bqd',v.astype(jnp.float32),kernel,precision=jax.lax.Precision.HIGHEST) for v in vectors]
            bias=_unbox(self.variables['params']['out'].get('bias',jnp.asarray(0.))).astype(self.dtype).astype(jnp.float32)
            output_error=jnp.sum((projected[-1]-(value[:,q].astype(jnp.float32)-bias))**2,axis=(1,2))
            start=min(64,out.shape[1]-1);a=native[:,start:].astype(jnp.float32);b=refs[:,0,start:]
            closure=out[:,start:].astype(jnp.float32)-c['applied'][:,start:].astype(jnp.float32)-direct[:,start:].astype(jnp.float32)
            errors=jnp.stack([jnp.sum((a-b)**2,axis=(1,2,3)),jnp.sum(a*a,axis=(1,2,3)),output_error,jnp.sum(closure*closure,axis=(1,2,3)),jnp.sum(out[:,start:].astype(jnp.float32)**2,axis=(1,2,3))],axis=-1)
            self.sow('intermediates','stream_contrast_raw_gram',gram(vectors))
            self.sow('intermediates','stream_contrast_projected_gram',gram(projected))
            self.sow('intermediates','stream_contrast_errors',errors)
        return value
    attentions.Attention.out_projection=projection

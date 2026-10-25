"""Observe native attention probability and value routes without replacing outputs."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
ACTIVE=False
_INSTALLED=False
_STACK=[]
RANK_NAMES=('V','dynamicV','staticV','MHA','O','dynamicO','dynamicAV','nonself_dynamicAV')
ENERGY_NAMES=('MHA','O','dynamicO','staticO','dynamicV','V','staticV','AV','dynamicAV','self_dynamicAV','nonself_dynamicAV','dynamicV_dot_dynamicO','dynamicAV_dot_dynamicO','dynamicAV_parallel_dynamicO','V_closure_error','O_closure_error','AV_reconstruction_error','MHA_dot_O','self_probability')
def positions(t):return np.unique(np.linspace(min(64,t-1),t-1,min(16,t),dtype=np.int32))
def attention_map(q,k,segments,qpos,window=None,soft_cap=0.):
    logits=jnp.einsum('bqhd,bshd->bhqs',q[:,qpos].astype(jnp.float32),k.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    if soft_cap:logits=jnp.tanh(logits/soft_cap)*soft_cap
    src=jnp.arange(k.shape[1]);valid=(src[None]<=qpos[:,None])[None]
    if window is not None:valid=valid&(src[None]>qpos[:,None]-window)[None]
    if segments is not None:valid=valid&(segments[:,qpos,None]==segments[:,None,:])&(segments[:,qpos,None]!=0)
    return jax.nn.softmax(jnp.where(valid[:,None],logits,-1e30),axis=-1)
def self_statistics(q,k,segments,window=None,soft_cap=0.):
    t=q.shape[1];chunk=min(64,t)
    assert t%chunk==0 and t==k.shape[1]
    # Chunk only observer QK work; the native attention result remains untouched.
    def block(_,i):
        ids=i*chunk+jnp.arange(chunk)
        a=attention_map(q,k,segments,ids,window,soft_cap)
        diag=jnp.take_along_axis(a,jnp.broadcast_to(ids[None,None,:,None],a.shape[:-1]+(1,)),axis=-1)[...,0]
        entropy=-jnp.sum(a*jnp.log(jnp.maximum(a,1e-30)),axis=-1)/jnp.log(jnp.maximum(ids+1,2))[None,None]
        x=jnp.stack([diag,(jnp.argmax(a,-1)==ids[None,None]).astype(jnp.float32),entropy],axis=-1)
        keep=ids>=min(64,t-1)
        return None,(jnp.sum(x,axis=2),jnp.sum(x*keep[None,None,:,None],axis=2),jnp.sum(keep))
    _,(all_,tail,count)=jax.lax.scan(block,None,jnp.arange(t//chunk))
    return jnp.stack([all_.sum(0)/t,tail.sum(0)/count.sum()],axis=-2)
def diversity(x):
    """Participation rank (not entropy rank), raw and unit-row directions."""
    x=x.astype(jnp.float32)
    def rank(y):
        g=jnp.einsum('bqhk,bqjk->bqhj',y,y,precision=jax.lax.Precision.HIGHEST)
        trace=jnp.trace(g,axis1=-2,axis2=-1)
        return trace*trace/jnp.maximum(jnp.sum(g*g,axis=(-2,-1)),1e-30)
    norm=jnp.sqrt(jnp.sum(x*x,axis=-1,keepdims=True))
    unit=x/jnp.maximum(norm,1e-20)
    return jnp.stack([rank(x),rank(unit),jnp.mean(x*x,axis=(-2,-1))],axis=-1)
def route_statistics(q,k,v,segments,mha,o,dyv,stv,dyo,sto,window=None,soft_cap=0.):
    qp=jnp.asarray(positions(q.shape[1]));alpha=attention_map(q,k,segments,qp,window,soft_cap)
    av=jnp.einsum('bhqs,bshk->bqhk',alpha,v.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    dav=jnp.einsum('bhqs,bshk->bqhk',alpha,dyv.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    diag=jnp.take_along_axis(alpha,jnp.broadcast_to(qp[None,None,:,None],alpha.shape[:-1]+(1,)),-1)[...,0].transpose(0,2,1)
    v,dyv,stv,mha,o,dyo,sto=[x[:,qp].astype(jnp.float32) for x in [v,dyv,stv,mha,o,dyo,sto]]
    selfdy=diag[...,None]*dyv;nonself=dav-selfdy
    energy=lambda a:jnp.sum(a*a,axis=-1)
    dot=lambda a,b:jnp.sum(a*b,axis=-1)
    e=[energy(x) for x in [mha,o,dyo,sto,dyv,v,stv,av,dav,selfdy,nonself]]
    e+=[dot(dyv,dyo),dot(dav,dyo),dot(dav,dyo)**2/jnp.maximum(energy(dyo),1e-30),energy(v-dyv-stv),energy(o-dyo-sto),energy(mha-av),dot(mha,o),diag]
    ranks=jnp.stack([diversity(x) for x in [v,dyv,stv,mha,o,dyo,dav,nonself]],axis=-2)
    return jnp.stack(e,axis=-1),ranks
def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,bam_pallas
    native=attentions.BamAttention.__call__
    @functools.wraps(native)
    def call(self,*args,**kwargs):
        if not ACTIVE or self.is_initializing():return native(self,*args,**kwargs)
        _STACK.append(dict(module=self))
        try:return native(self,*args,**kwargs)
        finally:_STACK.pop()
    attentions.BamAttention.__call__=call
    original_read=bam_pallas.read
    @functools.wraps(original_read)
    def read(*args,**kwargs):
        out=original_read(*args,**kwargs)
        if ACTIVE and _STACK:
            c=_STACK[-1];h=out[3].shape[2];dy=list(args);dy[1]=dy[1].at[2*h:4*h].set(0)
            st=list(args);st[7]=jnp.full_like(st[7],-80);st[8]=jnp.full_like(st[8],-80)
            dyn=original_read(*dy,**kwargs);sta=original_read(*st,**kwargs)
            c.update(o=out[3],dyv=dyn[2],dyo=dyn[3],stv=sta[2],sto=sta[3])
        return out
    bam_pallas.read=read
    original_vo=attentions.BamAttention._independent_local_vo
    @functools.wraps(original_vo)
    def vo(self,*args,**kwargs):
        out=original_vo(self,*args,**kwargs)
        if ACTIVE and _STACK:_STACK[-1].update(dyv=out[0],dyo=out[1])
        return out
    attentions.BamAttention._independent_local_vo=vo
    original_static=attentions.BamAttention._static_column
    @functools.wraps(original_static)
    def static(self,m,arm,*args,**kwargs):
        out=original_static(self,m,arm,*args,**kwargs)
        if ACTIVE and _STACK and arm in ('v','o'):_STACK[-1]['st'+arm]=out
        return out
    attentions.BamAttention._static_column=static
    def record_qkv(q,k,v,seg,window,soft_cap):
        if ACTIVE and _STACK:
            c=_STACK[-1]
            if 'q' not in c or q.shape[1]>=c['q'].shape[1]:c.update(q=q,k=k,v=v,seg=seg,window=window,soft_cap=soft_cap)
    original_splash=attentions._bam_splash_attention
    @functools.wraps(original_splash)
    def splash(q,k,v,seg,cfg,mesh,window=None,**kwargs):
        record_qkv(q,k,v,seg,window,cfg.attn_logits_soft_cap)
        out=original_splash(q,k,v,seg,cfg,mesh,window=window,**kwargs)
        if ACTIVE and _STACK:_STACK[-1]['mha']=out
        return out
    attentions._bam_splash_attention=splash
    original_chunk=attentions.BamAttention._query_chunk_op
    @functools.wraps(original_chunk)
    def chunk(self,q,k,v,seg,window,*args,**kwargs):
        record_qkv(q,k,v,seg,window,self.config.attn_logits_soft_cap)
        out=original_chunk(self,q,k,v,seg,window,*args,**kwargs)
        if ACTIVE and _STACK:_STACK[-1]['mha']=out[0]
        return out
    attentions.BamAttention._query_chunk_op=chunk
    original_block=attentions.BamAttention._attention_block
    @functools.wraps(original_block)
    def block(self,q,k,v,seg,**kwargs):
        full=q.shape[1]==k.shape[1] and kwargs['q0']==0 and kwargs['s0']==0
        if full:record_qkv(q,k,v,seg,kwargs['window_size'],self.config.attn_logits_soft_cap)
        out=original_block(self,q,k,v,seg,**kwargs)
        if ACTIVE and _STACK and full and 'mha' not in _STACK[-1]:_STACK[-1]['mha']=out[0]
        return out
    attentions.BamAttention._attention_block=block
    original_amp=attentions.BamAttention._record_concat_amplitude
    @functools.wraps(original_amp)
    def amplitude(self,name,bam,standard):
        if ACTIVE and _STACK and name=='local_o':_STACK[-1].update(o=bam,mha=standard)
        return original_amp(self,name,bam,standard)
    attentions.BamAttention._record_concat_amplitude=amplitude
    original_projection=attentions.BamAttention.out_projection
    @functools.wraps(original_projection)
    def projection(self,dim,out):
        if ACTIVE and _STACK:
            c=_STACK[-1];need=['q','k','v','seg','mha','o','dyv','stv','dyo','sto','window','soft_cap']
            if any(k not in c for k in need):raise ValueError(('Missing value route',self.scope.path,list(c)))
            if c['q'].shape[1]!=out.shape[1]:raise ValueError('Captured query chunk rather than full sequence')
            e,r=route_statistics(*(c[k] for k in need))
            s=self_statistics(c['q'],c['k'],c['seg'],c['window'],c['soft_cap'])
            self.sow('intermediates','stream_route_energy',e)
            self.sow('intermediates','stream_route_rank',r)
            self.sow('intermediates','stream_route_self',s)
        return original_projection(self,dim,out)
    attentions.BamAttention.out_projection=projection

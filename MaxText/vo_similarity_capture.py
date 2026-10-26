"""Read-only same-token/head V/O moments from native BAM reads.

Dynamic/static Pallas components use shadow calls only; the original read,
attention and W_o outputs are returned unchanged. XLA components are observed
at their native read/gate sites. Zero-energy vectors are retained as zero
moments and excluded by the host analyzer, never assigned a cosine of zero.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
ACTIVE=False
_INSTALLED=False
_STACK=[]
PAIR_NAMES=('dynamicV_dynamicO','totalV_totalO','staticV_staticO','MHA_totalO','keyV_keyO')
MOMENT_NAMES=('left_energy','right_energy','dot')
ERROR_NAMES=('V_closure','V_energy','O_closure','O_energy')
def positions(t):return np.unique(np.linspace(min(64,t-1),t-1,min(64,t),dtype=np.int32))
def moments(a,b):
    a,b=a.astype(jnp.float32),b.astype(jnp.float32)
    return jnp.stack([jnp.sum(a*a,-1),jnp.sum(b*b,-1),jnp.sum(a*b,-1)],axis=-1)
def attention_stats(q,k,segments,window=None,soft_cap=0.):
    ids=jnp.asarray(positions(q.shape[1]));src=jnp.arange(k.shape[1])
    logits=jnp.einsum('bqhd,bshd->bhqs',q[:,ids].astype(jnp.float32),k.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
    if soft_cap:logits=jnp.tanh(logits/soft_cap)*soft_cap
    valid=(src[None]<=ids[:,None])[None,None]
    if window is not None:valid=valid&(src[None]>ids[:,None]-window)[None,None]
    if segments is not None:valid=valid&(segments[:,None,ids,None]==segments[:,None,None,:])&(segments[:,None,ids,None]!=0)
    a=jax.nn.softmax(jnp.where(valid,logits,-1e30),-1)
    diagonal=jnp.take_along_axis(a,jnp.broadcast_to(ids[None,None,:,None],a.shape[:-1]+(1,)),-1)[...,0]
    own=(jnp.argmax(a,-1)==ids[None,None]).astype(jnp.float32)
    return jnp.stack([diagonal,own],axis=-1).transpose(0,2,1,3)
def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,bam_pallas,fusion
    original_layer=fusion.SubDecoderLayer.__call__
    @functools.wraps(original_layer)
    def layer(self,*args,**kwargs):
        if not ACTIVE or self.is_initializing():return original_layer(self,*args,**kwargs)
        idx=kwargs.get('layer_index')
        if idx is None:idx=self.layer_inx
        c=dict(layer=idx);_STACK.append(c)
        try:
            result=original_layer(self,*args,**kwargs)
            if not c.get('recorded'):raise ValueError(('Missing BAM V/O observation',self.scope.path))
            return result
        finally:_STACK.pop()
    fusion.SubDecoderLayer.__call__=layer
    native_read=bam_pallas.read
    @functools.wraps(native_read)
    def read(*args,**kwargs):
        out=native_read(*args,**kwargs)
        if ACTIVE and _STACK:
            h=out[3].shape[2];dynamic=list(args);dynamic[1]=dynamic[1].at[2*h:4*h].set(0)
            static=list(args);static[7]=jnp.full_like(static[7],-80);static[8]=jnp.full_like(static[8],-80)
            dy=native_read(*dynamic,**kwargs);st=native_read(*static,**kwargs);key=args[6]
            if key.shape[-2]==2*h:vk,ok=jnp.split(key,2,axis=-2)
            elif key.shape[-2]==h:vk=ok=key
            else:raise ValueError(('Unexpected packed V/O key shape',key.shape,h))
            _STACK[-1].update(o=out[3],dyv=dy[2],dyo=dy[3],stv=st[2],sto=st[3],vk=vk,ok=ok)
        return out
    bam_pallas.read=read
    native_vo=attentions.BamAttention._independent_local_vo
    @functools.wraps(native_vo)
    def vo(self,*args,**kwargs):
        out=native_vo(self,*args,**kwargs)
        if ACTIVE and _STACK:_STACK[-1].update(dyv=out[0],dyo=out[1])
        return out
    attentions.BamAttention._independent_local_vo=vo
    native_static=attentions.BamAttention._static_column
    @functools.wraps(native_static)
    def static(self,m,arm,*args,**kwargs):
        out=native_static(self,m,arm,*args,**kwargs)
        if ACTIVE and _STACK and arm in ('v','o'):_STACK[-1]['st'+arm]=out
        return out
    attentions.BamAttention._static_column=static
    native_key=attentions.BamAttention._project_column_read_key
    @functools.wraps(native_key)
    def key(self,name,*args,**kwargs):
        out=native_key(self,name,*args,**kwargs)
        if ACTIVE and _STACK and name in ('W_R','W_lv_c8'):
            value=jnp.squeeze(out,-2) if out.ndim==5 else out
            if value.ndim!=4:raise ValueError(('Unexpected address dimensions',name,value.shape))
            _STACK[-1]['ok' if name=='W_R' else 'vk']=value
        return out
    attentions.BamAttention._project_column_read_key=key
    def record(q,k,v,seg,window,cap):
        if ACTIVE and _STACK and q.shape[1]==k.shape[1] and ('q' not in _STACK[-1] or q.shape[1]>=_STACK[-1]['q'].shape[1]):_STACK[-1].update(q=q,k=k,v=v,seg=seg,window=window,cap=cap)
    native_splash=attentions._bam_splash_attention
    @functools.wraps(native_splash)
    def splash(q,k,v,seg,cfg,mesh,window=None,**kwargs):
        record(q,k,v,seg,window,cfg.attn_logits_soft_cap);out=native_splash(q,k,v,seg,cfg,mesh,window=window,**kwargs)
        if ACTIVE and _STACK:_STACK[-1]['mha']=out
        return out
    attentions._bam_splash_attention=splash
    native_chunk=attentions.BamAttention._query_chunk_op
    @functools.wraps(native_chunk)
    def chunk(self,q,k,v,seg,window,*args,**kwargs):
        record(q,k,v,seg,window,self.config.attn_logits_soft_cap);out=native_chunk(self,q,k,v,seg,window,*args,**kwargs)
        if ACTIVE and _STACK:_STACK[-1]['mha']=out[0]
        return out
    attentions.BamAttention._query_chunk_op=chunk
    native_block=attentions.BamAttention._attention_block
    @functools.wraps(native_block)
    def block(self,q,k,v,seg,**kwargs):
        full=q.shape[1]==k.shape[1] and kwargs['q0']==kwargs['s0']==0
        if full:record(q,k,v,seg,kwargs['window_size'],self.config.attn_logits_soft_cap)
        out=native_block(self,q,k,v,seg,**kwargs)
        if ACTIVE and _STACK and full:_STACK[-1]['mha']=out[0]
        return out
    attentions.BamAttention._attention_block=block
    native_amp=attentions.BamAttention._record_concat_amplitude
    @functools.wraps(native_amp)
    def amplitude(self,name,bam,standard):
        if ACTIVE and _STACK and name=='local_o':_STACK[-1].update(o=bam,mha=standard)
        return native_amp(self,name,bam,standard)
    attentions.BamAttention._record_concat_amplitude=amplitude
    native_projection=attentions.BamAttention.out_projection
    @functools.wraps(native_projection)
    def projection(self,dim,out):
        result=native_projection(self,dim,out)
        if not ACTIVE or not _STACK:return result
        c=_STACK[-1]
        if not getattr(self.config,'bam_local_vo_independent_keys',False) and 'vk' not in c:c['vk']=c.get('ok')
        need=['q','k','v','o','dyv','dyo','stv','sto','mha','vk','ok','seg','window','cap']
        if any(k not in c or c[k] is None and k not in ['seg','window','cap'] for k in need):raise ValueError(('Missing V/O fields',self.scope.path,list(c)))
        if c.get('recorded'):raise ValueError('Duplicate V/O capture')
        if c['q'].shape[1]!=out.shape[1] or any(c[k].shape[1]!=out.shape[1] for k in ['v','o','dyv','dyo','stv','sto','mha','vk','ok']):raise ValueError('Captured a query/key chunk rather than full sequence')
        ids=positions(out.shape[1]);arrays=[c[k][:,ids].astype(jnp.float32) for k in ['dyv','dyo','v','o','stv','sto','mha','vk','ok']];dv,do,v,o,sv,so,mha,vk,ok=arrays
        pairs=jnp.stack([moments(a,b) for a,b in [(dv,do),(v,o),(sv,so),(mha,o),(vk,ok)]],axis=-2)
        err=lambda x:jnp.sum(x*x,axis=(1,2,3))
        errors=jnp.stack([err(v-dv-sv),err(v),err(o-do-so),err(o)],axis=-1)
        self.sow('intermediates','stream_vo_moments',pairs)
        self.sow('intermediates','stream_vo_errors',errors)
        self.sow('intermediates','stream_vo_attention',attention_stats(c['q'],c['k'],c['seg'],c['window'],c['cap']))
        self.sow('intermediates','stream_vo_layer_index',jnp.asarray(c['layer'],jnp.int32));c['recorded']=True
        return result
    attentions.BamAttention.out_projection=projection

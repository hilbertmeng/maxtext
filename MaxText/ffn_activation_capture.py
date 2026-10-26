"""Observe signed pre-GELU activations and the separate M-write gate, without changing outputs."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

ACTIVE=False
_INSTALLED=False
_ATTN_STACK=[]
METRICS=('neuron_positive_count','position_positive_count','position_mean','position_rms','position_zero_count')

def positions(length):return np.arange(length,dtype=np.int32)

def statistics(x):
    if x.ndim!=3:raise ValueError(('Expected batch, sequence, neurons',x.shape))
    positive=x>0
    xf=x.astype(jnp.float32)
    return dict(neuron_positive_count=jnp.sum(positive,axis=1,dtype=jnp.int32),
                position_positive_count=jnp.sum(positive,axis=-1,dtype=jnp.int32),
                position_mean=jnp.mean(xf,axis=-1),
                position_rms=jnp.sqrt(jnp.mean(xf*xf,axis=-1)),
                position_zero_count=jnp.sum(x==0,axis=-1,dtype=jnp.int32))

def balance_statistics(mha,m_read,total):
    if mha.shape!=m_read.shape or mha.shape!=total.shape or mha.ndim!=4:
        raise ValueError(('Expected identical [B,T,H,D] readout arms',mha.shape,m_read.shape,total.shape))
    a,b,y=(v.astype(jnp.float32) for v in (mha,m_read,total))
    return dict(mha_square=jnp.mean(a*a,axis=(-2,-1)),m_square=jnp.mean(b*b,axis=(-2,-1)),
                cross=jnp.mean(a*b,axis=(-2,-1)),sum_square=jnp.mean(y*y,axis=(-2,-1)),
                reconstruction_error=jnp.max(jnp.abs(total-(mha+m_read)),axis=(-2,-1)))

def install(include_readout=True):
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import linears
    original=linears.DenseGeneral.__call__
    @functools.wraps(original)
    def observed(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        if not ACTIVE or self.is_initializing():return value
        path=self.scope.path
        role=None
        if self.name=='wi' and len(path)>1 and path[-2]=='mlp':role='main'
        elif self.name=='wi' and len(path)>1 and path[-2]=='memory_mlp':role='memory'
        elif self.name=='mlp_address_down':role='address'
        elif self.name=='mlp_write_gate':role='write_logit'
        if role is None:return value
        measured=value
        if role=='write_logit':
            # The true per-head write logits add this bias outside DenseGeneral.
            bias=self.scope.parent.get_variable('params','mlp_write_gate_bias')
            if bias is None:raise ValueError('Missing restored MLP write bias')
            measured=value+jnp.asarray(nn.unbox(bias),value.dtype)
        for name,stat in statistics(measured).items():
            self.sow('intermediates','stream_ffn_'+role+'_'+name,stat)
        if role=='write_logit':
            self.sow('intermediates','stream_ffn_write_logit_position_sigmoid_mean',jnp.mean(jax.nn.sigmoid(measured.astype(jnp.float32)),axis=-1))
        return value
    linears.DenseGeneral.__call__=observed
    if not include_readout:return
    # Capture the two actual arms at their native pre-W_o addition. This does not
    # reconstruct either arm from weights or substitute a different attention kernel.
    from layers import attentions,bam_pallas
    native_call=attentions.BamAttention.__call__
    @functools.wraps(native_call)
    def attention(self,*args,**kwargs):
        if not ACTIVE or self.is_initializing():return native_call(self,*args,**kwargs)
        _ATTN_STACK.append(dict(module=self))
        try:return native_call(self,*args,**kwargs)
        finally:_ATTN_STACK.pop()
    attentions.BamAttention.__call__=attention
    native_read=bam_pallas.read
    @functools.wraps(native_read)
    def read(*args,**kwargs):
        result=native_read(*args,**kwargs)
        if ACTIVE and _ATTN_STACK:_ATTN_STACK[-1]['m']=result[3]
        return result
    bam_pallas.read=read
    native_splash=attentions._bam_splash_attention
    @functools.wraps(native_splash)
    def splash(*args,**kwargs):
        result=native_splash(*args,**kwargs)
        if ACTIVE and _ATTN_STACK:_ATTN_STACK[-1]['mha']=result
        return result
    attentions._bam_splash_attention=splash
    native_block=attentions.BamAttention._attention_block
    @functools.wraps(native_block)
    def block(self,*args,**kwargs):
        result=native_block(self,*args,**kwargs)
        if ACTIVE and _ATTN_STACK:_ATTN_STACK[-1]['mha']=result[0]
        return result
    attentions.BamAttention._attention_block=block
    native_amplitude=attentions.BamAttention._record_concat_amplitude
    @functools.wraps(native_amplitude)
    def amplitude(self,name,bam,standard):
        if ACTIVE and _ATTN_STACK and name=='local_o':
            # Current full-head local-o models pass their complete head here.
            _ATTN_STACK[-1].update(m=bam,mha=standard)
        return native_amplitude(self,name,bam,standard)
    attentions.BamAttention._record_concat_amplitude=amplitude
    native_projection=attentions.BamAttention.out_projection
    @functools.wraps(native_projection)
    def projection(self,output_dim,out):
        if ACTIVE and _ATTN_STACK:
            ctx=_ATTN_STACK[-1]
            if ctx['module'] is not self or 'mha' not in ctx or 'm' not in ctx:
                raise ValueError('Missing actual native pre-projection readout pair')
            for name,value in balance_statistics(ctx['mha'],ctx['m'],out).items():
                self.sow('intermediates','stream_readout_'+name,value)
            a,b,y=(v.astype(jnp.float32) for v in (ctx['mha'],ctx['m'],out))
            head=jnp.stack([jnp.mean(v,axis=(1,3)) for v in (a*a,b*b,a*b,y*y)],axis=-1)
            self.sow('intermediates','stream_readout_head_moments',head)
        return native_projection(self,output_dim,out)
    attentions.BamAttention.out_projection=projection

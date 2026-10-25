"""Read-only moments of actual vector-residual additions, with terminal biases separate."""
import functools
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

ACTIVE=False
_INSTALLED=False
_STACK=[]
METRICS=('x_square','m_square','v_square','bias_square','y_square','mv_cross','xm_cross','xv_cross')
def positions(length):return np.arange(length,dtype=np.int32)

def observe_dense(module,value):
    """Called at the native dot output, immediately BEFORE the native bias add."""
    if not ACTIVE or not _STACK or module.is_initializing():return
    path=module.scope.path
    name=None
    if module.name=='out' and len(path)>1 and path[-2]=='self_attention':name='m'
    elif module.name=='wo' and len(path)>1 and path[-2]=='mlp':name='v'
    if name is None:return
    if module.bias_norm:raise ValueError('Nonlinear output-bias normalization is outside the source decomposition')
    if value.ndim!=3:raise ValueError(('Expected projected [B,T,D]',path,value.shape))
    ctx=_STACK[-1]
    if name+'_core' in ctx:raise ValueError(('Duplicate source capture',path))
    ctx[name+'_core']=value
    if module.use_bias:
        bias=module.get_variable('params','bias')
        if bias is None:raise ValueError(('Missing restored terminal bias',path))
        bias=jnp.asarray(nn.unbox(bias),value.dtype)
    else:bias=jnp.zeros((value.shape[-1],),value.dtype)
    ctx[name+'_bias']=bias

def statistics(x,m,v,bm,bv,m_full,v_full,y):
    if not(x.shape==m.shape==v.shape==m_full.shape==v_full.shape==y.shape) or x.ndim!=3:
        raise ValueError('Source components must have matching [B,T,D] shapes')
    expected=v_full+(x+m_full)
    m_error=jnp.max(jnp.abs(m_full-(m+bm)),axis=-1)
    v_error=jnp.max(jnp.abs(v_full-(v+bv)),axis=-1)
    block_error=jnp.max(jnp.abs(y-expected),axis=-1)
    xf,mf,vf,yf,am,af=(a.astype(jnp.float32) for a in (x,m,v,y,m_full,v_full))
    bmf=jnp.broadcast_to(bm.astype(jnp.float32),x.shape);bvf=jnp.broadcast_to(bv.astype(jnp.float32),x.shape);bf=bmf+bvf
    def dot(a,b):return jnp.mean(a*b,axis=-1)
    return dict(x_square=dot(xf,xf),m_square=dot(mf,mf),v_square=dot(vf,vf),bias_square=dot(bf,bf),m_bias_square=dot(bmf,bmf),v_bias_square=dot(bvf,bvf),
                y_square=dot(yf,yf),m_full_square=dot(am,am),v_full_square=dot(af,af),
                mv_cross=dot(mf,vf),xm_cross=dot(xf,mf),xv_cross=dot(xf,vf),xb_cross=dot(xf,bf),mb_cross=dot(mf,bf),vb_cross=dot(vf,bf),
                update_core_square=dot(mf+vf,mf+vf),actual_increment_square=dot(yf-xf,yf-xf),
                carry_plus_vector_square=dot(xf+vf,xf+vf),
                full_m_v_cross=dot(am,af),
                m_terminal_reconstruction_error=m_error,v_terminal_reconstruction_error=v_error,block_reconstruction_error=block_error)

def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from layers import attentions,linears,fusion
    original_block=fusion.SubDecoderLayer.__call__
    @functools.wraps(original_block)
    def block(self,*args,**kwargs):
        if not ACTIVE or self.is_initializing():return original_block(self,*args,**kwargs)
        ctx={};_STACK.append(ctx)
        try:
            result=original_block(self,*args,**kwargs)
            required={'m_core','v_core','m_bias','v_bias','m_full','v_full'}
            if set(ctx)!=required:raise ValueError(('Missing source components',self.scope.path,set(ctx),required))
            for key,value in statistics(args[0],ctx['m_core'],ctx['v_core'],ctx['m_bias'],ctx['v_bias'],ctx['m_full'],ctx['v_full'],result[0]).items():
                self.sow('intermediates','stream_source_'+key,value)
            return result
        finally:_STACK.pop()
    fusion.SubDecoderLayer.__call__=block
    original_attention=attentions.BamAttention.__call__
    @functools.wraps(original_attention)
    def attention(self,*args,**kwargs):
        result=original_attention(self,*args,**kwargs)
        if ACTIVE and _STACK and not self.is_initializing():_STACK[-1]['m_full']=result[0]
        return result
    attentions.BamAttention.__call__=attention
    original_mlp=linears.MlpBlock.__call__
    @functools.wraps(original_mlp)
    def mlp(self,*args,**kwargs):
        result=original_mlp(self,*args,**kwargs)
        if ACTIVE and _STACK and not self.is_initializing() and self.name=='mlp':
            if isinstance(result,tuple):raise ValueError('Shared-memory tuple MLP is outside this probe')
            _STACK[-1]['v_full']=result
        return result
    linears.MlpBlock.__call__=mlp

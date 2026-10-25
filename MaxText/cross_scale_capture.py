"""Native head/FFN observations and controlled direct-link residual interventions.

Controls are additive scale offsets [3,layers]: attention->vector link,
FFN->vector link, current block's M write increment. Intervention is applied
at block exit, after that block's native FFN and M writes. Downstream layers
then run normally. These effects are not an additive decomposition of loss.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import head_attention_capture as head_hooks
import head_position_capture as head_stats

ACTIVE=False
TOKENS=None
CONTROLS=None
_INSTALLED=False
_STACK=[]
METRICS=('position_top_key_index','position_entropy','position_allowed_entropy')
def positions(length):return np.arange(length,dtype=np.int32)

def record_head(module,q,k,segments,window=None,soft_cap=0.):
 if not ACTIVE or module.is_initializing():return
 vals=head_stats.statistics(q,k,TOKENS,segments,window,soft_cap)
 for name,a in zip(METRICS,vals):module.sow('intermediates','stream_'+name,a)

def source_statistics(x,m,v,y):
 x,m,v,y=(a.astype(jnp.float32) for a in (x,m,v,y))
 dot=lambda a,b:jnp.mean(a*b,-1)
 return dict(x_square=dot(x,x),m_square=dot(m,m),v_square=dot(v,v),y_square=dot(y,y),mv_cross=dot(m,v),xm_cross=dot(x,m),xv_cross=dot(x,v))

def intervene(result,x,m,v,M_in,control):
 """Keep identity branch bitwise native. Controls perturb only block-exit links."""
 y,M_out=result
 c=control
 y=jax.lax.cond(jnp.any(c[:2]!=0),lambda z:(z.astype(jnp.float32)+c[0]*m.astype(jnp.float32)+c[1]*v.astype(jnp.float32)).astype(z.dtype),lambda z:z,y)
 if M_out is not None and M_in is not None:
  M_out=jax.lax.cond(c[2]!=0,lambda z:(z.astype(jnp.float32)+c[2]*(z.astype(jnp.float32)-M_in.astype(jnp.float32))).astype(z.dtype),lambda z:z,M_out)
 return y,M_out

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 head_hooks.install();head_hooks.record=record_head
 from layers import fusion,linears,attentions
 original=fusion.SubDecoderLayer.__call__
 @functools.wraps(original)
 def block(self,*args,**kwargs):
  if (not ACTIVE and CONTROLS is None) or self.is_initializing():return original(self,*args,**kwargs)
  incoming=args[0]
  if isinstance(incoming,(tuple,list)):
   if self.config.scan_use_mudd and len(incoming)==2:incoming=incoming[1]
   else:incoming=incoming[3]
  ctx={};_STACK.append(ctx)
  try:
   result=original(self,*args,**kwargs)
   if set(ctx)!={'m','v'}:raise ValueError(('Missing native projected updates',self.scope.path,list(ctx)))
   if ACTIVE:
    for name,a in source_statistics(incoming,ctx['m'],ctx['v'],result[0]).items():self.sow('intermediates','stream_source_'+name,a)
    expected=ctx['v']+(incoming+ctx['m'])
    self.sow('intermediates','stream_source_reconstruction_error',jnp.max(jnp.abs(result[0]-expected),axis=-1))
    Min=kwargs.get('M_in')
    if Min is not None and result[1] is not None:
     a,b=Min.astype(jnp.float32),result[1].astype(jnp.float32)
     self.sow('intermediates','stream_source_M_input_square',jnp.mean(a*a,axis=(-1,-2)))
     self.sow('intermediates','stream_source_M_write_square',jnp.mean((b-a)**2,axis=(-1,-2)))
   if CONTROLS is not None:
    layer=kwargs.get('layer_index')
    if layer is None:layer=self.layer_inx
    result=intervene(result,incoming,ctx['m'],ctx['v'],kwargs.get('M_in'),CONTROLS[:,layer])
   return result
  finally:_STACK.pop()
 fusion.SubDecoderLayer.__call__=block
 for klass,is_bam in [(attentions.Attention,False),(attentions.BamAttention,True)]:
  native=klass.__call__
  def wrap(native,is_bam):
   @functools.wraps(native)
   def attention(self,*args,**kwargs):
    value=native(self,*args,**kwargs)
    if _STACK and not self.is_initializing():_STACK[-1]['m']=value[0] if is_bam else value
    return value
   return attention
  klass.__call__=wrap(native,is_bam)
 native_mlp=linears.MlpBlock.__call__
 @functools.wraps(native_mlp)
 def mlp(self,*args,**kwargs):
  value=native_mlp(self,*args,**kwargs)
  if _STACK and not self.is_initializing() and self.name=='mlp':
   if isinstance(value,tuple):raise ValueError('Tuple-return MLP needs a dedicated adapter')
   _STACK[-1]['v']=value
  return value
 linears.MlpBlock.__call__=mlp
 native_dense=linears.DenseGeneral.__call__
 @functools.wraps(native_dense)
 def dense(self,*args,**kwargs):
  value=native_dense(self,*args,**kwargs)
  if not ACTIVE or self.is_initializing():return value
  path=self.scope.path;name=None;measured=value
  if len(path)>1 and path[-2]=='mlp' and self.name in ('wi','wi_0'):
   name='main'
   if measured.ndim==4:measured=measured[:,:,0,:]
  elif self.name=='mlp_address_down':name='address'
  if len(path)>1 and path[-2]=='mlp' and self.name=='wo':
   hidden=args[0].astype(jnp.float32)
   square=jnp.mean(hidden**2,-1);fourth=jnp.mean(hidden**4,-1)
   self.sow('intermediates','stream_ffn_hidden_square',square)
   self.sow('intermediates','stream_ffn_hidden_energy_participation',square**2/jnp.maximum(fourth,1e-30))
   self.sow('intermediates','stream_ffn_hidden_positive_ratio',jnp.mean(hidden>0,-1))
  if name:
   if measured.ndim!=3:raise ValueError((path,measured.shape))
   self.sow('intermediates','stream_ffn_'+name+'_positive_ratio',jnp.mean(measured>0,-1))
   self.sow('intermediates','stream_ffn_'+name+'_square',jnp.mean(measured.astype(jnp.float32)**2,-1))
  return value
 linears.DenseGeneral.__call__=dense

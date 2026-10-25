"""Zero-delta parameter-path gradients for the shared Q/K/V/O memory reader.

DELTA is an explicit function input, never a Flax parameter or optimizer state.
At DELTA=0, the learned forward is native. No stop_gradient cuts downstream paths.
"""
import functools
import jax.numpy as jnp
from flax import linen as nn
DELTA=None
_INSTALLED=False
_CONTEXT=[]
_VO_ARM=None

def _context():
 if not _CONTEXT:raise ValueError('Memory read outside attention context')
 c=_CONTEXT[-1]
 if 'pre' not in c or 'read_x' not in c:raise ValueError('Native shared reader hidden input not captured')
 return c

def _extra(self,name,arm):
 c=_context();layer=c['layer'];pre=c['pre'];x=c['read_x'];h=nn.gelu(pre)
 down=DELTA['down'][layer,arm].astype(x.dtype)
 dp=jnp.einsum('btd,dr->btr',x,down,precision=self.config.matmul_precision)
 changed=nn.gelu(pre+dp.astype(pre.dtype))
 extra=getattr(self,name+'_up')(changed-h)
 if arm in (2,3):
  up=DELTA['up'][layer,arm-2].astype(changed.dtype)
  extra=extra+jnp.einsum('btr,rhcv->bthcv',changed,up,precision=self.config.matmul_precision)
 return extra

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import attentions,linears,bam_pallas
 native_attention=attentions.BamAttention.__call__
 @functools.wraps(native_attention)
 def attention(self,*args,**kwargs):
  if DELTA is None or self.is_initializing():return native_attention(self,*args,**kwargs)
  layer=kwargs.get('layer_index')
  if layer is None:layer=self.layer_inx
  _CONTEXT.append(dict(self=self,layer=layer))
  try:return native_attention(self,*args,**kwargs)
  finally:_CONTEXT.pop()
 attentions.BamAttention.__call__=attention
 native_dense=linears.DenseGeneral.__call__
 @functools.wraps(native_dense)
 def dense(self,*args,**kwargs):
  value=native_dense(self,*args,**kwargs)
  if DELTA is not None and _CONTEXT and self.name=='full_m_read_down' and not self.is_initializing():
   _CONTEXT[-1]['pre']=value;_CONTEXT[-1]['read_x']=args[0]
  return value
 linears.DenseGeneral.__call__=dense
 native_fused=attentions.BamAttention._fused_input_projections
 @functools.wraps(native_fused)
 def fused(self,x,*args,**kwargs):
  raw=native_fused(self,x,*args,**kwargs)
  if DELTA is not None and _CONTEXT and 'full_m_read_down' in raw:
   _CONTEXT[-1]['pre']=raw['full_m_read_down'];_CONTEXT[-1]['read_x']=x
  return raw
 attentions.BamAttention._fused_input_projections=fused
 native_key=attentions.BamAttention._project_column_read_key
 @functools.wraps(native_key)
 def key(self,name,*args,**kwargs):
  value=native_key(self,name,*args,**kwargs)
  if DELTA is None or self.is_initializing():return value
  if name in ['W_lq_c8','W_lk_c8']:
   arm=0 if name=='W_lq_c8' else 1
  elif name=='W_R':
   if _VO_ARM is None:raise ValueError('VO gradient path not specified')
   arm=_VO_ARM
  else:return value
  return value+_extra(self,name,arm).astype(value.dtype)
 attentions.BamAttention._project_column_read_key=key
 native_vo=attentions.BamAttention._independent_local_vo
 @functools.wraps(native_vo)
 def vo(self,*args,**kwargs):
  global _VO_ARM
  if DELTA is None or self.is_initializing():return native_vo(self,*args,**kwargs)
  previous=_VO_ARM
  try:
   _VO_ARM=2;v,_=native_vo(self,*args,**kwargs)
   _VO_ARM=3;_,o=native_vo(self,*args,**kwargs)
   return v,o
  finally:_VO_ARM=previous
 attentions.BamAttention._independent_local_vo=vo
 native_read=bam_pallas.read
 @functools.wraps(native_read)
 def read(*args,**kwargs):
  if DELTA is None:return native_read(*args,**kwargs)
  self=_context()['self'];qk=list(args)
  qk[2]=qk[2]+_extra(self,'W_lq_c8',0).astype(qk[2].dtype)
  qk[4]=qk[4]+_extra(self,'W_lk_c8',1).astype(qk[4].dtype)
  original=native_read(*qk,**kwargs)
  va=list(args);oa=list(args)
  va[6]=va[6]+jnp.squeeze(_extra(self,'W_R',2),axis=-2).astype(va[6].dtype)
  oa[6]=oa[6]+jnp.squeeze(_extra(self,'W_R',3),axis=-2).astype(oa[6].dtype)
  v=native_read(*va,**kwargs)[2];o=native_read(*oa,**kwargs)[3]
  return original[:2]+(v,o)+original[4:]
 bam_pallas.read=read

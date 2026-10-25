"""Observe actual native MHA and direct-M readouts, and direct O's two read sources."""
import functools
import jax.numpy as jnp
ACTIVE=False
_INSTALLED=False
_STACK=[]

def _moments(a,b,y,axes):
 a,b,y=(x.astype(jnp.float32) for x in (a,b,y));count=1
 for i in axes:count*=a.shape[i]
 sums=[jnp.sum(v,axis=axes) for v in (a*a,b*b,a*b,y*y)]
 return jnp.stack(sums+[jnp.full_like(sums[0],count)],axis=-1)

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import attentions,bam_pallas
 native_call=attentions.BamAttention.__call__
 @functools.wraps(native_call)
 def attention(self,*args,**kwargs):
  if not ACTIVE or self.is_initializing():return native_call(self,*args,**kwargs)
  _STACK.append(dict(module=self))
  try:return native_call(self,*args,**kwargs)
  finally:_STACK.pop()
 attentions.BamAttention.__call__=attention
 native_read=bam_pallas.read
 @functools.wraps(native_read)
 def read(*args,**kwargs):
  out=native_read(*args,**kwargs)
  if ACTIVE and _STACK:
   c=_STACK[-1];c['m']=out[3];H=out[3].shape[2]
   dy=list(args);dy[1]=dy[1].at[3*H:4*H].set(0)
   st=list(args);st[7]=jnp.full_like(st[7],-80)
   c['dynamic_o']=native_read(*dy,**kwargs)[3]
   c['static_o']=native_read(*st,**kwargs)[3]
  return out
 bam_pallas.read=read
 native_vo=attentions.BamAttention._independent_local_vo
 @functools.wraps(native_vo)
 def vo(self,*args,**kwargs):
  out=native_vo(self,*args,**kwargs)
  if ACTIVE and _STACK:_STACK[-1]['dynamic_o']=out[1]
  return out
 attentions.BamAttention._independent_local_vo=vo
 native_static=attentions.BamAttention._static_column
 @functools.wraps(native_static)
 def static(self,M,arm,*args,**kwargs):
  out=native_static(self,M,arm,*args,**kwargs)
  if ACTIVE and _STACK and arm=='o':_STACK[-1]['static_o']=out
  return out
 attentions.BamAttention._static_column=static
 native_splash=attentions._bam_splash_attention
 @functools.wraps(native_splash)
 def splash(*args,**kwargs):
  out=native_splash(*args,**kwargs)
  if ACTIVE and _STACK:_STACK[-1]['mha']=out
  return out
 attentions._bam_splash_attention=splash
 native_block=attentions.BamAttention._attention_block
 @functools.wraps(native_block)
 def block(self,*args,**kwargs):
  out=native_block(self,*args,**kwargs)
  if ACTIVE and _STACK:_STACK[-1]['mha']=out[0]
  return out
 attentions.BamAttention._attention_block=block
 native_amplitude=attentions.BamAttention._record_concat_amplitude
 @functools.wraps(native_amplitude)
 def amplitude(self,name,bam,standard):
  if ACTIVE and _STACK and name=='local_o':_STACK[-1].update(m=bam,mha=standard)
  return native_amplitude(self,name,bam,standard)
 attentions.BamAttention._record_concat_amplitude=amplitude
 native_projection=attentions.BamAttention.out_projection
 @functools.wraps(native_projection)
 def projection(self,output_dim,out):
  if ACTIVE and _STACK:
   c=_STACK[-1]
   if c['module'] is not self or any(k not in c for k in ['mha','m','dynamic_o','static_o']):raise ValueError(('Missing native readout pair',self.scope.path,list(c)))
   a,b=c['mha'],c['m'];dy,st=c['dynamic_o'],c['static_o']
   if a.shape!=b.shape or a.shape!=out.shape:raise ValueError(('Readout shape mismatch',a.shape,b.shape,out.shape))
   self.sow('intermediates','stream_prewo_pair',_moments(a,b,out,(1,2,3)))
   self.sow('intermediates','stream_prewo_heads',_moments(a,b,out,(1,3)))
   self.sow('intermediates','stream_prewo_O_sources',_moments(dy,st,b,(1,2,3)))
   self.sow('intermediates','stream_prewo_O_closure',jnp.sum((b.astype(jnp.float32)-dy.astype(jnp.float32)-st.astype(jnp.float32))**2,axis=(1,2,3)))
   self.sow('intermediates','stream_prewo_reconstruction',jnp.max(jnp.abs(out-(a+b)),axis=(1,2,3)))
  return native_projection(self,output_dim,out)
 attentions.BamAttention.out_projection=projection

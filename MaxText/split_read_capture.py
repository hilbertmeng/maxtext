"""Capture native sigmoid V/O read gates without changing learned computations."""
import functools
import jax,jax.numpy as jnp
ACTIVE=False
_INSTALLED=False
_CONTEXT=[]
def stats(logits):
 p=jax.nn.sigmoid(logits.astype(jnp.float32));axes=tuple(range(1,p.ndim))
 return jnp.stack([p.mean(axes),(p<.05).mean(axes),(p<.01).mean(axes),(p>.95).mean(axes)],axis=-1)
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import attentions,bam_pallas
 native=attentions.BamAttention.__call__
 @functools.wraps(native)
 def attention(self,*args,**kwargs):
  if not ACTIVE or self.is_initializing():return native(self,*args,**kwargs)
  _CONTEXT.append(self)
  try:return native(self,*args,**kwargs)
  finally:_CONTEXT.pop()
 attentions.BamAttention.__call__=attention
 native_read=bam_pallas.read
 @functools.wraps(native_read)
 def read(*args,**kwargs):
  if ACTIVE and _CONTEXT:
   self=_CONTEXT[-1]
   self.sow('intermediates','stream_split_V_gate',stats(args[8]))
   self.sow('intermediates','stream_split_O_gate',stats(args[7]))
  return native_read(*args,**kwargs)
 bam_pallas.read=read
 native_gate=attentions.BamAttention._project_read_gate_logits
 @functools.wraps(native_gate)
 def gate(self,gate_name,*args,**kwargs):
  value=native_gate(self,gate_name,*args,**kwargs)
  if ACTIVE and not self.is_initializing() and gate_name in ['W_lv_gate','W_R_gate']:
   name='V' if gate_name=='W_lv_gate' else 'O'
   self.sow('intermediates','stream_split_'+name+'_gate',stats(value))
  return value
 attentions.BamAttention._project_read_gate_logits=gate

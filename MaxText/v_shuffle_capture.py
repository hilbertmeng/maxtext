"""Analysis-only live cross-text V retrieval; memory contents remain with their text."""
import functools
import jax.numpy as jnp
SHIFT=None
_INSTALLED=False

def pallas_v_read(native,args,kwargs,shift):
 out=native(*args,**kwargs)
 if shift is None:return out
 changed=list(args)
 changed[6]=jnp.roll(changed[6],shift,axis=0)  # shared dynamic address: keep only V below
 changed[8]=jnp.roll(changed[8],shift,axis=0)
 opts=dict(kwargs)
 gates=opts.get('static_gates')
 if gates is not None:
  gates=list(gates);gates[2]=jnp.roll(gates[2],shift,axis=0);opts['static_gates']=tuple(gates)
 foreign=native(*changed,**opts)
 return tuple(foreign[i] if i==2 else x for i,x in enumerate(out))

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import mudd,attentions
 original=mudd.Mlp.__call__
 @functools.wraps(original)
 def router(self,*args,**kwargs):
  w=original(self,*args,**kwargs)
  if SHIFT is None or self.is_initializing() or w.shape[-2]==1:return w
  if w.shape[-2]!=4:raise ValueError(('Expected Q/K/V/residual channels',w.shape))
  return w.at[:,:,2,:].set(jnp.roll(w[:,:,2,:],SHIFT,axis=0))
 mudd.Mlp.__call__=router
 try:
  from layers import bam_pallas
 except ImportError:
  # Pre-Pallas XL Mudd only needs its signed depth-router hook.
  return
 native=bam_pallas.read
 @functools.wraps(native)
 def read(*args,**kwargs):return pallas_v_read(native,args,kwargs,SHIFT)
 bam_pallas.read=read
 native_vo=attentions.BamAttention._independent_local_vo
 @functools.wraps(native_vo)
 def vo(self,M,x,compressed_M=None,shared_hidden=None):
  v,o=native_vo(self,M,x,compressed_M=compressed_M,shared_hidden=shared_hidden)
  if SHIFT is None or self.is_initializing():return v,o
  sx=jnp.roll(x,SHIFT,axis=0);sh=None if shared_hidden is None else jnp.roll(shared_hidden,SHIFT,axis=0)
  foreign,_=native_vo(self,M,sx,compressed_M=compressed_M,shared_hidden=sh)
  return foreign,o
 attentions.BamAttention._independent_local_vo=vo
 native_static=attentions.BamAttention._static_column
 @functools.wraps(native_static)
 def static(self,M,arm,x=None):
  if SHIFT is not None and not self.is_initializing() and arm=='v' and x is not None:x=jnp.roll(x,SHIFT,axis=0)
  return native_static(self,M,arm,x)
 attentions.BamAttention._static_column=static

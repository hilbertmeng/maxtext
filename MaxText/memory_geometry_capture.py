"""Read-only, sampled native M carries in canonical [B,P,K,V] coordinates."""
import functools
import numpy as np
import jax.numpy as jnp
ACTIVE=False
_INSTALLED=False
def positions(T):return np.unique(np.r_[0,1,np.linspace(min(64,T-1),T-1,16,dtype=np.int32)]).astype(np.int32)
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import fusion
 native=fusion.SubDecoderLayer.__call__
 @functools.wraps(native)
 def block(self,*args,**kwargs):
  result=native(self,*args,**kwargs)
  if not ACTIVE or self.is_initializing() or not self.config.bam_enabled:return result
  for name,a in [('M_in',kwargs.get('M_in')),('M_out',result[1])]:
   if a is None:raise ValueError(('Missing native M',name,self.scope.path))
   if getattr(self.config,'bam_pallas_core',False):
    body=getattr(self.config,'bam_pallas_body',None)
    if body=='v8t':a=jnp.swapaxes(a,-1,-2)
    elif body in ('v7','v7u'):a=jnp.transpose(a,(0,3,1,2))
    else:a=jnp.transpose(a,(0,3,2,1))
   if a.shape[-2:]!=(self.config.bam_k,self.config.bam_v):raise ValueError(('M layout mismatch',a.shape))
   self.sow('intermediates','stream_geometry_'+name,a[:,positions(a.shape[1])])
  return result
 fusion.SubDecoderLayer.__call__=block

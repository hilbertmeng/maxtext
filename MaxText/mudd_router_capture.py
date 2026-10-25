"""Native signed depth routers; replay removes input conditioning without LM training."""
import functools
import numpy as np
import jax.numpy as jnp
ACTIVE=False
VALUES=None
MASK=None
_INSTALLED=False
_CONTEXT=[]
def positions(T):return np.unique(np.r_[0,1,np.linspace(min(64,T-1),T-1,16,dtype=np.int32)]).astype(np.int32)
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import mudd
 native_compose=mudd.Compose.__call__
 @functools.wraps(native_compose)
 def compose(self,*args,**kwargs):
  if not ACTIVE or self.is_initializing():return native_compose(self,*args,**kwargs)
  hids=kwargs.get('hids',args[1] if len(args)>1 else None)
  if hids is None:raise ValueError('Missing native Mudd history')
  _CONTEXT.append(hids)
  try:return native_compose(self,*args,**kwargs)
  finally:_CONTEXT.pop()
 mudd.Compose.__call__=compose
 native_mlp=mudd.Mlp.__call__
 @functools.wraps(native_mlp)
 def mlp(self,*args,**kwargs):
  w=native_mlp(self,*args,**kwargs)
  if self.is_initializing():return w
  if ACTIVE:
   if not _CONTEXT:raise ValueError(('Router outside Compose',self.scope.path))
   hids=_CONTEXT[-1];pos=positions(w.shape[1])
   norm=jnp.stack([jnp.sqrt(jnp.mean(x[:,pos].astype(jnp.float32)**2,-1)) for x in hids],axis=-1)
   self.sow('intermediates','stream_router_weights',w)
   self.sow('intermediates','stream_router_history_rms',norm)
  if MASK is not None:
   path='/'.join(self.scope.path)
   if VALUES is None or path not in VALUES:raise ValueError(('Missing replay router',path))
   cached=jnp.asarray(VALUES[path],w.dtype)
   if cached.shape!=w.shape:raise ValueError(('Replay shape mismatch',path,cached.shape,w.shape))
   channels=MASK[4:5] if w.shape[-2]==1 else MASK[:4]
   if channels.shape[0]!=w.shape[-2]:raise ValueError(('Unexpected router channels',w.shape))
   return jnp.where(channels[None,None,:,None],cached,w)
  return w
 mudd.Mlp.__call__=mlp

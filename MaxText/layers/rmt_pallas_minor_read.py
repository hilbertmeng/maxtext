"""Token-contiguous C8 key normalization, contraction and destination gates."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _norm, _spec


def reference(m,k,g,epsilon=1e-6):
  f=k.astype(jnp.float32)
  key=(f*jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+epsilon)).astype(k.dtype)
  read=jnp.einsum('rv,hr->hv',m,key)
  return (.2*g)[...,None]*read[:,None,:]


def _tile(m,k,g,epsilon):
  r,v,t=m.shape
  h=k.shape[0]
  key=_norm(k,epsilon).astype(jnp.float32)
  read=jnp.zeros((h,v,t),jnp.float32)
  for i in range(r):
    ki=jax.lax.slice_in_dim(key,i,i+1,axis=1).reshape(h,t)
    mi=jax.lax.slice_in_dim(m,i,i+1,axis=0).reshape(v,t).astype(jnp.float32)
    read=read+ki[:,None,:]*mi[None,:,:]
  read=read.astype(m.dtype)
  return ((.2*g).transpose(1,0,2)[:,:,None,:]*read[None,:,:,:]).astype(m.dtype)


def _call(m,k,g,epsilon,interpret,tile):
  b,r,v,t=m.shape;h,d=k.shape[1],g.shape[2]
  def kernel(m,k,g,y):y[...]=_tile(m[...],k[...],g[...],epsilon)
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[_spec((r,v),tile),_spec((h,r),tile),_spec((h,d),tile)],
      out_specs=_spec((d,h,v),tile),out_shape=jax.ShapeDtypeStruct((b,d,h,v,t),m.dtype),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_c8')(m,k,g)


@partial(jax.custom_vjp,nondiff_argnums=(3,4,5))
def _read(m,k,g,epsilon,interpret,tile):return _call(m,k,g,epsilon,interpret,tile)


def _fwd(m,k,g,epsilon,interpret,tile):
  return _call(m,k,g,epsilon,interpret,tile),(m,k,g)


def _bwd(epsilon,interpret,tile,args,dy):
  m,k,g=args;b,r,v,t=m.shape;h,d=k.shape[1],g.shape[2]
  def kernel(m,k,g,dy,dm,dk,dg):
    _,pb=jax.vjp(lambda mm,kk,gg:_tile(mm,kk,gg,epsilon),m[...],k[...],g[...])
    dm[...],dk[...],dg[...]=pb(dy[...])
  specs=[_spec((r,v),tile),_spec((h,r),tile),_spec((h,d),tile)]
  return tuple(pl.pallas_call(kernel,grid=(b,t//tile),in_specs=specs+[_spec((d,h,v),tile)],
      out_specs=specs,out_shape=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in args],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_c8_backward')(*args,dy))


_read.defvjp(_fwd,_bwd)


def c8_read(matrix,key,gates,epsilon=1e-6,*,interpret=False,tile=128):
  # Public M[B,T,R,V], key[B,T,H,R], gates[B,T,H,D] -> [B,T,H,D,V].
  unbatched=matrix.ndim==3
  if unbatched:matrix,key,gates=(x[None] for x in (matrix,key,gates))
  def local(m,k,g):
    n=min(tile,m.shape[1])
    if m.shape[1]%n:raise ValueError('Token tile must divide sequence length')
    y=_read(m.transpose(0,2,3,1),k.transpose(0,2,3,1),g.transpose(0,2,3,1),epsilon,interpret,n)
    return y.transpose(0,4,2,1,3)
  y=_map_batch(local,(matrix,key,gates),(True,True,True))
  return y[0] if unbatched else y

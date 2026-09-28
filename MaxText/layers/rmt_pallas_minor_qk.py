"""Keep the full matrix contraction in XLA; fuse rank-four QK postprocessing."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _spec


def reference(read,basis,mix,gate,epsilon=1e-6):
  f=basis.astype(jnp.float32);w=mix.astype(jnp.float32)
  gram=jnp.einsum('rc,sc->rs',f,f)
  norm2=jnp.einsum('hr,rs,hs->h',w,gram,w)
  inv=jax.lax.rsqrt(norm2/basis.shape[-1]+epsilon)
  value=jnp.einsum('rv,hr->hv',read,mix)
  return value*(inv.astype(value.dtype)*(.2*gate))[:,None]


def _tile(read,basis,mix,gate,epsilon):
  r,v,t=read.shape;h=mix.shape[0];c=basis.shape[1]
  key=jnp.zeros((h,c,t),jnp.float32);value=jnp.zeros((h,v,t),jnp.float32)
  for i in range(r):
    w=jax.lax.slice_in_dim(mix,i,i+1,axis=1).reshape(h,t).astype(jnp.float32)
    b=jax.lax.slice_in_dim(basis,i,i+1,axis=0).reshape(c,t).astype(jnp.float32)
    a=jax.lax.slice_in_dim(read,i,i+1,axis=0).reshape(v,t).astype(jnp.float32)
    key=key+w[:,None,:]*b[None,:,:]
    value=value+w[:,None,:]*a[None,:,:]
  inv=jax.lax.rsqrt(jnp.mean(key*key,axis=1)+epsilon).astype(read.dtype)
  return value.astype(read.dtype)*(inv*(.2*gate))[:,None,:]


def _call(a,b,w,g,epsilon,interpret,tile):
  batch,r,v,t=a.shape;c=b.shape[2];h=w.shape[1]
  def kernel(a,b,w,g,out):out[...]=_tile(a[...],b[...],w[...],g[...],epsilon)
  return pl.pallas_call(kernel,grid=(batch,t//tile),
      in_specs=[_spec((r,v),tile),_spec((r,c),tile),_spec((h,r),tile),_spec((h,),tile)],
      out_specs=_spec((h,v),tile),out_shape=jax.ShapeDtypeStruct((batch,h,v,t),a.dtype),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_qk_post')(a,b,w,g)


@partial(jax.custom_vjp,nondiff_argnums=(4,5,6))
def _read(a,b,w,g,epsilon,interpret,tile):return _call(a,b,w,g,epsilon,interpret,tile)


def _fwd(a,b,w,g,epsilon,interpret,tile):
  return _call(a,b,w,g,epsilon,interpret,tile),(a,b,w,g)


def _bwd(epsilon,interpret,tile,args,dy):
  a,b,w,g=args;batch,r,v,t=a.shape;c=b.shape[2];h=w.shape[1]
  def kernel(a,b,w,g,dy,da,db,dw,dg):
    _,pb=jax.vjp(lambda aa,bb,ww,gg:_tile(aa,bb,ww,gg,epsilon),a[...],b[...],w[...],g[...])
    da[...],db[...],dw[...],dg[...]=pb(dy[...])
  specs=[_spec((r,v),tile),_spec((r,c),tile),_spec((h,r),tile),_spec((h,),tile)]
  return tuple(pl.pallas_call(kernel,grid=(batch,t//tile),in_specs=specs+[_spec((h,v),tile)],
      out_specs=specs,out_shape=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in args],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_qk_post_backward')(*args,dy))


_read.defvjp(_fwd,_bwd)


def qk_post(read,basis,mix,gate,epsilon=1e-6,*,interpret=False,tile=128):
  unbatched=read.ndim==3
  if unbatched:read,basis,mix,gate=(x[None] for x in (read,basis,mix,gate))
  def local(a,b,w,g):
    n=min(tile,a.shape[1])
    if a.shape[1]%n:raise ValueError('Token tile must divide sequence length')
    out=_read(a.transpose(0,2,3,1),b.transpose(0,2,3,1),w.transpose(0,2,3,1),g.transpose(0,2,1),epsilon,interpret,n)
    return out.transpose(0,3,1,2)
  out=_map_batch(local,(read,basis,mix,gate),(True,True,True,True))
  return out[0] if unbatched else out

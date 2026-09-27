"""Fuse the rank-four Q/K matrix read, Gram RMS and head expansion."""
from functools import partial
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch, _tile_size


def qk_reference(matrix,basis,mix,gates,epsilon=1e-6):
  basis_read=jnp.einsum('cv,rc->rv',matrix,basis)
  b=basis.astype(jnp.float32)
  gram=jnp.einsum('rc,sc->rs',b,b)
  f=mix.astype(jnp.float32)
  norm2=jnp.einsum('hr,rs,hs->h',f,gram,f)
  inverse=jax.lax.rsqrt(norm2/matrix.shape[-2]+epsilon)
  read=jnp.einsum('rv,hr->hv',basis_read,mix)
  return read*(inverse.astype(read.dtype)*(.2*gates))[:,None]


def _tile(matrix,basis,mix,gates,epsilon):
  basis_read=jnp.einsum('tcv,trc->trv',matrix,basis,
                         preferred_element_type=jnp.float32).astype(matrix.dtype)
  b=basis.astype(jnp.float32)
  gram=jnp.einsum('trc,tsc->trs',b,b,preferred_element_type=jnp.float32)
  f=mix.astype(jnp.float32)
  temp=jnp.einsum('thr,trs->ths',f,gram,preferred_element_type=jnp.float32)
  norm2=jnp.sum(temp*f,axis=-1)
  inverse=jax.lax.rsqrt(norm2/matrix.shape[-2]+epsilon)
  read=jnp.einsum('trv,thr->thv',basis_read,mix,
                  preferred_element_type=jnp.float32).astype(matrix.dtype)
  scale=(inverse.astype(read.dtype)*(.2*gates)).astype(read.dtype)
  return (read.astype(jnp.float32)*scale.astype(jnp.float32)[...,None]).astype(read.dtype)


def _spec(shape,tile):
  return pl.BlockSpec((tile,)+shape,lambda i:(i,)+(0,)*len(shape))


def _call(args,epsilon,interpret,tile):
  m,b,h,g=args
  def kernel(m,b,h,g,out):out[...]=_tile(m[...],b[...],h[...],g[...],epsilon)
  shape=(m.shape[0],h.shape[1],m.shape[2])
  return pl.pallas_call(kernel,grid=(m.shape[0]//tile,),
      in_specs=[_spec(x.shape[1:],tile) for x in args],out_specs=_spec(shape[1:],tile),
      out_shape=jax.ShapeDtypeStruct(shape,m.dtype),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_rank4_qk')(*args)


@partial(jax.custom_vjp,nondiff_argnums=(4,5,6))
def _read(m,b,h,g,epsilon,interpret,tile):return _call((m,b,h,g),epsilon,interpret,tile)


def _fwd(m,b,h,g,epsilon,interpret,tile):
  args=(m,b,h,g)
  return _call(args,epsilon,interpret,tile),args


def _bwd(epsilon,interpret,tile,args,dy):
  def kernel(m,b,h,g,dy,dm,db,dh,dg):
    _,pb=jax.vjp(lambda *x:_tile(*x,epsilon),m[...],b[...],h[...],g[...])
    dm[...],db[...],dh[...],dg[...]=pb(dy[...])
  return tuple(pl.pallas_call(kernel,grid=(args[0].shape[0]//tile,),
      in_specs=[_spec(x.shape[1:],tile) for x in (*args,dy)],
      out_specs=[_spec(x.shape[1:],tile) for x in args],
      out_shape=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in args],interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_rank4_qk_backward')(*args,dy))


_read.defvjp(_fwd,_bwd)


def qk_read(matrix,basis,mix,gates,epsilon=1e-6,*,interpret=False,tile=32):
  def local(m,b,h,g):
    n=math.prod(m.shape[:-2])
    y=_read(m.reshape((n,)+m.shape[-2:]),b.reshape((n,)+b.shape[-2:]),
            h.reshape((n,)+h.shape[-2:]),g.reshape(n,g.shape[-1]),epsilon,interpret,_tile_size(n,tile))
    return y.reshape(m.shape[:-2]+y.shape[1:])
  return _map_batch(local,(matrix,basis,mix,gates),(True,True,True,True))

"""Experimental V-only C8 read: move its sole output gate onto the read key.

Real-arithmetic equivalent to the original single-destination read. BF16 cast
locations change, so this is a separate numerical/performance experiment.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _norm, _norm_backward, _spec


def _parts(k,g,epsilon):
  key=_norm(k,epsilon)
  scale=(.2*g).astype(g.dtype)
  return key,scale,(key*scale[:,None,:]).astype(k.dtype)


def _forward(m,k,g,epsilon):
  key,scale,gated=_parts(k,g,epsilon)
  r,v,t=m.shape;h=k.shape[0]
  out=jnp.zeros((h,v,t),jnp.float32)
  for i in range(r):
    ki=jax.lax.slice_in_dim(gated,i,i+1,axis=1).reshape(h,t).astype(jnp.float32)
    mi=jax.lax.slice_in_dim(m,i,i+1,axis=0).reshape(v,t).astype(jnp.float32)
    out=out+ki[:,None,:]*mi[None,:,:]
  return out.astype(m.dtype)


def _reverse(m,k,g,dy,epsilon):
  key,scale,gated=_parts(k,g,epsilon)
  r,v,t=m.shape;h=k.shape[0]
  dm=[];du=[]
  for i in range(r):
    ki=jax.lax.slice_in_dim(gated,i,i+1,axis=1).reshape(h,t).astype(jnp.float32)
    mi=jax.lax.slice_in_dim(m,i,i+1,axis=0).reshape(v,t).astype(jnp.float32)
    dm.append(jnp.sum(dy.astype(jnp.float32)*ki[:,None,:],axis=0).astype(m.dtype))
    du.append(jnp.sum(dy.astype(jnp.float32)*mi[None,:,:],axis=1).astype(k.dtype))
  du=jnp.stack(du,axis=1)
  dk=_norm_backward(k,(du*scale[:,None,:]).astype(k.dtype),epsilon)
  dg=(.2*jax.lax.reduce_sum((du*key).astype(g.dtype),axes=(1,))).astype(g.dtype)
  return jnp.stack(dm),dk,dg


def _call(m,k,g,epsilon,interpret,tile):
  b,r,v,t=m.shape;h=k.shape[1]
  def kernel(m,k,g,out):out[...]=_forward(m[...],k[...],g[...],epsilon)
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[_spec((r,v),tile),_spec((h,r),tile),_spec((h,),tile)],
      out_specs=_spec((h,v),tile),out_shape=jax.ShapeDtypeStruct((b,h,v,t),m.dtype),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_v_only_read')(m,k,g)


@partial(jax.custom_vjp,nondiff_argnums=(3,4,5))
def _read(m,k,g,epsilon,interpret,tile):return _call(m,k,g,epsilon,interpret,tile)


def _fwd(m,k,g,epsilon,interpret,tile):
  return _call(m,k,g,epsilon,interpret,tile),(m,k,g)


def _bwd(epsilon,interpret,tile,args,dy):
  m,k,g=args;b,r,v,t=m.shape;h=k.shape[1]
  def kernel(m,k,g,dy,dm,dk,dg):
    dm[...],dk[...],dg[...]=_reverse(m[...],k[...],g[...],dy[...],epsilon)
  specs=[_spec((r,v),tile),_spec((h,r),tile),_spec((h,),tile)]
  return tuple(pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=specs+[_spec((h,v),tile)],out_specs=specs,
      out_shape=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in args],interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_v_only_read_backward')(*args,dy))


_read.defvjp(_fwd,_bwd)


def v_read(matrix,key,gates,epsilon=1e-6,*,interpret=False,tile=128):
  if gates.shape[-1]!=1:raise ValueError('V-only kernel requires one destination')
  unbatched=matrix.ndim==3
  if unbatched:matrix,key,gates=(x[None] for x in (matrix,key,gates))
  def local(m,k,g):
    n=min(tile,m.shape[1])
    if m.shape[1]%n:raise ValueError('Token tile must divide sequence length')
    out=_read(m.transpose(0,2,3,1),k.transpose(0,2,3,1),g[...,0].transpose(0,2,1),epsilon,interpret,n)
    return out.transpose(0,3,1,2)[...,None,:]
  out=_map_batch(local,(matrix,key,gates),(True,True,True))
  return out[0] if unbatched else out

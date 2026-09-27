"""One full-M projection feeds both static reads and dynamic C8 reads."""
from functools import partial
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _rms, _tile_size, _block_diagonal, _map_batch


def joined_reference(matrix,key,projection,gates,epsilon=1e-6):
  split=projection.shape[-1]-key.shape[-1]
  all_reads=jnp.einsum('kv,kr->rv',matrix,projection)
  read=jnp.einsum('rv,hr->hv',all_reads[split:],_rms(key,epsilon))
  return all_reads[:split],(.2*gates)[...,None]*read[:,None,:]


def _projection_tile(matrix,projection):
  t,k,v=matrix.shape
  vp=((v+127)//128)*128
  padded=jnp.concatenate((matrix,jnp.zeros((t,k,vp-v),matrix.dtype)),axis=-1)
  projected=jnp.dot(padded.transpose(0,2,1).reshape(t*vp,k),projection,
                    preferred_element_type=jnp.float32).astype(matrix.dtype)
  return projected.reshape(t,vp,projection.shape[-1]).transpose(0,2,1)


def _dynamic_tile(compressed,key,gates,epsilon,v):
  t,r,vp=compressed.shape
  h=key.shape[-2]
  if os.environ.get('RMT_PALLAS_BATCHED_DOT')=='1':
    read=jnp.einsum('thr,trv->thv',_rms(key,epsilon),compressed,
                    preferred_element_type=jnp.float32).astype(compressed.dtype)[...,:v]
  else:
    blocked=_block_diagonal(_rms(key,epsilon))
    read=jnp.dot(blocked,compressed.reshape(t*r,vp),
                 preferred_element_type=jnp.float32).astype(compressed.dtype)
    read=read.reshape(t,h,vp)[...,:v]
  return ((.2*gates).astype(jnp.float32).transpose(0,2,1)[...,None]*
          read.astype(jnp.float32)[:,None,:,:]).astype(compressed.dtype)


def _joined_tile(matrix,key,projection,gates,epsilon,save=False):
  projected=_projection_tile(matrix,projection)
  split=projection.shape[-1]-key.shape[-1]
  compressed=projected[:,split:,:]
  dynamic=_dynamic_tile(compressed,key,gates,epsilon,matrix.shape[-1])
  outputs=(projected[:,:split,:matrix.shape[-1]],dynamic)
  return outputs+(compressed,) if save else outputs


def _spec(shape,tile,shared=False,partial_grad=False):
  zeros=(0,)*len(shape)
  if shared:return pl.BlockSpec(shape,lambda i:zeros)
  return pl.BlockSpec((None if partial_grad else tile,)+shape,lambda i:(i,)+zeros)


def _call(matrix,key,projection,gates,epsilon,interpret,tile,save=False):
  n,k,v=matrix.shape
  h,r=key.shape[-2:]
  dest=gates.shape[-1]
  static_heads=projection.shape[-1]-r
  def kernel(m,q,p,g,*outs):
    values=_joined_tile(m[...],q[...],p[...],g[...],epsilon,save)
    for ref,value in zip(outs,values):ref[...]=value
  shapes=((n,static_heads,v),(n,dest,h,v))
  if save:shapes+=((n,r,((v+127)//128)*128),)
  return pl.pallas_call(kernel,grid=(n//tile,),
      in_specs=[_spec((k,v),tile),_spec((h,r),tile),_spec(projection.shape,tile,shared=True),_spec((h,dest),tile)],
      out_specs=tuple(_spec(s[1:],tile) for s in shapes),
      out_shape=tuple(jax.ShapeDtypeStruct(s,matrix.dtype) for s in shapes),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_joined_static_c8')(matrix,key,projection,gates)


@partial(jax.custom_vjp,nondiff_argnums=(4,5,6))
def _joined(matrix,key,projection,gates,epsilon,interpret,tile):
  return _call(matrix,key,projection,gates,epsilon,interpret,tile)


def _fwd(matrix,key,projection,gates,epsilon,interpret,tile):
  s,d,compressed=_call(matrix,key,projection,gates,epsilon,interpret,tile,save=True)
  return (s,d),(matrix,key,projection,gates,compressed)


def _bwd(epsilon,interpret,tile,args,cotangents):
  n=args[0].shape[0]
  original=args[:4]
  def kernel(m,q,p,g,c,ds,dd,dm,dq,dp,dg):
    # Retain only the small C8 projection from forward. Recomputing the
    # full-M projection here repeats work already done by the layer remat.
    v=m.shape[-1]
    _,read_pullback=jax.vjp(lambda cc,qq,gg:_dynamic_tile(cc,qq,gg,epsilon,v),c[...],q[...],g[...])
    dc,qgrad,ggrad=read_pullback(dd[...])
    padded_ds=jnp.concatenate((ds[...],jnp.zeros(ds.shape[:-1]+(c.shape[-1]-v,),ds.dtype)),axis=-1)
    dprojected=jnp.concatenate((padded_ds,dc),axis=1)
    _,project_pullback=jax.vjp(_projection_tile,m[...],p[...])
    mgrad,pgrad=project_pullback(dprojected)
    dm[...],dq[...],dp[...],dg[...]=mgrad,qgrad,pgrad,ggrad
  shapes=[x.shape if i!=2 else (n//tile,)+x.shape for i,x in enumerate(original)]
  grads=pl.pallas_call(kernel,grid=(n//tile,),
      in_specs=[_spec(x.shape if i==2 else x.shape[1:],tile,shared=i==2) for i,x in enumerate(args)]+
               [_spec(x.shape[1:],tile) for x in cotangents],
      out_specs=[_spec(s[1:],tile,partial_grad=i==2) for i,s in enumerate(shapes)],
      out_shape=[jax.ShapeDtypeStruct(s,x.dtype) for s,x in zip(shapes,original)],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_joined_static_c8_backward')(*args,*cotangents)
  return grads[0],grads[1],jnp.sum(grads[2].astype(jnp.float32),axis=0).astype(args[2].dtype),grads[3]


_joined.defvjp(_fwd,_bwd)


def joined_read(matrix,key,projection,gates,epsilon=1e-6,*,interpret=False,tile=64):
  def local(m,q,p,g):
    n=math.prod(m.shape[:-2])
    s,d=_joined(m.reshape((n,)+m.shape[-2:]),q.reshape((n,)+q.shape[-2:]),p,
                g.reshape((n,)+g.shape[-2:]),epsilon,interpret,_tile_size(n,tile))
    return (s.reshape(m.shape[:-2]+s.shape[1:]),
            d.transpose(0,2,1,3).reshape(m.shape[:-2]+(q.shape[-2],g.shape[-1],m.shape[-1])))
  return _map_batch(local,(matrix,key,projection,gates),(True,True,False,True),output_tuple=True)

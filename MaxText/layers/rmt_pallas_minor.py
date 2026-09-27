"""Vectorize token lanes without forcing matrix-flow tensors to value-minor."""
from functools import partial
import os
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch


def _norm(x,epsilon):
  f=x.astype(jnp.float32)
  return (f*jax.lax.rsqrt(jnp.mean(f*f,axis=-2,keepdims=True)+epsilon)).astype(x.dtype)


def _tile(matrix,address,data,gate,static_key,epsilon):
  # Shapes K,V,T; H,K,T; H,V,T; H,T. T stays in SIMD lanes.
  h,k,t=address.shape
  v=data.shape[1]
  a=(_norm(address,epsilon)*gate.astype(jnp.float32)[:,None,:].astype(gate.dtype)).astype(jnp.float32)
  d=_norm(data,epsilon).astype(jnp.float32)
  vp=((v+127)//128)*128
  padded=jnp.concatenate((data,jnp.zeros((h,vp-v,t),data.dtype)),axis=1)
  static=jnp.dot(static_key.T,padded.reshape(h,vp*t),
                 preferred_element_type=jnp.float32).reshape(k,vp,t)[:,:v,:]
  dynamic=jnp.zeros_like(static)
  for head in range(h):
    av=jax.lax.slice_in_dim(a,head,head+1,axis=0).reshape(k,t)
    dv=jax.lax.slice_in_dim(d,head,head+1,axis=0).reshape(v,t)
    dynamic=dynamic+av[:,None,:]*dv[None,:,:]
  return matrix+static.astype(matrix.dtype)+dynamic.astype(matrix.dtype)


def _spec(shape,tile):
  return pl.BlockSpec((None,)+shape+(tile,),lambda b,i:(b,)+(0,)*len(shape)+(i,))


def _call(m,a,d,g,s,epsilon,interpret,tile):
  b,k,v,t=m.shape
  h=a.shape[1]
  def kernel(m,a,d,g,s,out):out[...]=_tile(m[...],a[...],d[...],g[...],s[...],epsilon)
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[_spec((k,v),tile),_spec((h,k),tile),_spec((h,v),tile),_spec((h,),tile),
                pl.BlockSpec(s.shape,lambda b,i:(0,0))],out_specs=_spec((k,v),tile),
      out_shape=jax.ShapeDtypeStruct(m.shape,m.dtype),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel'),
          allow_input_fusion=((True,)*5 if os.environ.get('RMT_PALLAS_INPUT_FUSION')=='1' else None)),
      name='rmt_token_minor_write')(m,a,d,g,s)


@partial(jax.custom_vjp,nondiff_argnums=(5,6,7))
def _write(m,a,d,g,s,epsilon,interpret,tile):return _call(m,a,d,g,s,epsilon,interpret,tile)


def _fwd(m,a,d,g,s,epsilon,interpret,tile):
  return _call(m,a,d,g,s,epsilon,interpret,tile),(a,d,g,s)


def _bwd(epsilon,interpret,tile,args,dy):
  a,d,g,s=args
  b,k,v,t=dy.shape;h=a.shape[1]
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    _,pb=jax.vjp(lambda aa,dd,gg,ss:_tile(jnp.zeros((k,v,tile),dy.dtype),aa,dd,gg,ss,epsilon),
                 a[...],d[...],g[...],s[...])
    da[...],dd[...],dg[...],ds[...]=pb(dy[...])
  shapes=[a.shape,d.shape,g.shape,(b,t//tile)+s.shape]
  specs=[_spec((h,k),tile),_spec((h,v),tile),_spec((h,),tile),pl.BlockSpec(s.shape,lambda b,i:(0,0)),_spec((k,v),tile)]
  da,dd,dg,ds=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=specs,
      out_specs=[*specs[:3],pl.BlockSpec((None,None)+s.shape,lambda b,i:(b,i,0,0))],
      out_shape=[jax.ShapeDtypeStruct(shape,a.dtype) for shape in shapes],interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel'),
          allow_input_fusion=((True,)*5 if os.environ.get('RMT_PALLAS_INPUT_FUSION')=='1' else None)),
      name='rmt_token_minor_write_backward')(a,d,g,s,dy)
  return dy,da,dd,dg,jnp.sum(ds.astype(jnp.float32),axis=(0,1)).astype(s.dtype)


_write.defvjp(_fwd,_bwd)


def write_residual(matrix,address,data,gate,static_key,epsilon=1e-6,*,interpret=False,tile=128):
  unbatched=matrix.ndim==3
  if unbatched:
    matrix,address,data,gate=(x[None] for x in (matrix,address,data,gate))
  def local(m,a,d,g,s):
    token_tile=min(tile,m.shape[1])
    if m.shape[1]%token_tile:raise ValueError('Token tile must divide sequence length')
    result=_write(m.transpose(0,2,3,1),a.transpose(0,2,3,1),d.transpose(0,2,3,1),
                  g.transpose(0,2,1),s,epsilon,interpret,token_tile)
    return result.transpose(0,3,1,2)
  out=_map_batch(local,(matrix,address,data,gate,static_key),(True,True,True,True,False))
  return out[0] if unbatched else out

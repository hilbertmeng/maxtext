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


def _call(m,a,d,g,s,epsilon,interpret,tile,key_contiguous):
  b,_,_,t=m.shape
  k,v=(m.shape[2],m.shape[1]) if key_contiguous else m.shape[1:3]
  h=a.shape[1]
  def kernel(m,a,d,g,s,out):
    mm,dd=m[...],d[...]
    if key_contiguous:mm,dd=mm.swapaxes(0,1),dd.swapaxes(0,1)
    y=_tile(mm,a[...],dd,g[...],s[...],epsilon)
    out[...]=y.swapaxes(0,1) if key_contiguous else y
  matrix_spec=_spec((v,k) if key_contiguous else (k,v),tile)
  data_spec=_spec((v,h) if key_contiguous else (h,v),tile)
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[matrix_spec,_spec((h,k),tile),data_spec,_spec((h,),tile),
                pl.BlockSpec(s.shape,lambda b,i:(0,0))],out_specs=matrix_spec,
      out_shape=jax.ShapeDtypeStruct(m.shape,m.dtype),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel'),
          allow_input_fusion=((True,)*5 if os.environ.get('RMT_PALLAS_INPUT_FUSION')=='1' else None)),
      name='rmt_token_minor_write')(m,a,d,g,s)


@partial(jax.custom_vjp,nondiff_argnums=(5,6,7,8))
def _write(m,a,d,g,s,epsilon,interpret,tile,key_contiguous):return _call(m,a,d,g,s,epsilon,interpret,tile,key_contiguous)


def _fwd(m,a,d,g,s,epsilon,interpret,tile,key_contiguous):
  return _call(m,a,d,g,s,epsilon,interpret,tile,key_contiguous),(a,d,g,s)


def _bwd(epsilon,interpret,tile,key_contiguous,args,dy):
  a,d,g,s=args
  b,_,_,t=dy.shape;h=a.shape[1]
  k,v=(dy.shape[2],dy.shape[1]) if key_contiguous else dy.shape[1:3]
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    _,pb=jax.vjp(lambda aa,dd,gg,ss:_tile(jnp.zeros((k,v,tile),dy.dtype),aa,dd,gg,ss,epsilon),
                 a[...],d[...].swapaxes(0,1) if key_contiguous else d[...],g[...],s[...])
    ga,gd,gg,gs=pb(dy[...].swapaxes(0,1) if key_contiguous else dy[...])
    da[...],dd[...],dg[...],ds[...]=ga,gd.swapaxes(0,1) if key_contiguous else gd,gg,gs
  shapes=[a.shape,d.shape,g.shape,(b,t//tile)+s.shape]
  specs=[_spec((h,k),tile),_spec((v,h) if key_contiguous else (h,v),tile),_spec((h,),tile),pl.BlockSpec(s.shape,lambda b,i:(0,0)),_spec((v,k) if key_contiguous else (k,v),tile)]
  da,dd,dg,ds=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=specs,
      out_specs=[*specs[:3],pl.BlockSpec((None,None)+s.shape,lambda b,i:(b,i,0,0))],
      out_shape=[jax.ShapeDtypeStruct(shape,a.dtype) for shape in shapes],interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel'),
          allow_input_fusion=((True,)*5 if os.environ.get('RMT_PALLAS_INPUT_FUSION')=='1' else None)),
      name='rmt_token_minor_write_backward')(a,d,g,s,dy)
  return dy,da,dd,dg,jnp.sum(ds.astype(jnp.float32),axis=(0,1)).astype(s.dtype)


_write.defvjp(_fwd,_bwd)


def write_residual(matrix,address,data,gate,static_key,epsilon=1e-6,*,interpret=False,tile=128,key_contiguous=False):
  unbatched=matrix.ndim==3
  if unbatched:
    matrix,address,data,gate=(x[None] for x in (matrix,address,data,gate))
  def local(m,a,d,g,s):
    token_tile=min(tile,m.shape[1])
    if m.shape[1]%token_tile:raise ValueError('Token tile must divide sequence length')
    order=(0,3,2,1) if key_contiguous else (0,2,3,1)
    result=_write(m.transpose(order),a.transpose(0,2,3,1),d.transpose(order),
                  g.transpose(0,2,1),s,epsilon,interpret,token_tile,key_contiguous)
    return result.transpose((0,3,2,1) if key_contiguous else (0,3,1,2))
  out=_map_batch(local,(matrix,address,data,gate,static_key),(True,True,True,True,False))
  return out[0] if unbatched else out

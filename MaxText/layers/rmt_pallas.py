"""Experimental TPU matrix-stream kernels. No attention or parameter changes.

The reference functions intentionally retain BF16 cast boundaries. Custom VJPs
differentiate those equations inside the kernel, rather than materializing every
intermediate in HBM. Shared parameter gradients use an explicit token reduction.
"""
from functools import partial
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _rms(x, epsilon):
  f = x.astype(jnp.float32)
  return (f * jax.lax.rsqrt(jnp.mean(f * f, axis=-1, keepdims=True) + epsilon)).astype(x.dtype)


def write_reference(matrix, address, data, gate, static_key, epsilon=1e-6):
  """One token: preserve the two rounded writes and left-associated residual."""
  a, d = _rms(address, epsilon), _rms(data, epsilon)
  static = jnp.einsum('hv,hk->kv', data, static_key)
  dynamic = jnp.einsum('hk,hv->kv', gate * a, d)
  return matrix + static + dynamic


def _block_diagonal(x):
  t,rows,cols=x.shape
  same=jnp.arange(t)[:,None,None,None]==jnp.arange(t)[None,None,:,None]
  values=jnp.where(same,x.astype(jnp.float32)[:,:,None,:],0)
  return values.reshape(t*rows,t*cols).astype(x.dtype)


def _write_tile(matrix, address, data, gate, static_key, epsilon):
  t,h,k=address.shape
  v=data.shape[-1]
  a,d=_rms(address,epsilon),_rms(data,epsilon)
  # Pack independent small contractions into a larger block-diagonal MXU dot.
  # Static keys reuse one GEMM across all tokens instead of repeating the key.
  vp=((v+127)//128)*128
  dp=jnp.concatenate((data,jnp.zeros(data.shape[:-1]+(vp-v,),data.dtype)),axis=-1)
  dn=jnp.concatenate((d,jnp.zeros(d.shape[:-1]+(vp-v,),d.dtype)),axis=-1)
  static=jnp.dot(static_key.T,dp.transpose(1,0,2).reshape(h,t*vp),
                 preferred_element_type=jnp.float32).astype(data.dtype)
  static=static.reshape(k,t,vp).transpose(1,0,2)[...,:v]
  blocked=_block_diagonal((gate*a).transpose(0,2,1))
  dynamic=jnp.dot(blocked,dn.reshape(t*h,vp),
                  preferred_element_type=jnp.float32).astype(data.dtype)
  dynamic=dynamic.reshape(t,k,vp)[...,:v]
  return matrix+static+dynamic


def _tile_size(n):
  tile=min(int(os.environ.get('RMT_PALLAS_TILE','8')),n)
  if tile < 1:
    raise ValueError('RMT_PALLAS_TILE must be positive')
  return tile if n%tile==0 else 1


def _write_call(matrix, address, data, gate, static_key, epsilon, interpret):
  n,k,v=matrix.shape
  h=data.shape[-2]
  tile=_tile_size(n)
  def kernel(m,a,d,g,s,out):
    out[...]=_write_tile(m[...],a[...],d[...],g[...],s[...],epsilon)
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,0,0))
  return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(matrix.shape,matrix.dtype),
      grid=(n//tile,),in_specs=[token((k,v)),token((h,k)),token((h,v)),token((h,1)),
                               pl.BlockSpec((h,k),lambda i:(0,0))],
      out_specs=token((k,v)),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_residual')(matrix,address,data,gate,static_key)


@partial(jax.custom_vjp, nondiff_argnums=(5,6))
def _write(matrix, address, data, gate, static_key, epsilon, interpret):
  return _write_call(matrix,address,data,gate,static_key,epsilon,interpret)


def _write_fwd(matrix,address,data,gate,static_key,epsilon,interpret):
  return _write_call(matrix,address,data,gate,static_key,epsilon,interpret), (matrix,address,data,gate,static_key)


def _write_bwd(epsilon,interpret,res,cotangent):
  matrix,address,data,gate,static_key=res
  n,k,v=matrix.shape
  h=data.shape[-2]
  tile=_tile_size(n)
  # The residual derivative is identity; its primal is not read by the kernel.
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    _,pullback=jax.vjp(lambda aa,dd,gg,ss:_write_tile(
        jnp.zeros((tile,k,v),data.dtype),aa,dd,gg,ss,epsilon),a[...],d[...],g[...],s[...])
    da[...],dd[...],dg[...],ds[...]=pullback(dy[...])
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,0,0))
  specs=[token((h,k)),token((h,v)),token((h,1)),pl.BlockSpec((h,k),lambda i:(0,0)),token((k,v))]
  shapes=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (address,data,gate)]
  shapes.append(jax.ShapeDtypeStruct((n//tile,h,k),static_key.dtype))
  da,dd,dg,ds=pl.pallas_call(kernel,out_shape=shapes,grid=(n//tile,),in_specs=specs,
      out_specs=[token((h,k)),token((h,v)),token((h,1)),
                 pl.BlockSpec((None,h,k),lambda i:(i,0,0))],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_residual_backward')(address,data,gate,static_key,cotangent)
  # Accumulate shared-key gradients in FP32, then cast to the parameter dtype.
  ds=jnp.sum(ds.astype(jnp.float32),axis=0).astype(static_key.dtype)
  return cotangent,da,dd,dg,ds


_write.defvjp(_write_fwd,_write_bwd)


def write_residual(matrix,address,data,gate,static_key,epsilon=1e-6,*,interpret=False):
  """Batched [..., K, V] update; gate has shape [..., H]."""
  leading=matrix.shape[:-2]
  n=math.prod(leading)
  return _write(matrix.reshape((n,)+matrix.shape[-2:]),
                address.reshape((n,)+address.shape[-2:]),
                data.reshape((n,)+data.shape[-2:]),
                gate.reshape((n,data.shape[-2],1)),static_key,epsilon,interpret).reshape(matrix.shape)


def c8_reference(matrix,key,compression,gates,epsilon=1e-6):
  compressed=jnp.einsum('vc,cr->vr',matrix,compression)
  read=jnp.einsum('vr,hr->hv',compressed,_rms(key,epsilon))
  return (.2*gates)[...,None]*read[:,None,:]


def _c8_tile(matrix,key,compression,gates,epsilon):
  t,v,c=matrix.shape
  h,r=key.shape[-2:]
  vp=((v+127)//128)*128
  compressed=jnp.dot(matrix.reshape(t*v,c),compression,
                     preferred_element_type=jnp.float32).astype(matrix.dtype)
  compressed=compressed.reshape(t,v,r).transpose(0,2,1)
  compressed=jnp.concatenate((compressed,jnp.zeros(compressed.shape[:-1]+(vp-v,),compressed.dtype)),axis=-1)
  blocked=_block_diagonal(_rms(key,epsilon))
  read=jnp.dot(blocked,compressed.reshape(t*r,vp),
               preferred_element_type=jnp.float32).astype(matrix.dtype)
  read=read.reshape(t,h,vp)[...,:v]
  return (.2*gates)[...,None]*read[:,:,None,:]


def _c8_call(matrix,key,compression,gates,epsilon,interpret):
  n,v,c=matrix.shape
  h,r=key.shape[-2:]
  destinations=gates.shape[-1]
  tile=_tile_size(n)
  def kernel(m,k,p,g,y):
    y[...]=_c8_tile(m[...],k[...],p[...],g[...],epsilon)
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,0,0))
  return pl.pallas_call(kernel,grid=(n//tile,),
      in_specs=[token((v,c)),token((h,r)),pl.BlockSpec((c,r),lambda i:(0,0)),token((h,destinations))],
      out_specs=pl.BlockSpec((tile,h,destinations,v),lambda i:(i,0,0,0)),
      out_shape=jax.ShapeDtypeStruct((n,h,destinations,v),matrix.dtype),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_c8_read')(matrix,key,compression,gates)


@partial(jax.custom_vjp,nondiff_argnums=(4,5))
def _c8(matrix,key,compression,gates,epsilon,interpret):
  return _c8_call(matrix,key,compression,gates,epsilon,interpret)


def _c8_fwd(matrix,key,compression,gates,epsilon,interpret):
  return _c8_call(matrix,key,compression,gates,epsilon,interpret),(matrix,key,compression,gates)


def _c8_bwd(epsilon,interpret,res,dy):
  matrix,key,compression,gates=res
  n,v,c=matrix.shape
  h,r=key.shape[-2:]
  dest=gates.shape[-1]
  tile=_tile_size(n)
  def kernel(m,k,p,g,y,dm,dk,dp,dg):
    _,pb=jax.vjp(lambda mm,kk,pp,gg:_c8_tile(mm,kk,pp,gg,epsilon),m[...],k[...],p[...],g[...])
    dm[...],dk[...],dp[...],dg[...]=pb(y[...])
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,0,0))
  shapes=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (matrix,key)]
  shapes.extend([jax.ShapeDtypeStruct((n//tile,c,r),compression.dtype),jax.ShapeDtypeStruct(gates.shape,gates.dtype)])
  dm,dk,dp,dg=pl.pallas_call(kernel,grid=(n//tile,),out_shape=shapes,
      in_specs=[token((v,c)),token((h,r)),pl.BlockSpec((c,r),lambda i:(0,0)),token((h,dest)),
                pl.BlockSpec((tile,h,dest,v),lambda i:(i,0,0,0))],
      out_specs=[token((v,c)),token((h,r)),pl.BlockSpec((None,c,r),lambda i:(i,0,0)),token((h,dest))],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_c8_read_backward')(matrix,key,compression,gates,dy)
  return dm,dk,jnp.sum(dp.astype(jnp.float32),axis=0).astype(compression.dtype),dg


_c8.defvjp(_c8_fwd,_c8_bwd)


def c8_read(matrix,key,compression,gates,epsilon=1e-6,*,interpret=False):
  n=math.prod(matrix.shape[:-2])
  y=_c8(matrix.reshape((n,)+matrix.shape[-2:]),key.reshape((n,)+key.shape[-2:]),
        compression,gates.reshape((n,)+gates.shape[-2:]),epsilon,interpret)
  return y.reshape(matrix.shape[:-2]+y.shape[1:])

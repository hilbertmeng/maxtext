"""Experimental TPU matrix-stream kernels. No attention or parameter changes.

The reference functions intentionally retain BF16 cast boundaries. Custom VJPs
differentiate those equations inside the kernel, rather than materializing every
intermediate in HBM. Shared parameter gradients use an explicit token reduction.
"""
from functools import partial
import math

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


def _write_tile(matrix, address, data, gate, static_key, epsilon):
  t,h,k=address.shape
  v=data.shape[-1]
  a,d=_rms(address,epsilon),_rms(data,epsilon)
  # Pack independent small contractions into a larger block-diagonal MXU dot.
  # Static keys reuse one GEMM across all tokens instead of repeating the key.
  vp=((v+127)//128)*128
  dp=jnp.pad(data,((0,0),(0,0),(0,vp-v)))
  dn=jnp.pad(d,((0,0),(0,0),(0,vp-v)))
  static=jnp.dot(static_key.T,dp.transpose(1,0,2).reshape(h,t*vp),
                 preferred_element_type=jnp.float32).astype(data.dtype)
  static=static.reshape(k,t,vp).transpose(1,0,2)[...,:v]
  same=jnp.arange(t)[:,None]==jnp.arange(t)[None,:]
  blocked=jnp.where(same[:,None,:,None],(gate*a).transpose(0,2,1)[:,:,None,:],0)
  dynamic=jnp.dot(blocked.reshape(t*k,t*h),dn.reshape(t*h,vp),
                  preferred_element_type=jnp.float32).astype(data.dtype)
  dynamic=dynamic.reshape(t,k,vp)[...,:v]
  return matrix+static+dynamic


def _tile_size(n):
  return min(8,n) if n%min(8,n)==0 else 1


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

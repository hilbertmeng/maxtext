"""Experimental fusion across the attention-write / MLP-read boundary."""
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _rms, _tile_size, _write_tile, write_reference


def stage_reference(matrix,address,data,gate,static_key,read_key,gain,epsilon=1e-6):
  updated=write_reference(matrix,address,data,gate,static_key,epsilon)
  read=jnp.einsum('kv,kh->hv',updated,read_key)
  proxy=updated[:data.shape[-2]].reshape(-1)
  proxy=_rms(proxy,epsilon)*(1+gain)
  return updated,read,proxy


def _stage_tile(matrix,address,data,gate,static_key,read_key,gain,epsilon):
  updated=_write_tile(matrix,address,data,gate,static_key,epsilon)
  t,k,v=matrix.shape
  h=data.shape[-2]
  vp=((v+127)//128)*128
  padded=jnp.concatenate((updated,jnp.zeros((t,k,vp-v),updated.dtype)),axis=-1)
  read=jnp.dot(read_key.T,padded.transpose(1,0,2).reshape(k,t*vp),
               preferred_element_type=jnp.float32).astype(matrix.dtype)
  read=read.reshape(h,t,vp).transpose(1,0,2)[...,:v]
  # Preserve the 1200-coordinate norm while keeping the native matrix layout.
  # Packing V=75 into a vector is delegated to XLA outside the opaque call.
  proxy=jax.lax.slice_in_dim(updated,0,h,axis=1)
  fp32=proxy.astype(jnp.float32)
  inverse=jax.lax.rsqrt(jnp.mean(fp32*fp32,axis=(1,2),keepdims=True)+epsilon)
  proxy=(fp32*inverse).astype(matrix.dtype)*(1+gain)
  return updated,read,proxy


def _spec(shape,tile,shared=False,partial_grad=False):
  zeros=(0,)*len(shape)
  if shared:return pl.BlockSpec(shape,lambda i:zeros)
  return pl.BlockSpec((None if partial_grad else tile,)+shape,lambda i:(i,)+zeros)


def _call(args,epsilon,interpret):
  n,k,v=args[0].shape
  h=args[2].shape[-2]
  tile=_tile_size(n)
  def kernel(*refs):
    values=[r[...] for r in refs[:7]]
    out=_stage_tile(*values,epsilon)
    for ref,x in zip(refs[7:],out):ref[...]=x
  shapes=((n,k,v),(n,h,v),(n,h,v))
  specs=[_spec(x.shape[1:] if i<4 else x.shape,tile,shared=i>=4) for i,x in enumerate(args)]
  return pl.pallas_call(kernel,grid=(n//tile,),in_specs=specs,
      out_specs=tuple(_spec(s[1:],tile) for s in shapes),
      out_shape=tuple(jax.ShapeDtypeStruct(s,args[0].dtype) for s in shapes),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_mlp_read_stage')(*args)


@partial(jax.custom_vjp,nondiff_argnums=(7,8))
def _stage(m,a,d,g,s,r,gain,epsilon,interpret):
  return _call((m,a,d,g,s,r,gain),epsilon,interpret)


def _fwd(m,a,d,g,s,r,gain,epsilon,interpret):
  args=(m,a,d,g,s,r,gain)
  return _call(args,epsilon,interpret),args


def _bwd(epsilon,interpret,args,cotangents):
  n=args[0].shape[0]
  tile=_tile_size(n)
  def kernel(*refs):
    values=[r[...] for r in refs[:7]]
    _,pb=jax.vjp(lambda *x:_stage_tile(*x,epsilon),*values)
    grads=pb(tuple(r[...] for r in refs[7:10]))
    for ref,x in zip(refs[10:],grads):ref[...]=x
  specs=[_spec(x.shape[1:] if i<4 else x.shape,tile,shared=i>=4) for i,x in enumerate(args)]
  specs.extend(_spec(x.shape[1:],tile) for x in cotangents)
  outshapes=[x.shape if i<4 else (n//tile,)+x.shape for i,x in enumerate(args)]
  outspecs=[_spec(s[1:],tile,partial_grad=i>=4) for i,s in enumerate(outshapes)]
  grads=pl.pallas_call(kernel,grid=(n//tile,),in_specs=specs,out_specs=outspecs,
      out_shape=[jax.ShapeDtypeStruct(s,x.dtype) for s,x in zip(outshapes,args)],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_mlp_read_stage_backward')(*args,*cotangents)
  return tuple(x if i<4 else jnp.sum(x.astype(jnp.float32),axis=0).astype(args[i].dtype)
               for i,x in enumerate(grads))


_stage.defvjp(_fwd,_bwd)


def write_mlp_stage(matrix,address,data,gate,static_key,read_key,gain,epsilon=1e-6,*,interpret=False):
  leading=matrix.shape[:-2]
  n=math.prod(leading)
  args=[x.reshape((n,)+x.shape[-2:]) for x in (matrix,address,data)]
  args.extend((gate.reshape(n,data.shape[-2]),static_key,read_key,gain.reshape(data.shape[-2:])))
  out=_stage(*args,epsilon,interpret)
  return (out[0].reshape(matrix.shape),out[1].reshape(data.shape),
          out[2].reshape(leading+(data.shape[-2]*data.shape[-1],)))

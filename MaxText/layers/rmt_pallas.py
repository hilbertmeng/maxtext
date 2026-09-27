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
  blocks=[]
  for i in range(t):
    pieces=[]
    if i:pieces.append(jnp.zeros((rows,i*cols),jnp.float32))
    pieces.append(x[i].astype(jnp.float32))
    if i+1<t:pieces.append(jnp.zeros((rows,(t-i-1)*cols),jnp.float32))
    blocks.append(jnp.concatenate(pieces,axis=1))
  return jnp.concatenate(blocks,axis=0).astype(x.dtype)


def _write_tile(matrix, address, data, gate, static_key, epsilon):
  gate=gate.astype(jnp.float32)[...,None].astype(gate.dtype)
  if os.environ.get('RMT_PALLAS_WRITE_IMPL','mxu') == 'vpu':
    return _write_tile_vpu(matrix,address,data,gate,static_key,epsilon)
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
  if os.environ.get('RMT_PALLAS_BATCHED_DOT')=='1':
    dynamic=jnp.einsum('thk,thv->tkv',gate*a,dn,
                       preferred_element_type=jnp.float32).astype(data.dtype)[...,:v]
  else:
    blocked=_block_diagonal((gate*a).transpose(0,2,1))
    dynamic=jnp.dot(blocked,dn.reshape(t*h,vp),
                    preferred_element_type=jnp.float32).astype(data.dtype)
    dynamic=dynamic.reshape(t,k,vp)[...,:v]
  return matrix+static+dynamic


def _write_tile_vpu(matrix,address,data,gate,static_key,epsilon):
  """Use token lanes for short head reductions; avoid block-diagonal zeros."""
  t,h,k=address.shape
  v=data.shape[-1]
  a=(gate*_rms(address,epsilon)).astype(jnp.float32)
  d=_rms(data,epsilon).astype(jnp.float32)
  raw=data.astype(jnp.float32)
  keys=static_key.astype(jnp.float32)
  static=jnp.zeros((k,v,t),jnp.float32)
  dynamic=jnp.zeros_like(static)
  for head in range(h):
    sk=jnp.broadcast_to(jax.lax.slice_in_dim(keys,head,head+1,axis=0).reshape(k,1),(k,t))
    ak=jax.lax.slice_in_dim(a,head,head+1,axis=1).reshape(t,k).T
    rv=jax.lax.slice_in_dim(raw,head,head+1,axis=1).reshape(t,v).T
    dv=jax.lax.slice_in_dim(d,head,head+1,axis=1).reshape(t,v).T
    static=static+sk[:,None,:]*rv[None,:,:]
    dynamic=dynamic+ak[:,None,:]*dv[None,:,:]
  static=static.transpose(2,0,1).astype(matrix.dtype)
  dynamic=dynamic.transpose(2,0,1).astype(matrix.dtype)
  return matrix+static+dynamic


def _tile_size(n,requested=None):
  tile=min(int(os.environ.get('RMT_PALLAS_TILE','8') if requested is None else requested),n)
  if tile < 1:
    raise ValueError('RMT_PALLAS_TILE must be positive')
  return tile if n%tile==0 else 1


def _write_call(matrix, address, data, gate, static_key, epsilon, interpret, tile):
  n,k,v=matrix.shape
  h=data.shape[-2]
  def kernel(m,a,d,g,s,out):
    out[...]=_write_tile(m[...],a[...],d[...],g[...],s[...],epsilon)
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,)+(0,)*len(shape))
  return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(matrix.shape,matrix.dtype),
      grid=(n//tile,),in_specs=[token((k,v)),token((h,k)),token((h,v)),token((h,)),
                               pl.BlockSpec((h,k),lambda i:(0,0))],
      out_specs=token((k,v)),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_residual')(matrix,address,data,gate,static_key)


@partial(jax.custom_vjp, nondiff_argnums=(5,6,7,8))
def _write(matrix, address, data, gate, static_key, epsilon, interpret, tile,forward_jax):
  if forward_jax:
    return jax.vmap(partial(write_reference,epsilon=epsilon),in_axes=(0,0,0,0,None))(
        matrix,address,data,gate[...,None],static_key)
  return _write_call(matrix,address,data,gate,static_key,epsilon,interpret,tile)


def _write_fwd(matrix,address,data,gate,static_key,epsilon,interpret,tile,forward_jax):
  return _write(matrix,address,data,gate,static_key,epsilon,interpret,tile,forward_jax), (matrix,address,data,gate,static_key)


def _write_bwd(epsilon,interpret,tile,forward_jax,res,cotangent):
  matrix,address,data,gate,static_key=res
  n,k,v=matrix.shape
  h=data.shape[-2]
  # The residual derivative is identity; its primal is not read by the kernel.
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    _,pullback=jax.vjp(lambda aa,dd,gg,ss:_write_tile(
        jnp.zeros((tile,k,v),data.dtype),aa,dd,gg,ss,epsilon),a[...],d[...],g[...],s[...])
    da[...],dd[...],dg[...],ds[...]=pullback(dy[...])
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,)+(0,)*len(shape))
  specs=[token((h,k)),token((h,v)),token((h,)),pl.BlockSpec((h,k),lambda i:(0,0)),token((k,v))]
  shapes=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (address,data,gate)]
  shapes.append(jax.ShapeDtypeStruct((n//tile,h,k),static_key.dtype))
  da,dd,dg,ds=pl.pallas_call(kernel,out_shape=shapes,grid=(n//tile,),in_specs=specs,
      out_specs=[token((h,k)),token((h,v)),token((h,)),
                 pl.BlockSpec((None,h,k),lambda i:(i,0,0))],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_write_residual_backward')(address,data,gate,static_key,cotangent)
  # Accumulate shared-key gradients in FP32, then cast to the parameter dtype.
  ds=jnp.sum(ds.astype(jnp.float32),axis=0).astype(static_key.dtype)
  return cotangent,da,dd,dg,ds


_write.defvjp(_write_fwd,_write_bwd)


def _map_batch(fn,args,batch_args,*,output_tuple=False):
  # MaxText0.8.1 uses the legacy `with mesh` context. An unwrapped opaque TPU
  # call would otherwise replicate global batches on the target v5p pod.
  from jax._src import mesh as mesh_lib
  mesh=mesh_lib.thread_resources.env.physical_mesh
  if not mesh.axis_names or mesh.size==1:
    return fn(*args)
  from flax import linen as nn
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  axes=nn.logical_to_mesh_axes(('activation_batch',))[0]
  names=(axes,) if isinstance(axes,str) else (axes or ())
  if not names or any(size>1 and name not in names for name,size in mesh.shape.items()):
    raise ValueError('RMT Pallas currently supports batch/FSDP mesh axes only')
  spec=P(axes)
  outputs=(spec,spec) if output_tuple else spec
  return shard_map(fn,mesh=mesh,in_specs=tuple(spec if b else P() for b in batch_args),
                   out_specs=outputs,check_rep=False)(*args)


def write_residual(matrix,address,data,gate,static_key,epsilon=1e-6,*,interpret=False,tile=None,forward_jax=False):
  """Batched [..., K, V] update; batch sharding and shared-key VJP are explicit."""
  def local(m,a,d,g,s):
    n=math.prod(m.shape[:-2])
    return _write(m.reshape((n,)+m.shape[-2:]),a.reshape((n,)+a.shape[-2:]),
                  d.reshape((n,)+d.shape[-2:]),g.reshape(n,d.shape[-2]),s,
                  epsilon,interpret,_tile_size(n,tile),forward_jax).reshape(m.shape)
  return _map_batch(local,(matrix,address,data,gate,static_key),(True,True,True,True,False))


def c8_reference(matrix,key,compression,gates,epsilon=1e-6):
  compressed=jnp.einsum('vc,cr->vr',matrix,compression)
  read=jnp.einsum('vr,hr->hv',compressed,_rms(key,epsilon))
  return (.2*gates)[...,None]*read[:,None,:]


def _c8_tile(matrix,key,compression,gates,epsilon):
  t,v,c=matrix.shape
  h,r=key.shape[-2:]
  vp=((v+127)//128)*128
  padded=jnp.concatenate((matrix,jnp.zeros((t,vp-v,c),matrix.dtype)),axis=1)
  compressed=jnp.dot(padded.reshape(t*vp,c),compression,
                     preferred_element_type=jnp.float32).astype(matrix.dtype)
  compressed=compressed.reshape(t,vp,r).transpose(0,2,1)
  if os.environ.get('RMT_PALLAS_BATCHED_DOT')=='1':
    read=jnp.einsum('thr,trv->thv',_rms(key,epsilon),compressed,
                    preferred_element_type=jnp.float32).astype(matrix.dtype)[...,:v]
  elif os.environ.get('RMT_PALLAS_C8_IMPL','mxu')=='vpu':
    normkey=_rms(key,epsilon).astype(jnp.float32)
    compressed=compressed.astype(jnp.float32)
    accum=jnp.zeros((h,vp,t),jnp.float32)
    for channel in range(r):
      a=jax.lax.slice_in_dim(normkey,channel,channel+1,axis=2).reshape(t,h).T
      b=jax.lax.slice_in_dim(compressed,channel,channel+1,axis=1).reshape(t,vp).T
      accum=accum+a[:,None,:]*b[None,:,:]
    read=accum.transpose(2,0,1).astype(matrix.dtype)[...,:v]
  else:
    blocked=_block_diagonal(_rms(key,epsilon))
    read=jnp.dot(blocked,compressed.reshape(t*r,vp),
                 preferred_element_type=jnp.float32).astype(matrix.dtype)
    read=read.reshape(t,h,vp)[...,:v]
  return ((.2*gates).astype(jnp.float32).transpose(0,2,1)[...,None]*
          read.astype(jnp.float32)[:,None,:,:]).astype(matrix.dtype)


def _c8_call(matrix,key,compression,gates,epsilon,interpret):
  n,v,c=matrix.shape
  h,r=key.shape[-2:]
  destinations=gates.shape[-1]
  tile=_tile_size(n)
  def kernel(m,k,p,g,y):
    y[...]=_c8_tile(m[...],k[...],p[...],g[...],epsilon)
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,)+(0,)*len(shape))
  return pl.pallas_call(kernel,grid=(n//tile,),
      in_specs=[token((v,c)),token((h,r)),pl.BlockSpec((c,r),lambda i:(0,0)),token((h,destinations))],
      out_specs=pl.BlockSpec((tile,destinations,h,v),lambda i:(i,0,0,0)),
      out_shape=jax.ShapeDtypeStruct((n,destinations,h,v),matrix.dtype),
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
  token=lambda shape:pl.BlockSpec((tile,)+shape,lambda i:(i,)+(0,)*len(shape))
  shapes=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (matrix,key)]
  shapes.extend([jax.ShapeDtypeStruct((n//tile,c,r),compression.dtype),jax.ShapeDtypeStruct(gates.shape,gates.dtype)])
  dm,dk,dp,dg=pl.pallas_call(kernel,grid=(n//tile,),out_shape=shapes,
      in_specs=[token((v,c)),token((h,r)),pl.BlockSpec((c,r),lambda i:(0,0)),token((h,dest)),
                pl.BlockSpec((tile,dest,h,v),lambda i:(i,0,0,0))],
      out_specs=[token((v,c)),token((h,r)),pl.BlockSpec((None,c,r),lambda i:(i,0,0)),token((h,dest))],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
      name='rmt_c8_read_backward')(matrix,key,compression,gates,dy)
  return dm,dk,jnp.sum(dp.astype(jnp.float32),axis=0).astype(compression.dtype),dg


_c8.defvjp(_c8_fwd,_c8_bwd)


def c8_read(matrix,key,compression,gates,epsilon=1e-6,*,interpret=False):
  n=math.prod(matrix.shape[:-2])
  y=_c8(matrix.reshape((n,)+matrix.shape[-2:]),key.reshape((n,)+key.shape[-2:]),
        compression,gates.reshape((n,)+gates.shape[-2:]),epsilon,interpret)
  y=y.transpose(0,2,1,3)
  return y.reshape(matrix.shape[:-2]+y.shape[1:])

"""Attention write through dynamic MLP read in one TPU program per token tile.

The updated matrix is consumed on chip by both reads and the proxy projection.
The reverse program joins every matrix cotangent before the analytic write
pullback. Only final outputs and shared-parameter gradient partials reach HBM.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _tile, _norm, _norm_backward, _analytic_backward, _spec


def contract(w,x):
  """Shared [out,in] weight times [in,value,token], with native MXU lanes."""
  n,v,t=x.shape;vp=((v+127)//128)*128
  padded=jnp.pad(x,((0,0),(0,vp-v),(0,0)))
  return jnp.dot(w,padded.reshape(n,vp*t),preferred_element_type=jnp.float32).reshape(w.shape[0],vp,t)[:,:v,:].astype(x.dtype)


def weight_grad(left,right):
  """Sum matching value/token axes; FP32 partials stay FP32 until global sum."""
  v,t=left.shape[-2:];vp=((v+127)//128)*128
  l=jnp.pad(left,((0,0),(0,vp-v),(0,0))).reshape(left.shape[0],vp*t)
  r=jnp.pad(right,((0,0),(0,vp-v),(0,0))).reshape(right.shape[0],vp*t)
  return jnp.dot(l,r.T,preferred_element_type=jnp.float32)


def read_state(m,c,scale,wk,wg,bias,epsilon,read_epsilon):
  h=wg.shape[1];v=m.shape[1];t=m.shape[-1];rank=c.shape[1]
  raw=m[:h].reshape(h*v,t)
  normalized=_norm(raw,epsilon)
  x=(normalized*scale[:,None]).astype(m.dtype)
  key=jnp.dot(wk.T,x,preferred_element_type=jnp.float32).astype(m.dtype).reshape(h,rank,t)
  logits=jnp.dot(wg.T,x,preferred_element_type=jnp.float32).astype(m.dtype)
  gate=jax.nn.sigmoid(logits+bias[:,None])
  compressed=contract(c.T,m[h:])
  kn=_norm(key,read_epsilon)
  read=jnp.zeros((h,v,t),jnp.float32)
  for i in range(rank):
    ki=jax.lax.slice_in_dim(kn,i,i+1,axis=1).reshape(h,t)
    ci=jax.lax.slice_in_dim(compressed,i,i+1,axis=0).reshape(v,t)
    read=read+ki.astype(jnp.float32)[:,None,:]*ci.astype(jnp.float32)[None,:,:]
  return raw,normalized,x,key,gate,compressed,kn,read.astype(m.dtype)


def forward(m,a,d,g,s,r,c,scale,wk,wg,bias,epsilon,read_epsilon):
  updated=_tile(m,a,d,g,s,epsilon)
  _,_,x,_,gate,_,_,read=read_state(updated,c,scale,wk,wg,bias,epsilon,read_epsilon)
  static=contract(r.T,updated)
  out=(static+((.2*gate)[:,None,:]*read).astype(m.dtype)).astype(m.dtype)
  return updated,out,x


def reverse(m,a,d,g,s,r,c,scale,wk,wg,bias,dm,dy,dx,epsilon,read_epsilon):
  # m is the forward's updated matrix, so backward does not repeat the write.
  raw,normalized,x,key,gate,compressed,kn,read=read_state(m,c,scale,wk,wg,bias,epsilon,read_epsilon)
  h,v,t=read.shape;rank=c.shape[1]
  dr=weight_grad(m,dy)
  dm=(dm+contract(r,dy)).astype(m.dtype)
  dgate=(.2*jax.lax.reduce_sum((dy*read).astype(dy.dtype),axes=(1,))).astype(dy.dtype)
  dread=(dy*(.2*gate)[:,None,:]).astype(dy.dtype)
  dc=[];dkn=[]
  for i in range(rank):
    ki=jax.lax.slice_in_dim(kn,i,i+1,axis=1).reshape(h,t).astype(jnp.float32)
    ci=jax.lax.slice_in_dim(compressed,i,i+1,axis=0).reshape(v,t).astype(jnp.float32)
    dc.append(jnp.sum(dread.astype(jnp.float32)*ki[:,None,:],axis=0).astype(m.dtype))
    dkn.append(jnp.sum(dread.astype(jnp.float32)*ci[None,:,:],axis=1).astype(m.dtype))
  dc=jnp.stack(dc);dkn=jnp.stack(dkn,axis=1)
  dcompression=weight_grad(m[h:],dc)
  dmtail=contract(c,dc)
  dk=_norm_backward(key,dkn,read_epsilon).reshape(h*rank,t)
  dg=(dgate*(gate*(1-gate))).astype(m.dtype)
  dwk=jnp.dot(x,dk.T,preferred_element_type=jnp.float32)
  dwg=jnp.dot(x,dg.T,preferred_element_type=jnp.float32)
  db=jnp.sum(dg.astype(jnp.float32),axis=1)
  dx=(dx+jnp.dot(wk,dk,preferred_element_type=jnp.float32).astype(m.dtype)).astype(m.dtype)
  dx=(dx+jnp.dot(wg,dg,preferred_element_type=jnp.float32).astype(m.dtype)).astype(m.dtype)
  dscale=jnp.sum((dx*normalized).astype(m.dtype).astype(jnp.float32),axis=1)
  draw=_norm_backward(raw,(dx*scale[:,None]).astype(m.dtype),epsilon).reshape(h,v,t)
  dm=(dm+jnp.concatenate((draw,dmtail),axis=0)).astype(m.dtype)
  da,dd,dg_write,ds=_analytic_backward(a,d,g,s,dm,epsilon)
  return dm,da,dd,dg_write,ds.astype(jnp.float32),dr,dcompression,dscale,dwk,dwg,db


def shared_spec(x):
  return pl.BlockSpec(x.shape,lambda b,i:(0,)*x.ndim)


def specs(args,tile):
  return [_spec(x.shape[1:-1],tile) if i<4 else shared_spec(x) for i,x in enumerate(args)]


def call(args,epsilon,read_epsilon,interpret,tile):
  b,k,v,t=args[0].shape;h=args[2].shape[1]
  def kernel(*refs):
    outputs=forward(*(x[...] for x in refs[:11]),epsilon,read_epsilon)
    for ref,value in zip(refs[11:],outputs):ref[...]=value
  shapes=[args[0].shape,(b,h,v,t),(b,h*v,t)]
  return pl.pallas_call(kernel,grid=(b,t//tile),in_specs=specs(args,tile),
      out_specs=[_spec(x[1:-1],tile) for x in shapes],
      out_shape=tuple(jax.ShapeDtypeStruct(x,args[0].dtype) for x in shapes),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_fused_write_mlp_read')(*args)


@partial(jax.custom_vjp,nondiff_argnums=(11,12,13,14))
def fused(m,a,d,g,s,r,c,scale,wk,wg,bias,epsilon,read_epsilon,interpret,tile):
  return call((m,a,d,g,s,r,c,scale,wk,wg,bias),epsilon,read_epsilon,interpret,tile)


def fwd(m,a,d,g,s,r,c,scale,wk,wg,bias,epsilon,read_epsilon,interpret,tile):
  out=call((m,a,d,g,s,r,c,scale,wk,wg,bias),epsilon,read_epsilon,interpret,tile)
  return out,(out[0],a,d,g,s,r,c,scale,wk,wg,bias)


def bwd(epsilon,read_epsilon,interpret,tile,args,cotangents):
  b,_,_,t=args[0].shape
  def kernel(*refs):
    gradients=reverse(*(x[...] for x in refs[:14]),epsilon,read_epsilon)
    for ref,value in zip(refs[14:],gradients):ref[...]=value
  inp=specs(args,tile)+[_spec(x.shape[1:-1],tile) for x in cotangents]
  outspec=inp[:4]+[pl.BlockSpec((None,None)+x.shape,
      lambda b,i,ndim=x.ndim:(b,i)+(0,)*ndim) for x in args[4:]]
  outshape=[jax.ShapeDtypeStruct(x.shape,x.dtype) if i<4 else
            jax.ShapeDtypeStruct((b,t//tile)+x.shape,jnp.float32) for i,x in enumerate(args)]
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outspec,
      out_shape=tuple(outshape),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_fused_write_mlp_read_backward')(*args,*cotangents)
  return tuple(x if i<4 else jnp.sum(x,axis=(0,1)).astype(args[i].dtype) for i,x in enumerate(grads))


fused.defvjp(fwd,bwd)


def write_mlp_read(m,a,d,g,s,r,c,scale,wk,wg,bias,epsilon=1e-6,read_epsilon=1e-6,*,interpret=False,tile=128):
  """Public M[B,T,K,V], head[B,T,H,V], vector[B,T,H*V] interface."""
  def local(m,a,d,g,*weights):
    block=min(tile,m.shape[1])
    if m.shape[1]%block:raise ValueError('Sequence length must divide the fused token tile')
    out=fused(m.transpose(0,2,3,1),a.transpose(0,2,3,1),d.transpose(0,2,3,1),g.transpose(0,2,1),
              *weights,epsilon,read_epsilon,interpret,block)
    return out[0].transpose(0,3,1,2),out[1].transpose(0,3,1,2),out[2].transpose(0,2,1)
  return _map_batch(local,(m,a,d,g,s,r,c,scale,wk,wg,bias),(True,)*4+(False,)*7,output_tuple=3)

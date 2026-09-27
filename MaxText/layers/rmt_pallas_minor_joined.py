"""Shared static/C8 matrix projection with token-contiguous kernel buffers."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _spec
from layers.rmt_pallas_minor_read import _tile as _dynamic


def _project(m,p):
  k,v,t=m.shape
  vp=((v+127)//128)*128
  padded=jnp.concatenate((m,jnp.zeros((k,vp-v,t),m.dtype)),axis=1)
  return jnp.dot(p.T,padded.reshape(k,vp*t),preferred_element_type=jnp.float32).astype(m.dtype).reshape(p.shape[1],vp,t)


def _call(m,k,p,g,epsilon,interpret,tile,save=False):
  b,rows,v,t=m.shape;h,r=k.shape[1:3];d=g.shape[2];s=p.shape[1]-r;vp=((v+127)//128)*128
  def kernel(m,k,p,g,*outs):
    projected=_project(m[...],p[...]);c=projected[s:]
    outs[0][...]=projected[:s,:v]
    outs[1][...]=_dynamic(c,k[...],g[...],epsilon)[:,:,:v]
    if save:outs[2][...]=c
  shapes=[(b,s,v,t),(b,d,h,v,t)];specs=[_spec((s,v),tile),_spec((d,h,v),tile)]
  if save:shapes.append((b,r,vp,t));specs.append(_spec((r,vp),tile))
  return pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=[_spec((rows,v),tile),_spec((h,r),tile),pl.BlockSpec(p.shape,lambda b,i:(0,0)),_spec((h,d),tile)],
      out_specs=tuple(specs),out_shape=tuple(jax.ShapeDtypeStruct(x,m.dtype) for x in shapes),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_joined_read')(m,k,p,g)


@partial(jax.custom_vjp,nondiff_argnums=(4,5,6))
def _read(m,k,p,g,epsilon,interpret,tile):return _call(m,k,p,g,epsilon,interpret,tile)


def _fwd(m,k,p,g,epsilon,interpret,tile):
  s,d,c=_call(m,k,p,g,epsilon,interpret,tile,save=True)
  return (s,d),(m,k,p,g,c)


def _bwd(epsilon,interpret,tile,args,cotangents):
  m,k,p,g,c=args;ds,dd=cotangents
  b,rows,v,t=m.shape;h,r=k.shape[1:3];d=g.shape[2];s=p.shape[1]-r;vp=c.shape[2]
  def kernel(m,k,p,g,c,ds,dd,dm,dk,dp,dg):
    _,pull_read=jax.vjp(lambda cc,kk,gg:_dynamic(cc,kk,gg,epsilon),c[...],k[...],g[...])
    padded_dd=jnp.concatenate((dd[...],jnp.zeros((d,h,vp-v,tile),dd.dtype)),axis=2)
    dc,qgrad,ggrad=pull_read(padded_dd)
    padded_ds=jnp.concatenate((ds[...],jnp.zeros((s,vp-v,tile),ds.dtype)),axis=1)
    _,pull_project=jax.vjp(_project,m[...],p[...])
    mgrad,pgrad=pull_project(jnp.concatenate((padded_ds,dc),axis=0))
    dm[...],dk[...],dp[...],dg[...]=mgrad,qgrad,pgrad,ggrad
  specs=[_spec((rows,v),tile),_spec((h,r),tile),pl.BlockSpec(p.shape,lambda b,i:(0,0)),_spec((h,d),tile)]
  shapes=[m.shape,k.shape,(b,t//tile)+p.shape,g.shape]
  grads=pl.pallas_call(kernel,grid=(b,t//tile),
      in_specs=specs+[_spec((r,vp),tile),_spec((s,v),tile),_spec((d,h,v),tile)],
      out_specs=[specs[0],specs[1],pl.BlockSpec((None,None)+p.shape,lambda b,i:(b,i,0,0)),specs[3]],
      out_shape=[jax.ShapeDtypeStruct(x,m.dtype) for x in shapes],
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_token_minor_joined_read_backward')(*args,*cotangents)
  return grads[0],grads[1],jnp.sum(grads[2].astype(jnp.float32),axis=(0,1)).astype(p.dtype),grads[3]


_read.defvjp(_fwd,_bwd)


def joined_read(matrix,key,projection,gates,epsilon=1e-6,*,interpret=False,tile=128):
  unbatched=matrix.ndim==3
  if unbatched:matrix,key,gates=(x[None] for x in (matrix,key,gates))
  def local(m,k,p,g):
    n=min(tile,m.shape[1])
    if m.shape[1]%n:raise ValueError('Token tile must divide sequence length')
    s,d=_read(m.transpose(0,2,3,1),k.transpose(0,2,3,1),p,g.transpose(0,2,3,1),epsilon,interpret,n)
    return s.transpose(0,3,1,2),d.transpose(0,4,2,1,3)
  s,d=_map_batch(local,(matrix,key,projection,gates),(True,True,False,True),output_tuple=True)
  return (s[0],d[0]) if unbatched else (s,d)

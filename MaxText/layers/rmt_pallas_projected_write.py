"""One write program including packed GELU address/gate projection.

Forward reuses the proven token-minor write. Reverse uses the native chunked
joint MXU pullback, then explicit GELU and projection derivatives.
"""
from functools import partial
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _tile, _spec
from layers.rmt_pallas_write_reverse import joint
from layers.rmt_pallas_write_read_major import pack_proxy, unpack_proxy


def gelu(x):
  f=x.astype(jnp.float32)
  return (.5*f*(1+jnp.tanh(math.sqrt(2/math.pi)*(f+.044715*f*f*f)))).astype(x.dtype)


def gelu_reverse(x,u):
  f=x.astype(jnp.float32)
  t=jnp.tanh(math.sqrt(2/math.pi)*(f+.044715*f*f*f))
  slope=.5*(1+t)+.5*f*(1-t*t)*math.sqrt(2/math.pi)*(1+3*.044715*f*f)
  return (u.astype(jnp.float32)*slope).astype(x.dtype)


def project_minor(x,down,up,ub,wg,gb):
  return project_minor_state(x,down,up,ub,wg,gb)[2:]


def project_minor_state(x,down,up,ub,wg,gb):
  r=down.shape[1];h,k=ub.shape
  p=jnp.dot(jnp.concatenate((down,wg),axis=1).T,x,preferred_element_type=jnp.float32).astype(x.dtype)
  hidden=gelu(p[:r])
  raw=jnp.dot(up.T,hidden,preferred_element_type=jnp.float32).astype(x.dtype).reshape(h,k,x.shape[1])
  raw=(raw+ub.astype(jnp.float32)[:,:,None]).astype(x.dtype)
  logits=(p[r:].astype(jnp.float32)+gb.astype(jnp.float32)[:,None]).astype(x.dtype)
  gates=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(x.dtype)
  return p[:r],hidden,raw,gates


def project_reverse_minor(x,down,up,wg,raw_hidden,hidden,gate,da,dg,store,dot_store=None):
  ga=da.reshape(up.shape[1],x.shape[1])
  if dot_store is None:store(2,jnp.dot(hidden,ga.T,preferred_element_type=jnp.float32))
  else:dot_store(2,hidden,ga)
  store(3,jnp.sum(da.astype(jnp.float32),axis=2))
  dh=jnp.dot(up,ga,preferred_element_type=jnp.float32).astype(x.dtype)
  dh=gelu_reverse(raw_hidden,dh)
  dg=(dg*(gate*(1-gate))).astype(x.dtype)
  store(5,jnp.sum(dg.astype(jnp.float32),axis=1))
  dp=jnp.concatenate((dh,dg),axis=0)
  if dot_store is None:
    dw=jnp.dot(x,dp.T,preferred_element_type=jnp.float32)
    store(1,dw[:,:down.shape[1]])
    store(4,dw[:,down.shape[1]:])
  else:
    dot_store(1,x,dh)
    dot_store(4,x,dg)
  return jnp.dot(jnp.concatenate((down,wg),axis=1),dp,preferred_element_type=jnp.float32).astype(x.dtype)


def project_major(x,down,up,ub,wg,gb):
  r=down.shape[1];h,k=ub.shape
  p=jnp.dot(x,jnp.concatenate((down,wg),axis=1),preferred_element_type=jnp.float32).astype(x.dtype)
  raw_hidden=p[:,:r];hidden=gelu(raw_hidden)
  address=jnp.dot(hidden,up,preferred_element_type=jnp.float32).astype(x.dtype)
  address=(unpack_proxy(address,h,k)+ub[None,:,:]).astype(x.dtype)
  logits=(p[:,r:]+gb[None,:]).astype(x.dtype)
  gate=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(x.dtype)
  return raw_hidden,hidden,address,gate


def project_reverse(x,down,up,ub,wg,gb,raw_hidden,hidden,gate,da,dg,shared_sink=None):
  def finish(index,value):
    if shared_sink is None:return value
    shared_sink(index,value)
    return None
  ga=pack_proxy(da)
  du=finish(2,jnp.dot(hidden.T,ga,preferred_element_type=jnp.float32))
  dub=finish(3,jnp.sum(da.astype(jnp.float32),axis=0))
  dh=jnp.dot(ga,up.T,preferred_element_type=jnp.float32).astype(x.dtype)
  dh=gelu_reverse(raw_hidden,dh)
  dg=(dg*(gate*(1-gate))).astype(x.dtype)
  db=finish(5,jnp.sum(dg.astype(jnp.float32).T,axis=1)[None,:])
  dp=jnp.concatenate((dh,dg),axis=1)
  dw=jnp.dot(x.T,dp,preferred_element_type=jnp.float32)
  ddown=finish(1,dw[:,:down.shape[1]])
  dgate=finish(4,dw[:,down.shape[1]:])
  dx=jnp.dot(dp,jnp.concatenate((down,wg),axis=1).T,preferred_element_type=jnp.float32).astype(x.dtype)
  return dx,ddown,du,dub,dgate,db


def forward_call(args,epsilon,interpret,tile):
  m,x,d,s,down,up,ub,wg,gb=args;b,_,_,t=m.shape
  inp=[_spec(z.shape[1:-1],tile) if i<3 else
       pl.BlockSpec(z.shape,lambda b,j,n=z.ndim:(0,)*n) for i,z in enumerate(args)]
  def kernel(m,x,d,s,down,up,ub,wg,gb,out):
    a,g=project_minor(x[...],down[...],up[...],ub[...],wg[...],gb[...])
    out[...]=_tile(m[...],a,d[...],g,s[...],epsilon)
  return pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=inp[0],
      out_shape=jax.ShapeDtypeStruct(m.shape,m.dtype),input_output_aliases={0:0},
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_projected_mlp_write')(*args)


@partial(jax.custom_vjp,nondiff_argnums=(9,10,11,12))
def fused(m,x,d,s,down,up,ub,wg,gb,epsilon,interpret,forward_tile,reverse_tile):
  return forward_call((m,x,d,s,down,up,ub,wg,gb),epsilon,interpret,forward_tile)


def fwd(m,x,d,s,down,up,ub,wg,gb,epsilon,interpret,forward_tile,reverse_tile):
  args=(m,x,d,s,down,up,ub,wg,gb)
  return forward_call(args,epsilon,interpret,forward_tile),(x,d,s,down,up,ub,wg,gb)


def bwd(epsilon,interpret,forward_tile,reverse_tile,args,dm):
  x,d,s,down,up,ub,wg,gb=args
  x=x.transpose(0,2,1);d=d.transpose(0,3,1,2);dm=dm.transpose(0,3,1,2)
  args=(x,d,s,down,up,ub,wg,gb);b,t=x.shape[:2];tile=min(reverse_tile,t)
  def spec(z):return pl.BlockSpec((None,tile)+z.shape[2:],lambda b,j:(b,j)+(0,)*(z.ndim-2))
  inp=[spec(z) if i<2 else pl.BlockSpec(z.shape,lambda b,j,n=z.ndim:(0,)*n) for i,z in enumerate(args)]+[spec(dm)]
  shared_shapes=[(1,)+z.shape if z.ndim==1 else z.shape for z in args[2:]]
  out_specs=inp[:2]+[pl.BlockSpec((None,)+sh,lambda b,j,n=len(sh):(b,)+(0,)*n) for sh in shared_shapes]
  out_shapes=[jax.ShapeDtypeStruct(z.shape,z.dtype) if i<2 else
              jax.ShapeDtypeStruct((b,)+shared_shapes[i-2],jnp.float32) for i,z in enumerate(args)]
  def kernel(*refs):
    xx,dd,ss,wd,wu,bu,wg,bg,dy=(r[...] for r in refs[:9])
    pre,hidden,address,gate=project_major(xx,wd,wu,bu,wg,bg)
    da,dd,dg,ds=joint(address,dd,gate,ss,dy,epsilon,gate_layout='major')
    dx,dwd,dwu,dbu,dwg,dbg=project_reverse(xx,wd,wu,bu,wg,bg,pre,hidden,gate,da,dg)
    values=(dx,dd,ds,dwd,dwu,dbu,dwg,dbg)
    for i,(ref,value) in enumerate(zip(refs[9:],values)):
      if i<2:ref[...]=value
      else:
        @pl.when(pl.program_id(1)==0)
        def init():ref[...]=jnp.zeros(ref.shape,ref.dtype)
        ref[...]=ref[...]+(value[None,:] if value.ndim==1 else value)
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=out_specs,
      out_shape=tuple(out_shapes),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_projected_mlp_write_backward')(*args,dm)
  grads=tuple(z.transpose(0,2,1) if i==0 else z.transpose(0,2,3,1) if i==1 else
               jnp.sum(z,axis=0).reshape(args[i].shape).astype(args[i].dtype) for i,z in enumerate(grads))
  return (dm.transpose(0,2,3,1),*grads)


fused.defvjp(fwd,bwd)


def projected_write(m,x,d,s,down,up,ub,wg,gb,epsilon=1e-6,*,interpret=False,forward_tile=128,reverse_tile=32):
  def local(m,x,d,*weights):
    ft=min(forward_tile,m.shape[1]);rt=min(reverse_tile,m.shape[1])
    if m.shape[1]%ft or m.shape[1]%rt:raise ValueError('Write chunk must divide sequence length')
    out=fused(m.transpose(0,2,3,1),x.transpose(0,2,1),d.transpose(0,2,3,1),*weights,
              epsilon,interpret,ft,rt)
    return out.transpose(0,3,1,2)
  return _map_batch(local,(m,x,d,s,down,up,ub,wg,gb),(True,)*3+(False,)*6)

"""Write pullback with a token-major ABI independent of the forward kernel."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def norm(x,epsilon):
  f=x.astype(jnp.float32)
  inv=jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+epsilon)
  return (f*inv).astype(x.dtype),inv


def norm_backward(x,u,inv):
  f=x.astype(jnp.float32);u=u.astype(jnp.float32)
  return ((u-f*jnp.mean(u*f,axis=-1,keepdims=True)*inv*inv)*inv).astype(x.dtype)


def joint(address,data,gate,static_key,dy,epsilon):
  t,h,k=address.shape;v=data.shape[-1]
  an,ai=norm(address,epsilon);dn,di=norm(data,epsilon)
  gated=(an*gate[:,:,None]).astype(address.dtype)
  top=jnp.concatenate((jnp.zeros((t,k,k),dy.dtype),dy),axis=2)
  bottom=jnp.concatenate((dy.swapaxes(1,2),jnp.zeros((t,v,v),dy.dtype)),axis=2)
  square=jnp.concatenate((top,bottom),axis=1)
  dynamic=jnp.concatenate((gated,dn),axis=2)
  static_data=jnp.concatenate((jnp.broadcast_to(static_key[None,:,:],(t,h,k)),jnp.zeros((t,h,v),data.dtype)),axis=2)
  static_input=jnp.concatenate((jnp.zeros((t,h,k),data.dtype),data),axis=2)
  left=jnp.concatenate((dynamic,static_data,static_input),axis=1)
  product=jnp.einsum('thd,tdc->thc',left,square,preferred_element_type=jnp.float32)
  ua=product[:,:h,:k].astype(address.dtype)
  ud=product[:,:h,k:].astype(data.dtype)
  sd=product[:,h:2*h,k:].astype(data.dtype)
  ds=jnp.sum(product[:,2*h:,:k],axis=0).astype(static_key.dtype)
  gg=jnp.sum((ua*an).astype(gate.dtype),axis=2).astype(gate.dtype)
  ga=norm_backward(address,(ua*gate[:,:,None]).astype(address.dtype),ai)
  gd=(norm_backward(data,ud,di)+sd).astype(data.dtype)
  return ga,gd,gg,ds


def backward(a,d,g,s,dy,epsilon,interpret=False,tile=64):
  b,t,h,k=a.shape;v=d.shape[-1]
  tile=min(tile,t)
  if t%tile:raise ValueError('Reverse tile must divide token count')
  spec=lambda shape:pl.BlockSpec((None,tile)+shape,lambda b,i:(b,i)+(0,)*len(shape))
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    da[...],dd[...],dg[...],ds[...]=joint(a[...],d[...],g[...],s[...],dy[...],epsilon)
  specs=[spec((h,k)),spec((h,v)),spec((h,)),pl.BlockSpec(s.shape,lambda b,i:(0,0)),spec((k,v))]
  outputs=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (a,d,g)]
  outputs.append(jax.ShapeDtypeStruct((b,t//tile)+s.shape,s.dtype))
  da,dd,dg,ds=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=specs,
      out_specs=specs[:3]+[pl.BlockSpec((None,None)+s.shape,lambda b,i:(b,i,0,0))],
      out_shape=outputs,interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_write_reverse_major')(a,d,g,s,dy)
  return da,dd,dg,jnp.sum(ds.astype(jnp.float32),axis=(0,1)).astype(s.dtype)

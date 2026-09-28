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


def _tile(matrix,address,data,gate,static_key,epsilon,write_mode="original"):
  if write_mode != "original":
    from layers.rmt_pallas_rankh_write import forward
    return forward(matrix,address,data,gate,static_key,epsilon,write_mode)
  # Shapes K,V,T; H,K,T; H,V,T; H,T. T stays in SIMD lanes.
  h,k,t=address.shape
  v=data.shape[1]
  a=(_norm(address,epsilon)*gate.astype(jnp.float32)[:,None,:].astype(gate.dtype)).astype(jnp.float32)
  d=_norm(data,epsilon).astype(jnp.float32)
  vp=v if os.environ.get('RMT_PALLAS_MINOR_PAD_V')=='0' else ((v+127)//128)*128
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


def _norm_backward(x, cotangent, epsilon):
  """Derivative of FP32 RMS normalization, retaining the BF16 cast boundary."""
  f=x.astype(jnp.float32)
  u=cotangent.astype(jnp.float32)
  inv=jax.lax.rsqrt(jnp.mean(f*f,axis=-2,keepdims=True)+epsilon)
  correction=jnp.mean(u*f,axis=-2,keepdims=True)*inv*inv
  return ((u-f*correction)*inv).astype(x.dtype)


def _analytic_backward(address,data,gate,static_key,dy,epsilon,method="analytic"):
  """Shared contractions plus explicit RMS/gate derivatives; no AD of _tile."""
  h,k,t=address.shape
  v=data.shape[1]
  vp=((v+127)//128)*128
  yp=jnp.concatenate((dy,jnp.zeros((k,vp-v,t),dy.dtype)),axis=1)
  dp=jnp.concatenate((data,jnp.zeros((h,vp-v,t),data.dtype)),axis=1)
  static_dd=jnp.dot(static_key,yp.reshape(k,vp*t),
                   preferred_element_type=jnp.float32).reshape(h,vp,t)[:,:v,:].astype(data.dtype)
  ds=jnp.dot(dp.reshape(h,vp*t),yp.reshape(k,vp*t).T,
             preferred_element_type=jnp.float32).astype(static_key.dtype)
  ga,gd,gg=_dynamic_backward(address,data,gate,dy,epsilon,method)
  return ga,(gd+static_dd).astype(data.dtype),gg,ds


def _batch_dot_chunks(lhs,rhs,equation,chunk):
  """Bound temporary MXU packing storage while retaining 128-token DMA tiles."""
  return jnp.concatenate([jnp.einsum(equation,lhs[i:i+chunk],rhs[i:i+chunk],
      preferred_element_type=jnp.float32) for i in range(0,lhs.shape[0],chunk)],axis=0)


def _batch_dot_loop(lhs,rhs,equation,chunk):
  if equation=='tkv,thv->thk':shape=(lhs.shape[0],rhs.shape[1],lhs.shape[1])
  elif equation=='tkv,thk->thv':shape=(lhs.shape[0],rhs.shape[1],lhs.shape[2])
  else:shape=(lhs.shape[0],lhs.shape[1],rhs.shape[2])
  count=(lhs.shape[0]+chunk-1)//chunk
  chunk=min(chunk,lhs.shape[0])
  def body(i,result):
    l=jax.lax.dynamic_slice_in_dim(lhs,i*chunk,chunk,axis=0)
    r=jax.lax.dynamic_slice_in_dim(rhs,i*chunk,chunk,axis=0)
    out=jnp.einsum(equation,l,r,preferred_element_type=jnp.float32)
    return jax.lax.dynamic_update_slice_in_dim(result,out,i*chunk,axis=0)
  return jax.lax.fori_loop(0,count,body,jnp.zeros(shape,jnp.float32),unroll=False)


def _joint_backward(address,data,gate,static_key,dy,epsilon,chunk=128):
  """Four write contractions in one per-token 128-wide MXU operation."""
  h,k,t=address.shape;v=data.shape[1]
  an=_norm(address,epsilon);dn=_norm(data,epsilon)
  gated=(an*gate[:,None,:]).astype(address.dtype)
  y=dy.transpose(2,0,1)
  top=jnp.concatenate((jnp.zeros((t,k,k),dy.dtype),y),axis=2)
  bottom=jnp.concatenate((y.swapaxes(1,2),jnp.zeros((t,v,v),dy.dtype)),axis=2)
  square=jnp.concatenate((top,bottom),axis=1)
  dynamic=jnp.concatenate((gated,dn),axis=1).transpose(2,0,1)
  static_data=jnp.concatenate((jnp.broadcast_to(static_key[None,:,:],(t,h,k)),jnp.zeros((t,h,v),data.dtype)),axis=2)
  static_key_input=jnp.concatenate((jnp.zeros((t,h,k),data.dtype),data.transpose(2,0,1)),axis=2)
  left=jnp.concatenate((dynamic,static_data,static_key_input),axis=1)
  product=_batch_dot_chunks(left,square,'thd,tdc->thc',chunk).transpose(1,2,0)
  ua=product[:h,:k,:].astype(address.dtype)
  ud=product[:h,k:,:].astype(data.dtype)
  sd=product[h:2*h,k:,:].astype(data.dtype)
  ds=jnp.sum(product[2*h:,:k,:],axis=2).astype(static_key.dtype)
  gg=jax.lax.reduce_sum((ua*an).astype(gate.dtype),axes=(1,)).astype(gate.dtype)
  ga=_norm_backward(address,(ua*gate[:,None,:]).astype(address.dtype),epsilon)
  gd=(_norm_backward(data,ud,epsilon)+sd).astype(data.dtype)
  return ga,gd,gg,ds


def _dynamic_backward(address,data,gate,dy,epsilon,method="analytic"):
  h=address.shape[0]
  an=_norm(address,epsilon)
  dn=_norm(data,epsilon)
  gated=(an*gate[:,None,:]).astype(address.dtype)
  yf=dy.astype(jnp.float32)
  ua=[];ud=[]
  if method=='symmetric':
    # Both pullbacks in one padded MXU product: [A_gate,D_norm] @ [[0,Y],[Y.T,0]].
    # K+V=123 fits one 128-wide hardware tile; two separate products each need one.
    h,k,t=address.shape;v=data.shape[1]
    y=dy.transpose(2,0,1)
    top=jnp.concatenate((jnp.zeros((t,k,k),dy.dtype),y),axis=2)
    bottom=jnp.concatenate((y.swapaxes(1,2),jnp.zeros((t,v,v),dy.dtype)),axis=2)
    square=jnp.concatenate((top,bottom),axis=1)
    left=jnp.concatenate((gated,dn),axis=1).transpose(2,0,1)
    both=jnp.einsum('thd,tdc->thc',left,square,preferred_element_type=jnp.float32).transpose(1,2,0)
    ua=both[:,:k,:].astype(address.dtype)
    ud=both[:,k:,:].astype(data.dtype)
  elif method in ('batched','batched32','batched_loop'):
    # Use independent token contractions on MXU; normalize/gate in token lanes.
    chunk=32 if method in ('batched32','batched_loop') else dy.shape[-1]
    contract=_batch_dot_loop if method=='batched_loop' else _batch_dot_chunks
    ua=contract(dy.transpose(2,0,1),dn.transpose(2,0,1),'tkv,thv->thk',chunk).transpose(1,2,0).astype(address.dtype)
    ud=contract(dy.transpose(2,0,1),gated.transpose(2,0,1),'tkv,thk->thv',chunk).transpose(1,2,0).astype(data.dtype)
  else:
    for head in range(h):
      d=dn[head];ag=gated[head]
      ua.append(jnp.sum(yf*d.astype(jnp.float32)[None,:,:],axis=1).astype(address.dtype))
      ud.append(jnp.sum(yf*ag.astype(jnp.float32)[:,None,:],axis=0).astype(data.dtype))
    ua,ud=jnp.stack(ua),jnp.stack(ud)
  gg=jax.lax.reduce_sum((ua*an).astype(gate.dtype),axes=(1,)).astype(gate.dtype)
  ga=_norm_backward(address,(ua*gate[:,None,:]).astype(address.dtype),epsilon)
  gd=_norm_backward(data,ud,epsilon)
  return ga,gd,gg


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


@partial(jax.custom_vjp,nondiff_argnums=(5,6,7,8,9))
def _write(m,a,d,g,s,epsilon,interpret,tile,key_contiguous,backward):return _call(m,a,d,g,s,epsilon,interpret,tile,key_contiguous)


def _fwd(m,a,d,g,s,epsilon,interpret,tile,key_contiguous,backward):
  return _call(m,a,d,g,s,epsilon,interpret,tile,key_contiguous),(a,d,g,s)


def _bwd(epsilon,interpret,tile,key_contiguous,backward,args,dy):
  a,d,g,s=args
  b,_,_,t=dy.shape;h=a.shape[1]
  k,v=(dy.shape[2],dy.shape[1]) if key_contiguous else dy.shape[1:3]
  if backward in ('joint_major','joint_major32','joint_major128','joint_major256','joint_major_direct','joint_major_direct128'):
    if key_contiguous:raise ValueError('Joint reverse expects token-minor forward')
    from layers.rmt_pallas_write_reverse import backward as reverse
    da,dd,dg,ds=reverse(a.transpose(0,3,1,2),d.transpose(0,3,1,2),g.transpose(0,2,1),s,
                       dy.transpose(0,3,1,2),epsilon,interpret,tile=(int(backward[len("joint_major_direct"):] or 64) if backward.startswith("joint_major_direct") else int(backward[len("joint_major"):] or 64)),
                       gate_layout="major" if backward.startswith("joint_major_direct") else "minor")
    return dy,da.transpose(0,2,3,1),dd.transpose(0,2,3,1),dg.transpose(0,2,1),ds
  if backward in ('hybrid','hybrid_batched','split4','split8'):
    if key_contiguous:raise ValueError('Hybrid backward requires token-minor layout')
    # Shared static contractions use full-token GEMMs, avoiding one poorly
    # utilized H-by-K parameter-gradient dot (and V padding) per token tile.
    static_dd=jnp.einsum('hk,bkvt->bhvt',s,dy).astype(d.dtype)
    ds=jnp.einsum('bhvt,bkvt->hk',d,dy,preferred_element_type=jnp.float32).astype(s.dtype)
    def dynamic_kernel(a,d,g,dy,sd,da,dd,dg):
      ga,gd,gg=_dynamic_backward(a[...],d[...],g[...],dy[...],epsilon,"batched" if backward=="hybrid_batched" else "analytic")
      da[...],dd[...],dg[...]=ga,(gd+sd[...]).astype(d.dtype),gg
    if backward.startswith('split'):
      hg=int(backward[5:])
      tile=min(t,256)
      head_spec=lambda n:pl.BlockSpec((None,hg,n,tile),lambda b,i,j:(b,j,0,i))
      gate_spec=pl.BlockSpec((None,hg,tile),lambda b,i,j:(b,j,i))
      matrix_spec=pl.BlockSpec((None,k,v,tile),lambda b,i,j:(b,0,0,i))
      specs=[head_spec(k),head_spec(v),gate_spec,matrix_spec,head_spec(v)]
      grid=(b,t//tile,h//hg)
    else:
      specs=[_spec((h,k),tile),_spec((h,v),tile),_spec((h,),tile),_spec((k,v),tile),_spec((h,v),tile)]
      grid=(b,t//tile)
    da,dd,dg=pl.pallas_call(dynamic_kernel,grid=grid,in_specs=specs,
        out_specs=specs[:3],out_shape=[jax.ShapeDtypeStruct(x.shape,x.dtype) for x in (a,d,g)],
        interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)*len(grid)),
        name='rmt_token_minor_dynamic_write_backward')(a,d,g,dy,static_dd)
    return dy,da,dd,dg,ds
  def kernel(a,d,g,s,dy,da,dd,dg,ds):
    if backward in ('joint','joint32'):
      ga,gd,gg,gs=_joint_backward(a[...],d[...],g[...],s[...],dy[...],epsilon,32 if backward=='joint32' else tile)
    elif backward in ('analytic','batched','batched32','batched_loop','symmetric'):
      ga,gd,gg,gs=_analytic_backward(a[...],d[...].swapaxes(0,1) if key_contiguous else d[...],
                                   g[...],s[...],dy[...].swapaxes(0,1) if key_contiguous else dy[...],epsilon,backward)
    else:
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


def write_residual(matrix,address,data,gate,static_key,epsilon=1e-6,*,interpret=False,tile=128,key_contiguous=False,backward="autodiff"):
  if backward not in ("autodiff","analytic","hybrid","hybrid_batched","batched","batched32","batched_loop","symmetric","joint","joint32","joint_major","joint_major32","joint_major128","joint_major256","joint_major_direct","joint_major_direct128","split4","split8"):raise ValueError(f"Unknown write backward: {backward}")
  unbatched=matrix.ndim==3
  if unbatched:
    matrix,address,data,gate=(x[None] for x in (matrix,address,data,gate))
  def local(m,a,d,g,s):
    token_tile=min(tile,m.shape[1])
    if m.shape[1]%token_tile:raise ValueError('Token tile must divide sequence length')
    order=(0,3,2,1) if key_contiguous else (0,2,3,1)
    result=_write(m.transpose(order),a.transpose(0,2,3,1),d.transpose(order),
                  g.transpose(0,2,1),s,epsilon,interpret,token_tile,key_contiguous,backward)
    return result.transpose((0,3,2,1) if key_contiguous else (0,3,1,2))
  out=_map_batch(local,(matrix,address,data,gate,static_key),(True,True,True,True,False))
  return out[0] if unbatched else out

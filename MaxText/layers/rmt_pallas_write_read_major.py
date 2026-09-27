"""Complete write/read pullback with a native, independently sized token chunk.

All matrix/read/projection/write derivatives stay in one program. Token-major
storage lets 32/64-token programs actually reduce VMEM instead of padding a
token-minor SIMD lane dimension back to 128.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas_write_reverse import norm, norm_backward, joint
from layers.rmt_pallas_minor import _norm, _norm_backward


def contract(w,x):
  q,n,v=x.shape;vp=((v+127)//128)*128
  flat=jnp.pad(x,((0,0),(0,0),(0,vp-v))).transpose(1,0,2).reshape(n,q*vp)
  return jnp.dot(w,flat,preferred_element_type=jnp.float32).reshape(w.shape[0],q,vp).transpose(1,0,2)[...,:v].astype(x.dtype)


def weight_grad(left,right):
  q,n,v=left.shape;vp=((v+127)//128)*128
  l=jnp.pad(left,((0,0),(0,0),(0,vp-v))).transpose(1,0,2).reshape(n,q*vp)
  r=jnp.pad(right,((0,0),(0,0),(0,vp-v))).transpose(1,0,2).reshape(right.shape[1],q*vp)
  return jnp.dot(l,r.T,preferred_element_type=jnp.float32)


def matmul(a,b):
  return jnp.matmul(a,b,preferred_element_type=jnp.float32).astype(a.dtype)


def reverse(m,a,d,g,s,r,c,scale,wk,wg,bias,dm,dy,dx,epsilon,read_epsilon):
  q,k,v=m.shape;h=d.shape[1];rank=c.shape[1]
  vp=((v+127)//128)*128
  raw=jnp.pad(m[:,:h,:],((0,0),(0,0),(0,vp-v))).reshape(q,h*vp)
  f=raw.astype(jnp.float32)
  inv=jax.lax.rsqrt(jnp.sum(f*f,axis=-1,keepdims=True)/(h*v)+epsilon)
  normalized=(f*inv).astype(m.dtype)
  x=(normalized*scale[None,:]).astype(m.dtype)
  # Only this small key uses token-minor layout. The large M and its gradient
  # stay token-major, so shrinking q really shrinks their VMEM footprint.
  projection=matmul(x,jnp.concatenate((wk,wg.T),axis=1))
  key=projection[:,:h*rank].T.reshape(h,rank,q)
  logits=(projection[:,h*rank:]+bias[None,:]).astype(m.dtype)
  gate=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(m.dtype)
  compressed=contract(c.T,m[:,h:,:])
  kn=_norm(key,read_epsilon)
  read=jnp.zeros((q,h,v),jnp.float32)
  for i in range(rank):
    ki=jax.lax.slice_in_dim(kn,i,i+1,axis=1).reshape(h,q).T.astype(jnp.float32)
    ci=jax.lax.slice_in_dim(compressed,i,i+1,axis=1).reshape(q,v).astype(jnp.float32)
    read=read+ki[:,:,None]*ci[:,None,:]
  read=read.astype(m.dtype)
  dr=weight_grad(m,dy)
  dm=(dm+contract(r,dy)).astype(m.dtype)
  dgate=(.2*jax.lax.reduce_sum((dy*read).astype(m.dtype),axes=(2,))).astype(m.dtype)
  dread=(dy*(.2*gate).astype(jnp.float32)[:,:,None].astype(m.dtype)).astype(m.dtype)
  dc=[];dkn=[]
  for i in range(rank):
    ki=jax.lax.slice_in_dim(kn,i,i+1,axis=1).reshape(h,q).T.astype(jnp.float32)
    ci=jax.lax.slice_in_dim(compressed,i,i+1,axis=1).reshape(q,v).astype(jnp.float32)
    dc.append(jnp.sum(dread.astype(jnp.float32)*ki[:,:,None],axis=1).astype(m.dtype))
    dkn.append(jnp.sum(dread.astype(jnp.float32)*ci[:,None,:],axis=2).T.astype(m.dtype))
  dc=jnp.stack([x.astype(jnp.float32) for x in dc],axis=1).astype(m.dtype)
  dkn=jnp.stack([x.astype(jnp.float32) for x in dkn],axis=1).astype(m.dtype)
  dcompression=weight_grad(m[:,h:,:],dc)
  dmtail=contract(c,dc)
  dk=_norm_backward(key,dkn,read_epsilon).reshape(h*rank,q).T
  dg=(dgate*(gate*(1-gate))).astype(m.dtype)
  dp=jnp.concatenate((dk,dg),axis=1)
  dw=jnp.dot(x.T,dp,preferred_element_type=jnp.float32)
  dwk=dw[:,:h*rank];dwg=dw[:,h*rank:].T
  db=jnp.sum(dg.astype(jnp.float32).T,axis=1)[None,:]
  dx=(dx+matmul(dp,jnp.concatenate((wk.T,wg),axis=0))).astype(m.dtype)
  dscale=jnp.sum((dx*normalized).astype(m.dtype).astype(jnp.float32).T,axis=1)[None,:]
  u=(dx*scale[None,:]).astype(m.dtype).astype(jnp.float32)
  draw=((u-f*jnp.sum(u*f,axis=-1,keepdims=True)/(h*v)*inv*inv)*inv).astype(m.dtype)
  draw=draw.reshape(q,h,vp)[...,:v]
  dm=(dm+jnp.concatenate((draw,dmtail),axis=1)).astype(m.dtype)
  da,dd,dg_write,ds=joint(a,d,g,s,dm,epsilon,gate_layout='major')
  return dm,da,dd,dg_write,ds,dr,dcompression,dscale,dwk,dwg,db


def backward(args,cotangents,epsilon,read_epsilon,interpret,tile):
  args=tuple(x.transpose(0,3,1,2) if i<3 else x.transpose(0,2,1) if i==3 else x
             for i,x in enumerate(args))
  cotangents=tuple(x.transpose(0,3,1,2) if i<2 else x.transpose(0,2,1)
                   for i,x in enumerate(cotangents))
  b,t=args[0].shape[:2];tile=min(tile,t)
  h,v=args[2].shape[2:];vp=((v+127)//128)*128
  def pad_features(x,axis):
    shape=x.shape[:axis]+(h,v)+x.shape[axis+1:]
    pads=[(0,0)]*len(shape);pads[axis+1]=(0,vp-v)
    return jnp.pad(x.reshape(shape),pads).reshape(x.shape[:axis]+(h*vp,)+x.shape[axis+1:])
  def crop_features(x,axis):
    shape=x.shape[:axis]+(h,vp)+x.shape[axis+1:]
    return jax.lax.slice_in_dim(x.reshape(shape),0,v,axis=axis+1).reshape(x.shape[:axis]+(h*v,)+x.shape[axis+1:])
  args=list(args)
  for i,axis in ((7,0),(8,0),(9,1)):args[i]=pad_features(args[i],axis)
  cotangents=(*cotangents[:2],pad_features(cotangents[2],2))
  if t%tile:raise ValueError('Reverse token chunk must divide sequence length')
  def spec(x):return pl.BlockSpec((None,tile)+x.shape[2:],lambda b,i:(b,i)+(0,)*(x.ndim-2))
  inp=[spec(x) if i<4 else pl.BlockSpec(x.shape,lambda b,i,ndim=x.ndim:(0,)*ndim)
       for i,x in enumerate(args)]+[spec(x) for x in cotangents]
  shapes=[(1,)+x.shape if x.ndim==1 else x.shape for x in args[4:]]
  outspec=inp[:4]+[pl.BlockSpec((None,)+shape,lambda b,i,n=len(shape):(b,)+(0,)*n)
                    for shape in shapes]
  outshape=[jax.ShapeDtypeStruct(x.shape,x.dtype) if i<4 else
            jax.ShapeDtypeStruct((b,)+shapes[i-4],jnp.float32) for i,x in enumerate(args)]
  def kernel(*refs):
    grads=reverse(*(r[...] for r in refs[:14]),epsilon,read_epsilon)
    for i,(ref,value) in enumerate(zip(refs[14:],grads)):
      if i<4:ref[...]=value
      else:
        @pl.when(pl.program_id(1)==0)
        def init():ref[...]=jnp.zeros(ref.shape,ref.dtype)
        ref[...]=ref[...]+(value[None,:] if value.ndim==1 else value)
  gradients=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outspec,
      out_shape=tuple(outshape),interpret=interpret,
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_fused_write_read_reverse_major')(*args,*cotangents)
  gradients=[x.transpose(0,2,3,1) if i<3 else x.transpose(0,2,1) if i==3 else
             jnp.sum(x,axis=0).reshape(args[i].shape).astype(args[i].dtype)
             for i,x in enumerate(gradients)]
  for i,axis in ((7,0),(8,0),(9,1)):gradients[i]=crop_features(gradients[i],axis)
  return tuple(gradients)

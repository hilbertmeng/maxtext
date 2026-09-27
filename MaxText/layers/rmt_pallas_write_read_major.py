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


def pack_proxy(x):
  q,h,v=x.shape;vp=((v+127)//128)*128
  if vp!=128:raise ValueError('Compact proxy currently requires V <= 128')
  padded=jnp.pad(x,((0,0),(0,0),(0,128-v))).astype(jnp.float32)
  lane=jnp.arange(128);blocks=[]
  for start in range(0,h*v,128):
    index=start+lane
    out=jnp.zeros((q,128),jnp.float32)
    for head in range(start//v,min(h,(start+127)//v+1)):
      source=jax.lax.slice_in_dim(padded,head,head+1,axis=1).reshape(q,128)
      valid=(index>=head*v)&(index<(head+1)*v)
      take=jnp.where(valid,index-head*v,0)
      part=jnp.take_along_axis(source,jnp.broadcast_to(take[None,:],source.shape),axis=1)
      out=out+jnp.where(valid[None,:],part,0)
    blocks.append(out.astype(x.dtype))
  return jnp.concatenate(blocks,axis=1)[:,:h*v]


def unpack_proxy(x,h,v):
  q=x.shape[0];vp=((v+127)//128)*128
  if vp!=128:raise ValueError('Compact proxy currently requires V <= 128')
  padded=jnp.pad(x,((0,0),(0,((h*v+127)//128)*128-h*v)))
  lane=jnp.arange(128);heads=[]
  for head in range(h):
    index=head*v+lane
    out=jnp.zeros((q,128),jnp.float32)
    for block in range((head*v)//128,((head+1)*v-1)//128+1):
      source=jax.lax.slice_in_dim(padded,block*128,(block+1)*128,axis=1).astype(jnp.float32)
      valid=(lane<v)&(index>=block*128)&(index<(block+1)*128)
      take=jnp.where(valid,index-block*128,0)
      part=jnp.take_along_axis(source,jnp.broadcast_to(take[None,:],source.shape),axis=1)
      out=out+jnp.where(valid[None,:],part,0)
    heads.append(out)
  return jnp.stack(heads,axis=1)[...,:v].astype(x.dtype)


def reverse(m,a,d,g,s,r,c,scale,wk,wg,bias,dm,dy,dx,epsilon,read_epsilon,shared_sink=None):
  # Flush completed shared gradients before entering the much larger write
  # pullback. Returning every partial together unnecessarily extends their
  # live ranges across the joint MXU contraction.
  def finish(index,value):
    if shared_sink is None:return value
    shared_sink(index,value)
    return None
  q,k,v=m.shape;h=d.shape[1];rank=c.shape[1]
  vp=((v+127)//128)*128
  raw=pack_proxy(m[:,:h,:])
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
  dr=finish(5,weight_grad(m,dy))
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
  dcompression=finish(6,weight_grad(m[:,h:,:],dc))
  dmtail=contract(c,dc)
  dk=_norm_backward(key,dkn,read_epsilon).reshape(h*rank,q).T
  dg=(dgate*(gate*(1-gate))).astype(m.dtype)
  dp=jnp.concatenate((dk,dg),axis=1)
  dw=jnp.dot(x.T,dp,preferred_element_type=jnp.float32)
  dwk=finish(8,dw[:,:h*rank]);dwg=finish(9,dw[:,h*rank:].T)
  db=finish(10,jnp.sum(dg.astype(jnp.float32).T,axis=1)[None,:])
  dx=(dx+matmul(dp,jnp.concatenate((wk.T,wg),axis=0))).astype(m.dtype)
  dscale=finish(7,jnp.sum((dx*normalized).astype(m.dtype).astype(jnp.float32).T,axis=1)[None,:])
  u=(dx*scale[None,:]).astype(m.dtype).astype(jnp.float32)
  draw=((u-f*jnp.sum(u*f,axis=-1,keepdims=True)/(h*v)*inv*inv)*inv).astype(m.dtype)
  draw=unpack_proxy(draw,h,v)
  dm=(dm+jnp.concatenate((draw,dmtail),axis=1)).astype(m.dtype)
  da,dd,dg_write,ds=joint(a,d,g,s,dm,epsilon,gate_layout='major')
  ds=finish(4,ds)
  return dm,da,dd,dg_write,ds,dr,dcompression,dscale,dwk,dwg,db


def backward(args,cotangents,epsilon,read_epsilon,interpret,tile,compute_tile=0):
  args=tuple(x.transpose(0,3,1,2) if i<3 else x.transpose(0,2,1) if i==3 else x
             for i,x in enumerate(args))
  cotangents=tuple(x.transpose(0,3,1,2) if i<2 else x.transpose(0,2,1)
                   for i,x in enumerate(cotangents))
  b,t=args[0].shape[:2];tile=min(tile,t)
  compute_tile=min(compute_tile or tile,tile)
  if t%tile:raise ValueError('Reverse token chunk must divide sequence length')
  if tile%compute_tile:raise ValueError('Compute chunk must divide DMA tile')
  def spec(x):return pl.BlockSpec((None,tile)+x.shape[2:],lambda b,i:(b,i)+(0,)*(x.ndim-2))
  inp=[spec(x) if i<4 else pl.BlockSpec(x.shape,lambda b,i,ndim=x.ndim:(0,)*ndim)
       for i,x in enumerate(args)]+[spec(x) for x in cotangents]
  shapes=[(1,)+x.shape if x.ndim==1 else x.shape for x in args[4:]]
  outspec=inp[:4]+[pl.BlockSpec((None,)+shape,lambda b,i,n=len(shape):(b,)+(0,)*n)
                    for shape in shapes]
  outshape=[jax.ShapeDtypeStruct(x.shape,x.dtype) if i<4 else
            jax.ShapeDtypeStruct((b,)+shapes[i-4],jnp.float32) for i,x in enumerate(args)]
  def kernel(*refs):
    @pl.when(pl.program_id(1)==0)
    def init():
      for ref in refs[18:]:ref[...]=jnp.zeros(ref.shape,ref.dtype)
    def store(index,value):
      ref=refs[14+index]
      ref[...]=ref[...]+(value[None,:] if value.ndim==1 else value)
    def chunk(i,_):
      sl=pl.ds(i*compute_tile,compute_tile)
      values=[ref[sl,...] if j<4 or j>=11 else ref[...] for j,ref in enumerate(refs[:14])]
      grads=reverse(*values,epsilon,read_epsilon,shared_sink=store)
      for ref,value in zip(refs[14:18],grads[:4]):ref[sl,...]=value
    jax.lax.fori_loop(0,tile//compute_tile,chunk,None)
  gradients=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outspec,
      out_shape=tuple(outshape),interpret=interpret,input_output_aliases={11:0},
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_fused_write_read_reverse_major')(*args,*cotangents)
  gradients=[x.transpose(0,2,3,1) if i<3 else x.transpose(0,2,1) if i==3 else
             jnp.sum(x,axis=0).reshape(args[i].shape).astype(args[i].dtype)
             for i,x in enumerate(gradients)]
  return tuple(gradients)

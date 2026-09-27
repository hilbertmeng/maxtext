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
  n,v,t=x.shape;vp=((v+7)//8)*8
  padded=jnp.pad(x,((0,0),(0,vp-v),(0,0)))
  return jnp.dot(w,padded.reshape(n,vp*t),preferred_element_type=jnp.float32).reshape(w.shape[0],vp,t)[:,:v,:].astype(x.dtype)


def weight_grad(left,right):
  """Sum matching value/token axes; FP32 partials stay FP32 until global sum."""
  v,t=left.shape[-2:];vp=((v+7)//8)*8
  l=jnp.pad(left,((0,0),(0,vp-v),(0,0))).reshape(left.shape[0],vp*t)
  r=jnp.pad(right,((0,0),(0,vp-v),(0,0))).reshape(right.shape[0],vp*t)
  return jnp.dot(l,r.T,preferred_element_type=jnp.float32)


def write_reverse(a,d,g,s,dy,epsilon,write_refs):
  """Keep one head's VPU contraction live, rather than unrolling all heads."""
  h,k,t=a.shape;v=d.shape[1]
  static_dd=contract(s,dy)
  ds=weight_grad(d,dy).astype(s.dtype)
  yf=dy.astype(jnp.float32)
  def scoped(ga_ref,gd_ref,gg_ref,static_ref,gate_ref):
    static_ref[...]=static_dd
    gate_ref[...]=jnp.broadcast_to(g.astype(jnp.float32)[:,None,:],(h,8,t)).astype(g.dtype)
    def head(i,_):
      aa=write_refs[0][i,:,:];dd=write_refs[1][i,:,:];gg=gate_ref[i,0,:]
      an=_norm(aa,epsilon);dn=_norm(dd,epsilon)
      gated=(an*gg[None,:]).astype(a.dtype)
      ua=jnp.sum(yf*dn.astype(jnp.float32)[None,:,:],axis=1).astype(a.dtype)
      ud=jnp.sum(yf*gated.astype(jnp.float32)[:,None,:],axis=0).astype(d.dtype)
      ga_ref[i,:,:]=_norm_backward(aa,(ua*gg[None,:]).astype(a.dtype),epsilon)
      gd_ref[i,:,:]=(_norm_backward(dd,ud,epsilon)+static_ref[i,:,:]).astype(d.dtype)
      gate=jax.lax.reduce_sum((ua*an).astype(g.dtype),axes=(0,))
      gg_ref[i,:,:]=jnp.broadcast_to(gate[None,:],(8,t))
    jax.lax.fori_loop(0,h,head,None)
    return ga_ref[...],gd_ref[...],gg_ref[:,0,:]
  ga,gd,gg=pl.run_scoped(scoped,pltpu.VMEM(a.shape,a.dtype),pltpu.VMEM(d.shape,d.dtype),
                         pltpu.VMEM((h,8,t),g.dtype),pltpu.VMEM(d.shape,d.dtype),pltpu.VMEM((h,8,t),g.dtype))
  return ga,gd,gg,ds


def read_state(m,c,scale,wk,wg,bias,epsilon,read_epsilon):
  h=wg.shape[1];v=m.shape[1];t=m.shape[-1];rank=c.shape[1]
  raw=m[:h].reshape(h*v,t)
  normalized=_norm(raw,epsilon)
  x=(normalized.astype(jnp.float32)*scale.astype(jnp.float32)[:,None]).astype(m.dtype)
  key=jnp.dot(wk.T,x,preferred_element_type=jnp.float32).astype(m.dtype).reshape(h,rank,t)
  logits=jnp.dot(wg.T,x,preferred_element_type=jnp.float32).astype(m.dtype)
  # Mosaic's BF16 logistic lowering broadcasts an F32 constant as BF16.
  # Keep the original BF16 addition, evaluate sigmoid in F32, then round once.
  logits=(logits.astype(jnp.float32)+bias.astype(jnp.float32)[:,None]).astype(m.dtype)
  gate=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(m.dtype)
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


def store_partial(ref,value):
  if value.ndim==1:value=value[None,:]
  elif value.shape[0]>128 and value.shape[1]<128:value=value.T
  @pl.when(pl.program_id(1)==0)
  def init():ref[...]=jnp.zeros(ref.shape,ref.dtype)
  ref[...]=ref[...]+value


def reverse(m,a,d,g,s,r,c,scale,wk,wg,bias,dm,dy,dx,epsilon,read_epsilon,write_refs,out_refs):
  # m is the forward's updated matrix, so backward does not repeat the write.
  raw,normalized,x,key,gate,compressed,kn,read=read_state(m,c,scale,wk,wg,bias,epsilon,read_epsilon)
  h,v,t=read.shape;rank=c.shape[1]
  store_partial(out_refs[5],weight_grad(m,dy))
  dm=(dm+contract(r,dy)).astype(m.dtype)
  dgate=(.2*jax.lax.reduce_sum((dy*read).astype(dy.dtype),axes=(1,))).astype(dy.dtype)
  dread=(dy*(.2*gate)[:,None,:]).astype(dy.dtype)
  def read_reverse(kr,cr,dcr,dkr):
    kr[...]=kn.transpose(1,0,2);cr[...]=compressed
    def rank_step(i,_):
      ki=kr[i,:,:].astype(jnp.float32);ci=cr[i,:,:].astype(jnp.float32)
      dcr[i,:,:]=jnp.sum(dread.astype(jnp.float32)*ki[:,None,:],axis=0).astype(m.dtype)
      dkr[i,:,:]=jnp.sum(dread.astype(jnp.float32)*ci[None,:,:],axis=1).astype(m.dtype)
    jax.lax.fori_loop(0,rank,rank_step,None)
    return dcr[...],dkr[...].transpose(1,0,2)
  dc,dkn=pl.run_scoped(read_reverse,pltpu.VMEM((rank,h,t),m.dtype),
                       pltpu.VMEM((rank,v,t),m.dtype),pltpu.VMEM((rank,v,t),m.dtype),
                       pltpu.VMEM((rank,h,t),m.dtype))
  store_partial(out_refs[6],weight_grad(m[h:],dc))
  dmtail=contract(c,dc)
  dk=_norm_backward(key,dkn,read_epsilon).reshape(h*rank,t)
  dg=(dgate*(gate*(1-gate))).astype(m.dtype)
  store_partial(out_refs[8],jnp.dot(x,dk.T,preferred_element_type=jnp.float32))
  store_partial(out_refs[9],jnp.dot(x,dg.T,preferred_element_type=jnp.float32))
  store_partial(out_refs[10],jnp.sum(dg.astype(jnp.float32),axis=1))
  dx=(dx+jnp.dot(wk,dk,preferred_element_type=jnp.float32).astype(m.dtype)).astype(m.dtype)
  dx=(dx+jnp.dot(wg,dg,preferred_element_type=jnp.float32).astype(m.dtype)).astype(m.dtype)
  store_partial(out_refs[7],jnp.sum((dx*normalized).astype(m.dtype).astype(jnp.float32),axis=1))
  draw=_norm_backward(raw,(dx.astype(jnp.float32)*scale.astype(jnp.float32)[:,None]).astype(m.dtype),epsilon).reshape(h,v,t)
  dm=(dm+jnp.concatenate((draw,dmtail),axis=0)).astype(m.dtype)
  da,dd,dg_write,ds=write_reverse(a,d,g,s,dm,epsilon,write_refs)
  return dm,da,dd,dg_write,ds.astype(jnp.float32)


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
      input_output_aliases={0:0},
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
    gradients=reverse(*(x[...] for x in refs[:14]),epsilon,read_epsilon,refs[1:4],refs[14:])
    for ref,value in zip(refs[14:18],gradients[:4]):ref[...]=value
    store_partial(refs[18],gradients[4])
  inp=specs(args,tile)+[_spec(x.shape[1:-1],tile) for x in cotangents]
  transposed=[x.ndim==2 and x.shape[0]>128 and x.shape[1]<128 for x in args[4:]]
  partial_shapes=[(1,)+x.shape if x.ndim==1 else x.shape[::-1] if tr else x.shape
                  for x,tr in zip(args[4:],transposed)]
  outspec=inp[:4]+[pl.BlockSpec((None,)+shape,
      lambda b,i,ndim=len(shape):(b,)+(0,)*ndim) for shape in partial_shapes]
  outshape=[jax.ShapeDtypeStruct(x.shape,x.dtype) if i<4 else
            jax.ShapeDtypeStruct((b,)+partial_shapes[i-4],jnp.float32) for i,x in enumerate(args)]
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,out_specs=outspec,
      out_shape=tuple(outshape),interpret=interpret,
      input_output_aliases={11:0},
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_fused_write_mlp_read_backward')(*args,*cotangents)
  shared=[jnp.sum(x,axis=0) for x in grads[4:]]
  shared=[(x.T if tr else x).reshape(arg.shape).astype(arg.dtype)
          for x,tr,arg in zip(shared,transposed,args[4:])]
  return tuple(grads[:4])+tuple(shared)


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

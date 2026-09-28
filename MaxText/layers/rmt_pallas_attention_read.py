"""NoO attention read: static/C8 reads, all projections, QK/V routing and RoPE.

The reverse is analytic. No automatic differentiation is used inside a kernel.
Parameter gradients are accumulated on chip across token blocks.
"""
from functools import partial
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from layers.rmt_pallas import _map_batch
from layers.rmt_pallas_minor import _norm, _norm_backward, _spec
from layers.rmt_pallas_write_read import contract, weight_grad
from layers.rmt_pallas_v_read import _forward as v_forward, _reverse as v_reverse


def row(x,i):
  return jax.lax.slice_in_dim(x,i,i+1,axis=0).reshape(x.shape[1:])


def component(x,i):
  return jax.lax.slice_in_dim(x,i,i+1,axis=1).reshape(x.shape[0],x.shape[2])


def dynamic_basis_read(m,basis):
  # Four independent reductions; never materialize rank x C x V x token.
  rows=[]
  for i in range(basis.shape[0]):
    b=row(basis,i).astype(jnp.float32)
    rows.append(jnp.sum(m.astype(jnp.float32)*b[:,None,:],axis=0).astype(m.dtype))
  return jnp.stack(rows)


def qk_state(read,basis,mix,gate,epsilon):
  r,v,t=read.shape;h=mix.shape[0];c=basis.shape[1]
  key=jnp.zeros((h,c,t),jnp.float32);value=jnp.zeros((h,v,t),jnp.float32)
  for i in range(r):
    w=component(mix,i).astype(jnp.float32)
    key=key+w[:,None,:]*row(basis,i).astype(jnp.float32)[None,:,:]
    value=value+w[:,None,:]*row(read,i).astype(jnp.float32)[None,:,:]
  inv=jax.lax.rsqrt(jnp.mean(key*key,axis=1)+epsilon)
  value=value.astype(read.dtype)
  factor=(inv.astype(read.dtype)*(.2*gate)).astype(read.dtype)
  return key,inv,value,factor


def qk_reverse(m,basis,mix,gate,read,dy,epsilon):
  key,inv,value,factor=qk_state(read,basis,mix,gate,epsilon)
  dt=m.dtype;c=m.shape[0]
  dvalue=(dy*factor[:,None,:]).astype(dt)
  df=jax.lax.reduce_sum((dy*value).astype(dt),axes=(1,)).astype(dt)
  dg=(.2*(df*inv.astype(dt)).astype(dt)).astype(dt)
  di=(df*(.2*gate)).astype(dt).astype(jnp.float32)
  dkey=key*(-di*inv**3/c)[:,None,:]
  dm=jnp.zeros(m.shape,jnp.float32);db=[];dw=[]
  for i in range(basis.shape[0]):
    w=component(mix,i).astype(jnp.float32)
    dr=jnp.sum(dvalue.astype(jnp.float32)*w[:,None,:],axis=0).astype(dt)
    dw_read=jnp.sum(dvalue.astype(jnp.float32)*row(read,i).astype(jnp.float32)[None,:,:],axis=1).astype(dt)
    dw_norm=jnp.sum(dkey*row(basis,i).astype(jnp.float32)[None,:,:],axis=1).astype(dt)
    dw.append((dw_read+dw_norm).astype(dt))
    db_read=jnp.sum(m.astype(jnp.float32)*dr.astype(jnp.float32)[None,:,:],axis=1).astype(dt)
    db_norm=jnp.sum(dkey*w[:,None,:],axis=0).astype(dt)
    db.append((db_read+db_norm).astype(dt))
    dm=dm+row(basis,i).astype(jnp.float32)[:,None,:]*dr.astype(jnp.float32)[None,:,:]
  return dm.astype(dt),jnp.stack(db),jnp.stack(dw,axis=1),dg


def rope(x,positions,minimum,maximum,transpose=False):
  positions=positions.reshape(-1)
  n=x.shape[1]//2
  timescale=minimum*(maximum/minimum)**(jnp.arange(n,dtype=jnp.int32).astype(jnp.float32)/n)
  phase=positions.astype(jnp.float32)[None,:]/timescale[:,None]
  cs=jnp.cos(phase).astype(x.dtype)[None,:,:]
  sn=jnp.sin(phase).astype(x.dtype)[None,:,:]
  a,b=x[:,:n],x[:,n:]
  if transpose:return jnp.concatenate((a*cs+b*sn,b*cs-a*sn),axis=1).astype(x.dtype)
  return jnp.concatenate((a*cs-b*sn,b*cs+a*sn),axis=1).astype(x.dtype)


def state(m,s,c,scale,w,bb,gb,positions,epsilon,re,rd,rmin,rmax):
  k,v,t=m.shape;h=k//3;rank=4
  raw=m[:h].reshape(h*v,t);normalized=_norm(raw,epsilon)
  x=(normalized.astype(jnp.float32)*scale.astype(jnp.float32)[:,None]).astype(m.dtype)
  proj=jnp.dot(w.T,x,preferred_element_type=jnp.float32).astype(m.dtype)
  z=rank*(k-h)
  basis=(proj[:z].reshape(rank,k-h,t).astype(jnp.float32)+bb.astype(jnp.float32)[:,:,None]).astype(m.dtype)
  mix=proj[z:z+2*h*rank].reshape(2*h,rank,t);z+=2*h*rank
  qkg=jax.nn.sigmoid((proj[z:z+2*h]+jax.lax.slice_in_dim(gb.astype(jnp.float32),0,2*h).reshape(2*h,1)).astype(m.dtype).astype(jnp.float32)).astype(m.dtype);z+=2*h
  vk=proj[z:z+h*8].reshape(h,8,t);z+=h*8
  vg=jax.nn.sigmoid((proj[z:z+h]+jax.lax.slice_in_dim(gb.astype(jnp.float32),2*h,3*h).reshape(h,1)).astype(m.dtype).astype(jnp.float32)).astype(m.dtype);z+=h
  rp=proj[z:].reshape(2*h,rd,t)
  joined=jnp.concatenate((s,jnp.pad(c.T,((0,0),(h,0)))),axis=0)
  reads=contract(joined,m);static=reads[:3*h];compressed=reads[3*h:]
  br=dynamic_basis_read(m[h:],basis)
  _,_,qk_value,qk_factor=qk_state(br,basis,mix,qkg,re)
  qk=(static[:2*h]+(qk_value*qk_factor[:,None,:]).astype(m.dtype)).astype(m.dtype)
  qk=jnp.concatenate((qk[:,:v-rd,:],rope(rp,positions,rmin,rmax)),axis=1)
  query=(qk[:h]/math.sqrt(v)).astype(m.dtype)
  key=qk[h:]
  value=(static[2*h:]+v_forward(compressed,vk,vg,re)).astype(m.dtype)
  out=jnp.concatenate((query,key,value),axis=0)
  return out,x,(raw,normalized,x,basis,mix,qkg,vk,vg,compressed,br)


def reverse(m,s,c,scale,w,bb,gb,positions,dy,dx,epsilon,re,rd,rmin,rmax,sink):
  _,_,st=state(m,s,c,scale,w,bb,gb,positions,epsilon,re,rd,rmin,rmax)
  raw,normalized,x,basis,mix,qkg,vk,vg,compressed,br=st
  k,v,t=m.shape;h=k//3;dt=m.dtype
  dqk=jnp.concatenate(((dy[:h]/math.sqrt(v)).astype(dt),dy[h:2*h]),axis=0)
  drp=rope(dqk[:,v-rd:,:],positions,rmin,rmax,transpose=True).reshape(2*h*rd,t)
  dqk=jnp.pad(dqk[:,:v-rd,:],((0,0),(0,rd),(0,0)))
  ds=weight_grad(jnp.concatenate((dqk,dy[2*h:]),axis=0),m)
  sink(1,ds)
  dm=contract(s.T,jnp.concatenate((dqk,dy[2*h:]),axis=0))
  dc,dvk,dvg=v_reverse(compressed,vk,vg,dy[2*h:],re)
  sink(2,weight_grad(m[h:],dc))
  dmtail=contract(c,dc)
  qm,db,dmix,dqg=qk_reverse(m[h:],basis,mix,qkg,br,dqk,re)
  dmtail=(dmtail+qm).astype(dt)
  sink(5,jnp.sum(db.astype(jnp.float32),axis=-1))
  dlg=(dqg*(qkg*(1-qkg))).astype(dt)
  dlv=(dvg*(vg*(1-vg))).astype(dt)
  sink(6,jnp.sum(jnp.concatenate((dlg,dlv),axis=0).astype(jnp.float32),axis=1)[None,:])
  dp=jnp.concatenate((db.reshape(-1,t),dmix.reshape(-1,t),dlg,dvk.reshape(-1,t),dlv,drp),axis=0)
  sink(4,jnp.dot(x,dp.T,preferred_element_type=jnp.float32))
  dx=(dx+jnp.dot(w,dp,preferred_element_type=jnp.float32).astype(dt)).astype(dt)
  sink(3,jnp.sum((dx*normalized).astype(dt).astype(jnp.float32),axis=1)[None,:])
  draw=_norm_backward(raw,(dx.astype(jnp.float32)*scale.astype(jnp.float32)[:,None]).astype(dt),epsilon).reshape(h,v,t)
  return (dm+jnp.concatenate((draw,dmtail),axis=0)).astype(dt)


def forward_call(args,epsilon,re,rd,rmin,rmax,interpret,tile):
  m=args[0];b,k,v,t=m.shape;h=k//3
  def kernel(*refs):
    out,x,_=state(*(r[...] for r in refs[:8]),epsilon,re,rd,rmin,rmax)
    refs[8][...]=out;refs[9][...]=x
  inp=[_spec(m.shape[1:-1],tile)]+[pl.BlockSpec(x.shape,lambda b,i,n=x.ndim:(0,)*n) for x in args[1:7]]+[_spec((1,),tile)]
  return pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp,
      out_specs=[_spec((3*h,v),tile),_spec((h*v,),tile)],
      out_shape=(jax.ShapeDtypeStruct((b,3*h,v,t),m.dtype),jax.ShapeDtypeStruct((b,h*v,t),m.dtype)),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','parallel')),
      name='rmt_full_noo_attention_read')(*args)


@partial(jax.custom_vjp,nondiff_argnums=tuple(range(8,16)))
def fused(m,s,c,scale,w,bb,gb,pos,epsilon,re,rd,rmin,rmax,interpret,ft,rt):
  return forward_call((m,s,c,scale,w,bb,gb,pos),epsilon,re,rd,rmin,rmax,interpret,ft)


def fwd(m,s,c,scale,w,bb,gb,pos,epsilon,re,rd,rmin,rmax,interpret,ft,rt):
  args=(m,s,c,scale,w,bb,gb,pos)
  return forward_call(args,epsilon,re,rd,rmin,rmax,interpret,ft),args


def bwd(epsilon,re,rd,rmin,rmax,interpret,ft,rt,args,cts):
  m=args[0];b,k,v,t=m.shape;h=k//3;tile=min(rt,t)
  inp=[_spec(m.shape[1:-1],tile)]+[pl.BlockSpec(z.shape,lambda b,i,n=z.ndim:(0,)*n) for z in args[1:7]]+[_spec((1,),tile)]
  shapes=[(1,)+z.shape if z.ndim==1 else z.shape for z in args[1:7]]
  outs=[inp[0]]+[pl.BlockSpec((None,)+sh,lambda b,i,n=len(sh):(b,)+(0,)*n,pipeline_mode=pl.Buffered(1)) for sh in shapes]
  def kernel(*refs):
    @pl.when(pl.program_id(1)==0)
    def init():
      for ref in refs[11:]:ref[...]=jnp.zeros(ref.shape,ref.dtype)
    def sink(index,value):
      ref=refs[10+index];ref[...]=ref[...]+value
    refs[10][...]=reverse(*(ref[...] for ref in refs[:10]),epsilon,re,rd,rmin,rmax,sink)
  grads=pl.pallas_call(kernel,grid=(b,t//tile),in_specs=inp+[_spec((3*h,v),tile),_spec((h*v,),tile)],out_specs=outs,
      out_shape=(jax.ShapeDtypeStruct(m.shape,m.dtype),)+tuple(jax.ShapeDtypeStruct((b,)+sh,jnp.float32) for sh in shapes),
      interpret=interpret,compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel','arbitrary')),
      name='rmt_full_noo_attention_read_backward')(*args,*cts)
  return (grads[0],)+tuple(jnp.sum(g,axis=0).reshape(z.shape).astype(z.dtype) for g,z in zip(grads[1:],args[1:7]))+(None,)


fused.defvjp(fwd,bwd)


def attention_read(m,s,c,scale,w,bb,gb,positions,epsilon=1e-6,read_epsilon=1e-6,rope_dim=18,
                   rope_min=1.,rope_max=10000.,*,interpret=False,forward_tile=128,reverse_tile=128):
  def local(m,s,c,scale,w,bb,gb,pos):
    ft=min(forward_tile,m.shape[1]);rt=min(reverse_tile,m.shape[1])
    if m.shape[1]%ft or m.shape[1]%rt:raise ValueError('Attention chunks must divide tokens')
    out,x=fused(m.transpose(0,2,3,1),s,c,scale,w,bb,gb,pos[:,None,:],
                epsilon,read_epsilon,rope_dim,rope_min,rope_max,interpret,ft,rt)
    return out.transpose(0,3,1,2),x.transpose(0,2,1)
  return _map_batch(local,(m,s,c,scale,w,bb,gb,positions),(True,False,False,False,False,False,False,True),output_tuple=True)

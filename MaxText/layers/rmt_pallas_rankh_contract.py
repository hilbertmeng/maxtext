"""Rank-H contraction alternatives; inline in a complete write pullback.

Minor ABI: C[H,K,T], D[H,V,T], G[K,V,T]. The pure-contraction benchmark uses
this same ABI for all backends, so layout conversion is included fairly.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def mxu_major_loop(c, d, g):
  """Load/contract/store one token at a time to bound batched-dot SSA liveness.

  All staging is VMEM inside the enclosing fused stage; no extra HBM boundary.
  Compare against batched MXU: less spilling can lose to loop/weight-push latency.
  """
  t,h,k=c.shape;v=d.shape[-1]
  def scoped(cr,dr,gr,dcr,ddr):
    cr[...]=c;dr[...]=d;gr[...]=g
    def token(i,_):
      ci,di,gi=cr[i,:,:],dr[i,:,:],gr[i,:,:]
      dcr[i,:,:]=jnp.dot(di,gi.T,preferred_element_type=jnp.float32)
      ddr[i,:,:]=jnp.dot(ci,gi,preferred_element_type=jnp.float32)
    jax.lax.fori_loop(0,t,token,None)
    return dcr[...],ddr[...]
  return pl.run_scoped(scoped,pltpu.VMEM(c.shape,c.dtype),pltpu.VMEM(d.shape,d.dtype),
      pltpu.VMEM(g.shape,g.dtype),pltpu.VMEM((t,h,k),jnp.float32),pltpu.VMEM((t,h,v),jnp.float32))


def mxu(c, d, g, paired=False):
  h,k,t=c.shape;v=d.shape[1]
  cm,dm,gm=c.transpose(2,0,1),d.transpose(2,0,1),g.transpose(2,0,1)
  if paired:
    dc=jnp.einsum('thv,tkv->thk',dm,gm,preferred_element_type=jnp.float32)
    dd=jnp.einsum('thk,tkv->thv',cm,gm,preferred_element_type=jnp.float32)
  else:
    square=jnp.concatenate((
        jnp.concatenate((jnp.zeros((t,k,k),g.dtype),gm),axis=2),
        jnp.concatenate((gm.swapaxes(1,2),jnp.zeros((t,v,v),g.dtype)),axis=2)),axis=1)
    prod=jnp.einsum('thd,tdc->thc',jnp.concatenate((cm,dm),axis=2),square,
                    preferred_element_type=jnp.float32)
    dc,dd=prod[:,:,:k],prod[:,:,k:]
  return dc.transpose(1,2,0),dd.transpose(1,2,0)


def blocked_product(matrix_ref,coeff_ref,out_ref,heads,contract_dim,width,tokens,head_block,unroll=1):
  """J=4 heads x full output width: 24/40 accumulator vregs at T=128.

  Coefficients have explicit [1,T] trailing shape. This requests a sublane
  broadcast from a singleton Ref dimension; pl.ds(stride=0) is not legal in JAX.
  Verify its actual load/broadcast lowering in the Mosaic/LLO dumps.
  """
  def head_group(jb,_):
    def mac(i,acc):
      row=matrix_ref[i,:,:]
      return tuple(acc[j]+row*coeff_ref[i,jb*head_block+j,:,:]
                   for j in range(head_block))
    acc=jax.lax.fori_loop(0,contract_dim,mac,
        tuple(jnp.zeros((width,tokens),jnp.float32) for _ in range(head_block)),unroll=unroll)
    for j in range(head_block):out_ref[jb*head_block+j,:,:]=acc[j]
  jax.lax.fori_loop(0,heads//head_block,head_group,None)


def products(c,d,g,backend='symmetric',head_block=4,unroll=1):
  if backend=='loop':
    dc,dd=mxu_major_loop(c.astype(g.dtype).transpose(2,0,1),
                        d.astype(g.dtype).transpose(2,0,1),g.transpose(2,0,1))
    return dc.transpose(1,2,0),dd.transpose(1,2,0)
  if backend in ('symmetric','paired'):
    return mxu(c.astype(g.dtype),d.astype(g.dtype),g,backend=='paired')
  h,k,t=c.shape;v=d.shape[1];vp=(v+7)//8*8
  if t>128 or h%head_block:
    raise ValueError('Register-blocked contractions require <=128 compute tokens and divisible heads')
  if backend not in ('blocked','hybrid'):raise ValueError(backend)
  def scoped(dc_ref,dd_ref,c_ref,g_ref,*extra):
    c_ref[...]=c.astype(jnp.float32).transpose(1,0,2)[:,:,None,:]
    g_ref[...]=jnp.pad(g.astype(jnp.float32),((0,0),(0,vp-v),(0,0)))
    if backend=='blocked':
      d_ref,gt_ref=extra
      d_ref[...]=d.astype(jnp.float32).transpose(1,0,2)[:,:,None,:]
      gt_ref[...]=g_ref[...].transpose(1,0,2)
      blocked_product(gt_ref,d_ref,dc_ref,h,v,k,t,head_block,unroll)
    else:
      # Independent of the VPU contraction below: no dC-dependent epilogue here.
      dc=jnp.einsum('thv,tkv->thk',d.astype(g.dtype).transpose(2,0,1),g.transpose(2,0,1),
                    preferred_element_type=jnp.float32)
      dc_ref[...]=dc.transpose(1,2,0)
    blocked_product(g_ref,c_ref,dd_ref,h,k,vp,t,head_block,unroll)
    return dc_ref[...],dd_ref[...][:,:v,:]
  shapes=[(h,k,t),(h,vp,t),(k,h,1,t),(k,vp,t)]
  if backend=='blocked':shapes += [(v,h,1,t),(vp,k,t)]
  return pl.run_scoped(scoped,*[pltpu.VMEM(sh,jnp.float32) for sh in shapes])

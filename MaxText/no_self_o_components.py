"""Sampled native-read dynamic/static O decomposition; preserve total native forward.

Pallas sums the FP32 dynamic/static keys then casts combined key and M to BF16
before its dot. We decompose the FP32 keys against that same BF16 M, retaining
and checking the small key/output-quantization closure residual explicitly.
"""
import functools
import jax
import jax.numpy as jnp
import no_self_capture as base
SOURCE_NAMES=('MHA_o','M_o_dynamic','M_o_static','M_o_total_native')
_INSTALLED=False

def split(args,kw):
 m,st,qkey,_,_,_,vo,ol=args[:8];ids=base.positions(m.shape[1]);n=qkey.shape[2];dt=m.dtype
 assert vo.shape[2]==2*n and st.shape[0]>=4*n, (vo.shape,st.shape,qkey.shape)
 # Public bam_pallas.read carries compression rows after4N static rows; v8t ignores that tail.
 r=vo[:,ids,n:,:].astype(jnp.float32)
 # Sequential sum matches v8t _tm_slabs, including normalized generalized pre-bias.
 ss=r[...,0]*r[...,0]
 for i in range(1,r.shape[-1]):ss=ss+r[...,i]*r[...,i]
 inv=jax.lax.rsqrt(ss/r.shape[-1]+kw['read_epsilon']);g=kw['key_scale']*jax.nn.sigmoid(ol[:,ids].astype(jnp.float32))
 kd=(g*inv)[...,None]*r;ks=st[3*n:4*n].astype(jnp.float32)[None,None]*kw['static_gates'][3][:,ids,:,None].astype(jnp.float32)
 mr=m[:,ids].astype(jnp.float32)
 if kw.get('shift') is not None:
  prev=m[:,jnp.maximum(ids-1,0)].astype(jnp.float32);mr=mr+kw['shift'][:,ids,:,None].astype(jnp.float32)*(prev-mr)
 mr=mr.astype(dt).astype(jnp.float32)
 dot=lambda key:jnp.einsum('bqhc,bqck->bqhk',key,mr,precision=jax.lax.Precision.HIGHEST)
 return dot(kd),dot(ks),dot((kd+ks).astype(dt).astype(jnp.float32)).astype(dt).astype(jnp.float32)

def stats(x,kernel):
 # x[B,Q,H,S,K], no full token/head/outputdim activation is materialized.
 pre=jnp.einsum('bqhsk,bqhtk->bhst',x,x,precision=jax.lax.Precision.HIGHEST)
 gram=jnp.einsum('hkd,hjd->hkj',kernel,kernel,precision=jax.lax.Precision.HIGHEST)
 gx=jnp.einsum('bqhsk,hkj->bqhsj',x,gram,precision=jax.lax.Precision.HIGHEST)
 heads=jnp.einsum('bqhsk,bqhtk->bhst',gx,x,precision=jax.lax.Precision.HIGHEST)
 projected=jnp.einsum('bqhsk,hkd->bqsd',x,kernel,precision=jax.lax.Precision.HIGHEST)
 total=jnp.einsum('bqsd,bqtd->bst',projected,projected,precision=jax.lax.Precision.HIGHEST)
 return pre,heads,total

def observe(self,c,out,value):
 del value
 ids=base.positions(out.shape[1]);a=c['mha'][:,ids].astype(jnp.float32);o=c['mo'][:,ids].astype(jnp.float32);d,s,recon=c['o_components'];x=jnp.stack([a,d,s,o],axis=3)
 kernel=jnp.asarray(base.unbox(self.variables['params']['out']['kernel']),self.dtype).astype(jnp.float32);pre,heads,total=stats(x,kernel)
 err=lambda z:jnp.sum((o-z)**2,axis=(1,3))
 closure=jnp.stack([err(d+s),err(recon),jnp.sum(o*o,axis=(1,3))],-1)
 for key,v in dict(layer_index=jnp.asarray(c['layer'],jnp.int32),component_pre=pre,component_post_heads=heads,component_post_total=total,component_closure=closure,empty_history_mha_max=jnp.max(jnp.abs(c['mha'][:,0].astype(jnp.float32)),axis=(1,2))).items():self.sow('intermediates','readout_'+key,v)

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True;base.install();base.observe=observe
 from layers import bam_pallas
 previous=bam_pallas.read
 @functools.wraps(previous)
 def read(*args,**kw):
  result=previous(*args,**kw)
  if base.ACTIVE and base.OBSERVE and base._STACK:base._STACK[-1]['o_components']=split(args,kw)
  return result
 bam_pallas.read=read

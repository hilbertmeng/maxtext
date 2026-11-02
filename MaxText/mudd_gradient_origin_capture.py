"""Actual-trajectory source decomposition; no changes to native model math."""
import re
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

def normalized_integrated_parts(base,total,pieces,gamma,basis,nodes=128):
 """Conditional path IG of final LN; upstream block outputs remain frozen."""
 center=lambda x:x.astype(jnp.float32)-x.astype(jnp.float32).mean(-1,keepdims=True)
 b=center(base);d=center(total)-b;p=center(pieces)
 folded=gamma[:,None]*basis
 project=lambda x:jnp.einsum('...d,dk->...k',x,folded,precision=jax.lax.Precision.HIGHEST)
 bp,dp,pp=project(b),project(d),project(p)
 aa=(b*b).mean(-1);ab=(b*d).mean(-1);dd=(d*d).mean(-1)
 q,w=np.polynomial.legendre.leggauss(nodes)
 q=jnp.asarray((q+1)/2,jnp.float32)[:,None,None];w=jnp.asarray(w/2,jnp.float32)[:,None,None]
 inv=jax.lax.rsqrt(jnp.maximum(aa[None]+2*q*ab[None]+q*q*dd[None]+1e-5,1e-5))
 i1=(w*inv).sum(0)[...,None];i3=(w*inv**3).sum(0)[...,None]
 ia3=(w*q*inv**3).sum(0)[...,None];ia23=(w*q*q*inv**3).sum(0)[...,None]
 pb=(p*b[None]).mean(-1,keepdims=True);pd=(p*d[None]).mean(-1,keepdims=True)
 ig=pp*i1[None]-bp[None]*(pb*i3[None]+pd*ia3[None])-dp[None]*(pb*ia3[None]+pd*ia23[None])
 endpoint=lambda x:project(center(x))*jax.lax.rsqrt((center(x)**2).mean(-1,keepdims=True)+1e-5)
 expected=endpoint(total)-endpoint(base)
 return ig,expected

class OriginCapture:
 def __init__(self,gamma,bias,basis,layers):
  self.gamma=gamma;self.bias=bias;self.basis=basis;self.layers=layers
  self.folded=gamma[:,None]*basis;self.raw=self.h=self.final_weights=None;self.final_sources=None
  self.final_raw_sources=self.final_base=None
 def project(self,x):
  x=x.astype(jnp.float32);x=x-x.mean(-1,keepdims=True)
  return jnp.einsum('btd,dk->btk',x,self.folded,precision=jax.lax.Precision.HIGHEST)
 def index(self,module):return int(re.search(r'layers_(\d+)','/'.join(module.scope.path)).group(1))
 def interceptor(self,next_fun,args,kwargs,ctx):
  m=ctx.module;name=m.__class__.__name__
  if ctx.method_name!='__call__':return next_fun(*args,**kwargs)
  if name=='Mlp' and 'compose_final' in m.scope.path:
   out=next_fun(*args,**kwargs);self.final_weights=out;return out
  if name=='Compose':
   base=kwargs.get('layer_output',args[0] if args else None)
   history=kwargs.get('hids',args[1] if len(args)>1 else None)
   out=next_fun(*args,**kwargs);mixed=out[0][-1]
   delta=mixed.astype(jnp.float32)-base.astype(jnp.float32)
   m.sow('intermediates','origin_mudd',self.project(delta))
   m.sow('intermediates','origin_raw_mudd',delta)
   if m.name=='compose_final':
    assert self.final_weights is not None and len(history)==self.layers+1
    w=self.final_weights[...,0,:].astype(jnp.float32)
    values=[];raw_values=[]
    for i,v in enumerate(history):
     coef=w[...,i,None]-(1. if i==len(history)-1 else 0.)
     raw_values.append(coef*v.astype(jnp.float32));values.append(self.project(raw_values[-1]))
    values=jnp.stack(values)
    remainder=self.project(delta)-values.sum(0)
    self.final_sources=jnp.concatenate([values,remainder[None]],axis=0)
    raw_values=jnp.stack(raw_values)
    self.final_raw_sources=jnp.concatenate([raw_values,(delta-raw_values.sum(0))[None]],axis=0)
    self.final_base=base
   return out
  if name=='SubDecoderLayer':
   inputs=args[0];base=inputs[-1] if isinstance(inputs,(tuple,list)) else inputs
   index=self.index(m)
   if index==0:
    m.sow('intermediates','origin_embedding',self.project(base))
    m.sow('intermediates','origin_raw_embedding',base.astype(jnp.float32))
   out=next_fun(*args,**kwargs)
   delta=out[0].astype(jnp.float32)-base.astype(jnp.float32)
   m.sow('intermediates','origin_block',self.project(delta))
   m.sow('intermediates','origin_raw_block',delta);return out
  if name=='LayerNorm' and m.name=='decoder_norm' and 'lm_head' in m.scope.path:
   self.raw=args[0];out=next_fun(*args,**kwargs);self.h=out;return out
  return next_fun(*args,**kwargs)
 def result(self,collection):
  block={};mudd={};embedding=None;raw_block={};raw_mudd={};raw_embedding=None
  for path,values in flatten_dict(collection).items():
   key=path[-1]
   if not key.startswith('origin_'):continue
   assert len(values)==1,path
   if key=='origin_embedding':assert embedding is None;embedding=values[0]
   elif key=='origin_raw_embedding':raw_embedding=values[0]
   elif key=='origin_raw_block':raw_block[int(re.search(r'layers_(\d+)','/'.join(path)).group(1))]=values[0]
   elif key=='origin_raw_mudd':
    index=self.layers if 'compose_final' in path else int(re.search(r'layers_(\d+)','/'.join(path)).group(1))
    raw_mudd[index]=values[0]
   elif key=='origin_block':block[int(re.search(r'layers_(\d+)','/'.join(path)).group(1))]=values[0]
   elif key=='origin_mudd':
    index=self.layers if 'compose_final' in path else int(re.search(r'layers_(\d+)','/'.join(path)).group(1))
    mudd[index]=values[0]
  assert set(block)==set(range(self.layers)) and set(mudd)==set(range(1,self.layers+1))
  assert self.raw is not None and self.h is not None and embedding is not None
  names=['embedding']+[f'ordinary_block_{i+1:02d}' for i in range(self.layers)]+[f'mudd_before_block_{i+1:02d}' for i in range(1,self.layers)]+['mudd_final_readout']
  raw_parts=jnp.stack([embedding]+[block[i] for i in range(self.layers)]+[mudd[i] for i in range(1,self.layers+1)])
  raw=self.raw.astype(jnp.float32);centered=raw-raw.mean(-1,keepdims=True)
  inv=jax.lax.rsqrt((centered*centered).mean(-1,keepdims=True)+jnp.float32(1e-5))
  head=jnp.einsum('btd,dk->btk',self.h.astype(jnp.float32),self.basis,precision=jax.lax.Precision.HIGHEST)
  bias=jnp.broadcast_to(self.bias@self.basis,head.shape)
  parts=raw_parts*inv[None]
  numerical=head-parts.sum(0)-bias
  names+=['final_LN_bias','native_LN_numerical_remainder']
  parts=jnp.concatenate([parts,bias[None],numerical[None]],axis=0)
  sources=self.final_sources*inv[None]
  frozen_ordinary=raw_embedding+jnp.stack([raw_block[i] for i in range(self.layers)]).sum(0)
  increments=jnp.stack([raw_mudd[i] for i in range(1,self.layers+1)])
  global_ig,global_expected=normalized_integrated_parts(frozen_ordinary,raw,increments,self.gamma,self.basis)
  final_ig,final_expected=normalized_integrated_parts(self.final_base,raw,self.final_raw_sources,self.gamma,self.basis)
  def normalized_projection(x):
   c=x.astype(jnp.float32)-x.astype(jnp.float32).mean(-1,keepdims=True)
   n=c*jax.lax.rsqrt((c*c).mean(-1,keepdims=True)+1e-5)
   h=(n*self.gamma+self.bias).astype(self.h.dtype).astype(jnp.float32)
   return jnp.einsum('btd,dk->btk',h,self.basis,precision=jax.lax.Precision.HIGHEST)
  global_base=normalized_projection(frozen_ordinary);final_base=normalized_projection(self.final_base)
  global_parts=jnp.concatenate([global_base[None],global_ig,(head-global_base-global_ig.sum(0))[None]],axis=0)
  final_parts=jnp.concatenate([final_base[None],final_ig,(head-final_base-final_ig.sum(0))[None]],axis=0)
  return dict(h=self.h,projected_components=parts,projected_final_sources=sources,raw_projection=self.project(raw),raw_component_sum=raw_parts.sum(0),head_projection=head,invstd=inv,
   normalized_global_components=global_parts,normalized_final_components=final_parts,
   global_IG_expected=global_expected,global_IG_sum=global_ig.sum(0),final_IG_expected=final_expected,final_IG_sum=final_ig.sum(0),
   raw_telescoping_error=jnp.sqrt(jnp.mean((frozen_ordinary+increments.sum(0)-raw)**2)/jnp.mean(raw**2))),names

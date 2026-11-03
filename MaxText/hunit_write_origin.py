"""Direct residual-write accounting; native Pallas/Splash functions stay unchanged."""
from contextlib import contextmanager
from unittest import mock
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
from layers import attentions,bam_pallas
from head_spectral_diagnostic import BF16_round

NAMES=['mha','mo_dynamic','mo_static','attn_bias','attn_rounding','ffn','residual_add_rounding']
class WriteObserver:
 def __init__(self,u,gamma):
  self.u=u;self.rank=u.shape[1];b=gamma[:,None]*u
  self.transform=jnp.concatenate((u-u.mean(0),b-b.mean(0)),axis=1)
  self.active=None;self.raw=self.h=None;self.original_read=bam_pallas.read;self.original_splash=attentions._bam_splash_attention
 def project(self,x):
  # Preserve the declared BF16 boundary even if XLA can fuse a BF16->FP32
  # consumer into the upstream elementwise operation with excess precision.
  xf=BF16_round(x) if x.dtype==jnp.bfloat16 else x.astype(jnp.float32)
  return jnp.einsum('btd,dr->btr',xf,self.transform,precision=jax.lax.Precision.HIGHEST)
 def read(self,*args,**kw):
  out=self.original_read(*args,**kw)
  if self.active is not None:
   assert kw['body']=='v8t' and kw.get('o_dynamic_scale') is not None
   k=dict(kw,o_dynamic_scale=jnp.zeros_like(kw['o_dynamic_scale']))
   self.active['mo']=out[3];self.active['mo_static']=self.original_read(*args,**k)[3]
  return out
 def splash(self,*args,**kw):
  out=self.original_splash(*args,**kw)
  if self.active is not None:self.active['mha_raw']=out
  return out
 def interceptor(self,next_fun,args,kw,ctx):
  mod=ctx.module;cls=mod.__class__.__name__;method=ctx.method_name
  if cls=='SubDecoderLayer' and method=='__call__':
   previous=self.active;s={};self.active=s;before=args[0];out=next_fun(*args,**kw);after=out[0]
   assert all(k in s for k in ['mha','mo_dynamic','mo_static','attn_bias','attn_rounding','ffn'])
   bp=self.project(before);ap=self.project(after);s['residual_add_rounding']=ap-bp-s['attn_projected']-s['ffn']
   mod.sow('intermediates','write_origin_parts',jnp.stack([s[k] for k in NAMES]))
   mod.sow('intermediates','write_origin_before',bp);mod.sow('intermediates','write_origin_after',ap)
   mod.sow('intermediates','write_origin_read_power',s['read_power'])
   self.active=previous;return out
  if cls=='BamAttention' and method=='out_projection' and self.active is not None:
   out=next_fun(*args,**kw);s=self.active;params=nn.unbox(mod.get_variable('params','out'));wf=BF16_round(params['kernel']).reshape(-1,out.shape[-1]);wb=jnp.matmul(wf,self.transform,precision=jax.lax.Precision.HIGHEST)
   def proj(x):return jnp.matmul(x.astype(jnp.float32).reshape(x.shape[:2]+(-1,)),wb,precision=jax.lax.Precision.HIGHEST)
   s['mha']=proj(s['mha_raw']);s['mo_static']=proj(s['mo_static']);s['mo_dynamic']=proj(s['mo'])-s['mo_static']
   bias=BF16_round(params['bias']) if 'bias' in params else jnp.zeros(out.shape[-1]);s['attn_bias']=jnp.broadcast_to(bias@self.transform,s['mha'].shape)
   s['attn_projected']=self.project(out);s['attn_rounding']=s['attn_projected']-s['mha']-s['mo_static']-s['mo_dynamic']-s['attn_bias']
   s['read_power']=jnp.stack([jnp.mean(s[k].astype(jnp.float32)**2,(0,1,3)) for k in ['mha_raw','mo']])
   return out
  if cls=='MlpBlock' and method=='__call__' and mod.name=='mlp' and self.active is not None:
   out=next_fun(*args,**kw);assert not isinstance(out,tuple);self.active['ffn']=self.project(out);return out
  if cls=='LayerNorm' and method=='__call__' and mod.name=='decoder_norm' and 'lm_head' in mod.scope.path:
   self.raw=args[0];out=next_fun(*args,**kw);self.h=out;return out
  return next_fun(*args,**kw)
 @contextmanager
 def observe(self):
  with mock.patch.object(bam_pallas,'read',self.read),mock.patch.object(attentions,'_bam_splash_attention',self.splash),nn.intercept_methods(self.interceptor):yield self

def layers(collection,name):
 flat=flatten_dict(collection['intermediates']);values=[]
 for p,v in flat.items():
  if p[-1]!=name:continue
  assert isinstance(v,tuple) and len(v)==1;ptext='/'.join(p);a=v[0]
  if 'final_local_layer' in ptext:values.append((1,a[None]))
  else:values.append((0,a))
 assert len(values)==2,[(k,v.shape) for k,v in values]
 return jnp.concatenate([v for _,v in sorted(values,key=lambda x:x[0])],axis=0)

def aggregate(observer,collection,gamma,bias,epsilon=1e-5):
 u=observer.u;rank=u.shape[1];raw=BF16_round(observer.raw);h=BF16_round(observer.h);d=raw.shape[-1];xc=raw-raw.mean(-1,keepdims=True);inv=jax.lax.rsqrt(jnp.mean(xc*xc,-1)+epsilon)
 parts=layers(collection,'write_origin_parts');before=layers(collection,'write_origin_before');after=layers(collection,'write_origin_after');embed=before[0];target_raw=xc@u;target_h=h@u
 raw_parts=parts[...,:rank];head_parts=parts[...,rank:]*inv[None,None,...,None]
 inner=lambda a,b:jnp.mean(jnp.sum(a*b,axis=-1),axis=(-2,-1))/d
 power=lambda a:jnp.mean(jnp.sum(a*a,axis=-1),axis=(-2,-1))/d
 total_head=power(target_h);total_raw=power(target_raw)
 embed_h=embed[...,rank:]*inv[...,None];beta=jnp.broadcast_to(bias@u,target_h.shape)
 head_round=target_h-(xc*gamma)@u*inv[...,None]-beta
 allocation=inner(head_parts,target_h);raw_allocation=inner(raw_parts,target_raw)
 extra=jnp.stack([inner(embed_h,target_h),inner(beta,target_h),inner(head_round,target_h)])
 continuity=after[:-1]-before[1:];closure=embed+parts.sum((0,1))-after[-1]
 return dict(head_allocation=allocation,raw_allocation=raw_allocation,head_self_power=power(head_parts),raw_self_power=power(raw_parts),extra_head_allocation=extra,extra_raw_embedding_allocation=inner(embed[...,:rank],target_raw),head_total_power=total_head,raw_total_power=total_raw,head_allocation_closure=(allocation.sum()+extra.sum())/total_head-1,raw_allocation_closure=(raw_allocation.sum()+inner(embed[...,:rank],target_raw))/total_raw-1,projection_ledger_relative_error=jnp.sqrt(jnp.sum(closure**2)/jnp.sum(after[-1]**2)),layer_continuity_max=jnp.max(jnp.abs(continuity)),read_head_power=layers(collection,'write_origin_read_power'),raw=raw,h=h)

class LastReadAblation:
 """Only final attention's input to native Wo changes; MHA/V/keys stay original."""
 def __init__(self,mode):self.mode=mode;self.active=None;self.raw=self.h=None;self.original_read=bam_pallas.read;self.original_splash=attentions._bam_splash_attention
 def read(self,*a,**kw):
  out=self.original_read(*a,**kw)
  if self.active is not None:self.active['mo']=out[3]
  return out
 def splash(self,*a,**kw):
  out=self.original_splash(*a,**kw)
  if self.active is not None:self.active['mha']=out
  return out
 def interceptor(self,next_fun,args,kw,ctx):
  mod=ctx.module
  if mod.__class__.__name__=='BamAttention' and ctx.method_name=='__call__' and any('final_local_layer' in p for p in mod.scope.path):
   self.active={};out=next_fun(*args,**kw);self.active=None;return out
  if mod.__class__.__name__=='BamAttention' and ctx.method_name=='out_projection' and self.active is not None:
   out=args[1];s=self.active;changed=jnp.where(self.mode==1,s['mha'],jnp.where(self.mode==2,s['mo'],out));return next_fun(args[0],changed,**kw)
  if mod.__class__.__name__=='LayerNorm' and ctx.method_name=='__call__' and mod.name=='decoder_norm' and 'lm_head' in mod.scope.path:
   self.raw=args[0];out=next_fun(*args,**kw);self.h=out;return out
  return next_fun(*args,**kw)
 @contextmanager
 def observe(self):
  with mock.patch.object(bam_pallas,'read',self.read),mock.patch.object(attentions,'_bam_splash_attention',self.splash),nn.intercept_methods(self.interceptor):yield self

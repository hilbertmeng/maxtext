"""Native MHA/M-o observations and independently controlled attention exit/write routes.

Controls[L,4] = [vector_MHA,vector_Mo,attention_write_MHA,attention_write_Mo].
Values are0/1; neutral paths retain their original sum without recomputation.
Downstream FFNs and M-only FFN writes remain live in every intervention.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
ACTIVE=False
OBSERVE=False
RETAIN=False
CONTROLS=None
_STACK=[]
_INSTALLED=False
GATE_NAMES=('v_dynamic_sigmoid','o_dynamic_sigmoid','v_static_sigmoid','o_static_sigmoid','v_static_actual_multiplier','o_static_actual_multiplier')
def positions(t):return np.unique(np.linspace(min(64,t-1),t-1,min(64,t),dtype=np.int32))
def unbox(v):return v.unbox(apply_constraint=False) if hasattr(v,'unbox') else v
def combine(a,o,coeff,original):
 return jax.lax.cond(jnp.all(coeff==1),lambda _:original,lambda _:coeff[0].astype(a.dtype)*a+coeff[1].astype(o.dtype)*o,None)
def pair_moments(a,o,y,axes,count):
 a,o,y=[x.astype(jnp.float32) for x in (a,o,y)]
 out=[jnp.sum(x,axis=axes) for x in [a*a,o*o,a*o,y*y]]
 return jnp.stack(out+[jnp.full_like(out[0],count)],-1)
def head_projected_moments(a,o,kernel):
 B,Q,H,K=a.shape;D=kernel.shape[-1]
 gram=jnp.einsum('hkd,hjd->hkj',kernel,kernel,precision=jax.lax.Precision.HIGHEST)
 ag=jnp.einsum('bqhk,hkj->bqhj',a,gram,precision=jax.lax.Precision.HIGHEST);og=jnp.einsum('bqhk,hkj->bqhj',o,gram,precision=jax.lax.Precision.HIGHEST)
 return jnp.stack([jnp.sum(ag*a,axis=(1,3)),jnp.sum(og*o,axis=(1,3)),jnp.sum(ag*o,axis=(1,3)),jnp.full((B,H),Q*D,jnp.float32)],-1)
def observe(self,c,out,value):
 ids=positions(out.shape[1]);a=c['mha'][:,ids].astype(jnp.float32);o=c['mo'][:,ids].astype(jnp.float32);y=out[:,ids].astype(jnp.float32)
 B,Q,H,K=a.shape;kernel=jnp.asarray(unbox(self.variables['params']['out']['kernel']),self.dtype).astype(jnp.float32);D=kernel.shape[-1]
 if kernel.shape!=(H,K,D):raise ValueError(('W_o geometry',kernel.shape,a.shape))
 pa=jnp.einsum('bqhk,hkd->bqd',a,kernel,precision=jax.lax.Precision.HIGHEST);po=jnp.einsum('bqhk,hkd->bqd',o,kernel,precision=jax.lax.Precision.HIGHEST)
 projected_head=head_projected_moments(a,o,kernel)
 bias=jnp.asarray(unbox(self.variables['params']['out']['bias']),self.dtype).astype(jnp.float32) if self.output_bias else jnp.zeros((D,),jnp.float32)
 actual=value[:,ids].astype(jnp.float32);closure=jnp.stack([jnp.sum((actual-(pa+po+bias))**2,axis=(1,2)),jnp.sum(actual*actual,axis=(1,2)),jnp.sum((y-(a+o))**2,axis=(1,2,3)),jnp.sum(y*y,axis=(1,2,3))],-1)
 gates=jnp.stack([c[k][:,ids].astype(jnp.float32) for k in GATE_NAMES],-1)
 values=dict(pre_heads=pair_moments(a,o,y,(1,3),Q*K),post_heads=projected_head,post_total=pair_moments(pa,po,pa+po,(1,2),Q*D),gates=gates,closure=closure,layer_index=jnp.asarray(c['layer'],jnp.int32),empty_history_mha_max=jnp.max(jnp.abs(c['mha'][:,0].astype(jnp.float32)),axis=(1,2)))
 for key,val in values.items():self.sow('intermediates','readout_'+key,val)
def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 from layers import attentions,fusion,bam_pallas
 native_layer=fusion.SubDecoderLayer.__call__
 @functools.wraps(native_layer)
 def layer(self,*args,**kw):
  if not ACTIVE or self.is_initializing():return native_layer(self,*args,**kw)
  index=kw.get('layer_index');index=self.layer_inx if index is None else index;c=dict(layer=index);_STACK.append(c)
  try:
   result=native_layer(self,*args,**kw)
   if not c.get('projection_seen'):raise ValueError(('No native readout',self.scope.path))
   if RETAIN:
    self.sow('intermediates','audit_layer_index',jnp.asarray(index));self.sow('intermediates','audit_x',result[0]);self.sow('intermediates','audit_M',result[1])
   return result
  finally:_STACK.pop()
 fusion.SubDecoderLayer.__call__=layer
 native_attention=attentions.BamAttention.__call__
 @functools.wraps(native_attention)
 def attention(self,*args,**kw):
  result=native_attention(self,*args,**kw)
  if ACTIVE and _STACK and not self.is_initializing():
   c=_STACK[-1];coeff=CONTROLS[c['layer']]
   if len(result)!=3:raise ValueError('Require native deferred Pallas write factors')
   if result[2] is not None:
    content,gate,address=result[2];changed=combine(c['mha'],c['mo'],coeff[2:],content)
    if RETAIN:
     self.sow('intermediates','audit_write_content',changed);self.sow('intermediates','audit_write_content_native',content)
    result=(result[0],result[1],(changed,gate,address))
  return result
 attentions.BamAttention.__call__=attention
 native_fused=attentions.BamAttention._fused_input_projections
 @functools.wraps(native_fused)
 def fused(self,*args,**kw):
  out=native_fused(self,*args,**kw)
  if ACTIVE and OBSERVE and _STACK and not self.is_initializing():
   for arm in ['v','o']:
    name='general_'+arm+'_static_gate'
    if name in out:
     # Match the native dtype and sigmoid BEFORE its1.1/init scale.
     z=out[name]+jnp.asarray(getattr(self,name+'_bias'),self.dtype);_STACK[-1][arm+'_static_sigmoid']=jax.nn.sigmoid(z)
  return out
 attentions.BamAttention._fused_input_projections=fused
 native_read=bam_pallas.read
 @functools.wraps(native_read)
 def read(*args,**kw):
  out=native_read(*args,**kw)
  if ACTIVE and _STACK:
   c=_STACK[-1];c['mo']=out[3]
   if OBSERVE:
    c['o_dynamic_sigmoid']=jax.nn.sigmoid(args[7].astype(jnp.float32));c['v_dynamic_sigmoid']=jax.nn.sigmoid(args[8].astype(jnp.float32))
    sg=kw.get('static_gates');assert sg is not None;c['v_static_actual_multiplier']=sg[2];c['o_static_actual_multiplier']=sg[3]
  return out
 bam_pallas.read=read
 native_splash=attentions._bam_splash_attention
 @functools.wraps(native_splash)
 def splash(*args,**kw):
  out=native_splash(*args,**kw)
  if ACTIVE and _STACK:_STACK[-1]['mha']=out
  return out
 attentions._bam_splash_attention=splash
 native_block=attentions.BamAttention._attention_block
 @functools.wraps(native_block)
 def block(self,*args,**kw):
  out=native_block(self,*args,**kw)
  if ACTIVE and _STACK:
   if kw.get('q0',0)!=0 or kw.get('s0',0)!=0 or args[0].shape[1]!=args[1].shape[1]:raise ValueError('CPU audit requires full attention block')
   _STACK[-1]['mha']=out[0]
  return out
 attentions.BamAttention._attention_block=block
 native_projection=attentions.BamAttention.out_projection
 @functools.wraps(native_projection)
 def projection(self,dim,out):
  if not ACTIVE or self.is_initializing():return native_projection(self,dim,out)
  c=_STACK[-1];assert 'mha' in c and 'mo' in c and self.config.bam_pallas_core
  coeff=CONTROLS[c['layer']];changed=combine(c['mha'],c['mo'],coeff[:2],out);value=native_projection(self,dim,changed);c['projection_seen']=True
  if OBSERVE:observe(self,c,out,value)
  if RETAIN:
   self.sow('intermediates','audit_projected_output',value);self.sow('intermediates','audit_output_bias',jnp.asarray(unbox(self.variables['params']['out']['bias']),self.dtype) if self.output_bias else jnp.zeros((dim,),self.dtype));self.sow('intermediates','audit_vector_content',changed);self.sow('intermediates','audit_vector_content_native',out);self.sow('intermediates','audit_mha',c['mha']);self.sow('intermediates','audit_mo',c['mo'])
  return value
 attentions.BamAttention.out_projection=projection

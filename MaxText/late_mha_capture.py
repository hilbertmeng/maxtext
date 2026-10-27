"""Whole-layer native MHA exit interventions for BAM and serial Mudd.

Reuse the verified BAM controls [vectorA,vectorO,attentionWriteA,attentionWriteO].
Plain Mudd uses vectorA only; preserve W_o bias, FFN and all native routing.
"""
import functools
import jax
import jax.numpy as jnp
import no_self_capture as base
_INSTALLED=False

def observe(self,c,out,value):
 del value
 ids=base.positions(out.shape[1]);a=c['mha'][:,ids].astype(jnp.float32);o=c.get('mo',jnp.zeros_like(out))[:,ids].astype(jnp.float32);B,Q,H,K=a.shape
 kernel=jnp.asarray(base.unbox(self.variables['params']['out']['kernel']),self.dtype).astype(jnp.float32);D=kernel.shape[-1]
 assert kernel.shape==(H,K,D)
 pa=jnp.einsum('bqhk,hkd->bqd',a,kernel,precision=jax.lax.Precision.HIGHEST);po=jnp.einsum('bqhk,hkd->bqd',o,kernel,precision=jax.lax.Precision.HIGHEST)
 vals=dict(layer_index=jnp.asarray(c['layer'],jnp.int32),geometry=jnp.broadcast_to(jnp.asarray([out.shape[1],H,K,D],jnp.int32),(B,4)),controls=jnp.broadcast_to(base.CONTROLS[c['layer']],(B,4)),pre=base.pair_moments(a,o,a+o,(1,2,3),Q*H*K),post=base.pair_moments(pa,po,pa+po,(1,2),Q*D))
 for key,v in vals.items():self.sow('intermediates','readout_'+key,v)

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True;base.install();base.observe=observe
 from layers import attentions
 native=attentions.Attention.out_projection
 @functools.wraps(native)
 def projection(self,dim,out):
  if not base.ACTIVE or self.is_initializing():return native(self,dim,out)
  assert not self.config.bam_enabled
  c=base._STACK[-1];coef=base.CONTROLS[c['layer'],0];c['mha']=out;c['projection_seen']=True
  changed=jax.lax.cond(coef==1,lambda _:out,lambda _:jnp.zeros_like(out),None);value=native(self,dim,changed)
  if base.OBSERVE:observe(self,c,out,value)
  if base.RETAIN:
   self.sow('intermediates','audit_vector_content',changed);self.sow('intermediates','audit_mha',out);self.sow('intermediates','audit_projected_output',value);self.sow('intermediates','audit_output_bias',jnp.asarray(base.unbox(self.variables['params']['out']['bias']),self.dtype))
  return value
 attentions.Attention.out_projection=projection

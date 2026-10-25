"""Compare pre-FFN branch perturbations with direct block-exit vector links.

Controls are explicit function inputs, never learned parameters or updates.
coefficients[L,3] scale native MHA/dynamicO/staticO readout directions.
Both modes add the same W_o-projected perturbation at the block vector exit.
Live mode also presents it to the post-attention FFN normalization input.
Frozen mode retains native FFN/M writes. Identical residual addition order
prevents low-precision arithmetic order from masquerading as FFN mediation.
"""
import functools
import jax
import jax.numpy as jnp
CONTROLS=None
OBSERVE=False
RETAIN_LOCAL=False
_INSTALLED=False
_BLOCK=[]

def _unbox(value):
 return value.unbox(apply_constraint=False) if hasattr(value,'unbox') else value

def tangent_moments(z,d):
 z,d=z.astype(jnp.float32),d.astype(jnp.float32)
 dc=d-d.mean(-1,keepdims=True);zc=z-z.mean(-1,keepdims=True)
 rms=d-z*jnp.sum(z*d,-1,keepdims=True)/jnp.maximum(jnp.sum(z*z,-1,keepdims=True),1e-30)
 ln=dc-zc*jnp.sum(zc*dc,-1,keepdims=True)/jnp.maximum(jnp.sum(zc*zc,-1,keepdims=True),1e-30)
 vals=[jnp.sum(a*a,axis=(1,2)) for a in [d,d-dc,rms,ln]]
 return jnp.stack(vals+[jnp.full_like(vals[0],d.shape[1]*d.shape[2])],axis=-1)

def normalization_response(z,d,epsilon,center,gamma):
 """Analytic FP32 JVP including finite epsilon and the learned affine scale."""
 z,d=z.astype(jnp.float32),d.astype(jnp.float32);gamma=gamma.astype(jnp.float32)
 if center:z=z-z.mean(-1,keepdims=True);d=d-d.mean(-1,keepdims=True)
 variance=jnp.mean(z*z,-1,keepdims=True);den=variance+epsilon
 return gamma*(d-z*jnp.mean(z*d,-1,keepdims=True)/den)*jax.lax.rsqrt(den)

def layer_writes_m(cfg,logical_index):
 """Match the historical static scan-offset write predicate, including tails."""
 keys=cfg.get_keys();every=int(keys.get('bam_mlp_write_every') or 0)
 if every<=0:return False
 tails=keys.get('bam_final_local_layer_count')
 if tails is None:tails=int(bool(keys.get('bam_extra_final_local_layer')))
 native_index=logical_index
 if cfg.scan_layers and cfg.bam_pair_scan and logical_index<cfg.num_decoder_layers-tails:
  native_index=logical_index%cfg.bam_local_fetch_block_size
 return not(bool(keys.get('bam_skip_final_mlp_write')) and native_index==cfg.num_decoder_layers-1) and (native_index+1)%every==int(keys.get('bam_mlp_write_offset') or 0)%every

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 import prewo_observer as observer
 observer.install()
 from layers import attentions,fusion,normalizations
 native_projection=attentions.BamAttention.out_projection
 @functools.wraps(native_projection)
 def projection(self,output_dim,out):
  value=native_projection(self,output_dim,out)
  if CONTROLS is None or self.is_initializing():return value
  if not _BLOCK or not observer._STACK:raise ValueError('Missing native branch/block observation context')
  b=_BLOCK[-1];c=observer._STACK[-1];coeff=CONTROLS['coefficients'][b['layer']]
  if c['module'] is not self:raise ValueError('Attention observation ownership mismatch')
  direction=sum(coeff[i].astype(self.dtype)*c[name].astype(self.dtype) for i,name in enumerate(['mha','dynamic_o','static_o']))
  kernel=_unbox(self.variables['params']['out']['kernel']).astype(self.dtype)
  # A branch has no independent output bias: project with W_o only.
  delta=jnp.einsum('bthk,hkd->btd',direction,kernel,precision=self.config.matmul_precision).astype(value.dtype)
  b.update(delta=delta,active=jnp.any(coeff!=0))
  return value
 attentions.BamAttention.out_projection=projection
 def wrap_norm(native,center):
  @functools.wraps(native)
  def norm(self,x,*args,**kwargs):
   context=None
   if CONTROLS is not None and _BLOCK and self.name=='post_self_attention_layer_norm' and not self.is_initializing():
    c=_BLOCK[-1]
    if 'delta' not in c:raise ValueError('Missing projected branch perturbation before FFN')
    c['z']=x;context=c
    x=jax.lax.cond(jnp.asarray(CONTROLS['live'],bool)&c['active'],lambda a:(a+c['delta']).astype(a.dtype),lambda a:a,x)
   value=native(self,x,*args,**kwargs)
   if OBSERVE and context is not None:
    z,d=context['z'],context['delta'];gamma=_unbox(self.variables['params']['scale']).astype(self.dtype)
    if not center and not getattr(self,'direct_scale',False):gamma=gamma+1
    response=normalization_response(z,d,self.epsilon,center,gamma)
    stats=z.astype(jnp.float32)
    if center:stats=stats-stats.mean(-1,keepdims=True)
    isotropic=d.astype(jnp.float32)*gamma.astype(jnp.float32)*jax.lax.rsqrt(jnp.mean(stats*stats,-1,keepdims=True)+self.epsilon)
    context['moments']=jnp.concatenate([tangent_moments(z,d),jnp.sum(response*response,axis=(1,2))[:,None],jnp.sum(isotropic*isotropic,axis=(1,2))[:,None]],axis=-1)
   return value
  return norm
 for name in ['RMSNorm','LayerNorm']:
  cls=getattr(normalizations,name,None)
  if cls is not None:cls.__call__=wrap_norm(cls.__call__,name=='LayerNorm')
 native_block=fusion.SubDecoderLayer.__call__
 @functools.wraps(native_block)
 def block(self,*args,**kwargs):
  if CONTROLS is None or self.is_initializing():return native_block(self,*args,**kwargs)
  if not self.config.bam_enabled or self.config.dense_conn:raise ValueError('Probe supports native BAM blocks only')
  if self.config.decoder_block=='pythia' and self.config.pythia_parallel_residual:raise ValueError('FFN mediation requires serial residual blocks')
  layer=kwargs.get('layer_index')
  if layer is None:layer=self.layer_inx
  c=dict(layer=layer,x=args[0]);_BLOCK.append(c)
  try:
   result=native_block(self,*args,**kwargs)
   if 'delta' not in c or 'z' not in c:raise ValueError('Missing native branch direction')
   y=jax.lax.cond(c['active'],lambda v:(v+c['delta']).astype(v.dtype),lambda v:v,result[0])
   if OBSERVE:self.sow('intermediates','stream_mediation_tangent',c['moments'])
   if RETAIN_LOCAL:
    self.sow('intermediates','stream_mediation_layer_index',jnp.asarray(c['layer']))
    self.sow('intermediates','stream_mediation_before_exit',result[0])
    self.sow('intermediates','stream_mediation_delta',c['delta'])
    if result[1] is not None:self.sow('intermediates','stream_mediation_M',result[1])
   return (y,)+result[1:]
  finally:_BLOCK.pop()
 fusion.SubDecoderLayer.__call__=block

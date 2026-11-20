"""Isolated V5 ablations; existing kernels are numerical backends, never modified."""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp

NAMES = ('control', 'static_mlp_write', 'static_attention_write', 'static_all_write',
         'static_read', 'static_read_write', 'no_dynamic_gate', 'no_address_norm',
         'no_dynamic_gate_norm', 'no_shift', 'no_memory_mlp', 'no_local_o',
         'raw_attention_content', 'm_qk_partial_rope',
         'static_attention_write_no_gate', 'static_mlp_write_no_gate', 'static_read_write_no_gate',
         'static_all_write_no_gate')

@dataclass(frozen=True)
class Settings:
  name: str = ''
  def __post_init__(self):
    if self.name and self.name not in NAMES: raise ValueError(self.name)
  @property
  def static_read(self): return self.name in ('static_read', 'static_read_write', 'static_read_write_no_gate')
  def static_write(self, kind):
    return self.name in ('static_all_write','static_all_write_no_gate','static_read_write','static_read_write_no_gate') or self.name in ('static_'+kind+'_write','static_'+kind+'_write_no_gate')
  def ungated_write(self, kind):
    return self.no_gate or self.name == 'static_all_write_no_gate' or self.name == 'static_'+kind+'_write_no_gate'
  @property
  def no_gate(self): return self.name in ('no_dynamic_gate','no_dynamic_gate_norm','static_read_write_no_gate')
  @property
  def no_norm(self): return self.name in ('no_address_norm','no_dynamic_gate_norm')
  @property
  def no_o(self): return self.name == 'no_local_o'
  @property
  def m_rope(self): return self.name == 'm_qk_partial_rope'

def settings(cfg): return Settings(getattr(cfg, 'bam_v5_ablation', '') or '')

def address_norm(att, x):
  return x if settings(att.config).no_norm else att.write_address_norm(x)

def forward(att, x, positions, segments, M, is_global, standard_width, vo_scale, qk_scale,
            defer_write=False):
  """Common ablation graph with separately implemented native and existing-kernel reads."""
  from layers import bam_kernels as kernel
  from layers import normalizations
  from layers import attentions as a
  cfg=att.config; ab=settings(cfg); b,t=x.shape[:2]; n=att.num_query_heads; v=att.bam_v
  dtype=att.dtype
  empty = jnp.zeros((b,t,n,0),dtype)
  if ab.m_rope:
    qs=ks=empty
  else:
    qs=att.query_projection(x,standard_width)
    ks=att.kv_projection(x,proj_name='key',projection_dim=standard_width)
    qs=att.apply_rotary_embedding(qs,positions,name='query_rotary',embedding_dims=standard_width)
    ks=att.apply_rotary_embedding(ks,positions,name='key_rotary',embedding_dims=standard_width)
  hidden=None if ab.static_read else jax.nn.gelu(att.full_m_read_down(x))
  names={'q':'W_lq_c8','k':'W_lk_c8','v':'W_lv_c8','o':'W_R'}
  gates={'q':'W_lq_gate','k':'W_lk_gate','v':'W_lv_gate','o':'W_R_gate'}
  keys={};logits={};static={};sg={}
  for arm in 'qkov':
    disabled=ab.static_read or (ab.no_o and arm=='o')
    keys[arm]=(jnp.zeros((b,t,n,v),dtype) if disabled else
               att._project_column_read_key(names[arm],x,hidden).reshape(b,t,n,v))
    logits[arm]=(jnp.full((b,t,n),-jnp.inf if disabled else jnp.inf,dtype) if disabled or ab.no_gate
                 else att._project_read_gate_logits(gates[arm],x,squeeze_fetch_axis=arm=='o').reshape(b,t,n))
    static[arm]=(jnp.zeros((v,n),dtype) if ab.no_o and arm=='o' else
                 getattr(att,'static_'+arm+'_key').astype(dtype))
    sg[arm]=(jnp.ones((b,t,n),dtype) if ab.static_read or (arm in 'qk' and not cfg.bam_qk_static_read_gate)
             or (ab.no_o and arm=='o') else att._static_gate('general_'+arm+'_static_gate',x)/cfg.bam_static_read_gate_init)
  shift=None
  if att._m_shift_mode!='none':
    z=att.m_shift_gate(x).astype(jnp.float32)+att.m_shift_bias.astype(jnp.float32)
    gate=jax.nn.sigmoid(z)
    valid=a.bam_m_shift_valid_mask(x.shape,segments,positions)
    shift=(gate.astype(dtype)*valid[...,None]).astype(dtype)
  if cfg.bam_pallas_core:
    assert not ab.no_norm
    st=kernel.static_weight(*(static[z] for z in 'qkvo'),jnp.eye(v,dtype=dtype))
    # The existing read kernel expects a nonzero standard-input block. Supply a
    # constant-zero slab, then drop it; no W_Q/K parameters or kernel changes.
    qs_kernel = jnp.zeros((b,t,n,standard_width),dtype) if ab.m_rope else qs
    ks_kernel = qs_kernel if ab.m_rope else ks
    outs=kernel.read(M,st,keys['q'],logits['q'],keys['k'],logits['k'],
        jnp.concatenate((keys['v'],keys['o']),axis=-2),logits['o'],logits['v'],qs_kernel,ks_kernel,
        qk_cols=att._qk_col_width,read_epsilon=att._read_key_epsilon,key_scale=att._read_key_scale,
        static_gates=tuple(sg[z] for z in 'qkvo'),shift=shift,
        m_norm=att._rms_epsilon if att._m_read_norm=='rms' else None,
        forward_tile=int(cfg.bam_pallas_read_tile or 128),reverse_tile=int(cfg.bam_pallas_read_reverse_tile or 128),
        vmem_mib=cfg.bam_pallas_vmem_mib,interpret=bool(cfg.bam_pallas_interpret))
    q,k,value,local_o=outs[:4]; M=outs[4]
    if ab.m_rope:q,k=q[...,:att._qk_col_width],k[...,:att._qk_col_width]
  else:
    mr=M
    if shift is not None: mr=a.mix_bam_previous_token(M,jax.nn.sigmoid(z),segments,positions)
    mr=att._matrix_for_read(mr)
    reads={}
    for arm in 'qkvo':
      key=keys[arm]
      norm=(key if ab.no_norm else normalizations.rms_norm(key,dtype=key.dtype,
            epsilon=att._read_key_epsilon,statistics_dtype=att._read_rms_statistics_dtype))
      g=(att._read_key_scale*jax.nn.sigmoid(logits[arm]))[...,None]
      if arm in 'qk': dyn=jnp.einsum('btkv,btnv->btnk',mr,(norm*g).astype(dtype))
      else: dyn=(jnp.einsum('btkv,btnv->btnk',mr,norm)*g).astype(dtype)
      sta=jnp.einsum('btkv,vn->btnk',mr,static[arm])*sg[arm][...,None]
      reads[arm]=(dyn+sta).astype(dtype)
    q=jnp.concatenate((reads['q'],qs),axis=-1);k=jnp.concatenate((reads['k'],ks),axis=-1)
    value,local_o=reads['v'],reads['o']
  if qk_scale is not None:
    def scaleq(y):return jnp.concatenate(((y[...,:att._qk_col_width].astype(jnp.float32)*qk_scale).astype(dtype),y[...,att._qk_col_width:]),axis=-1)
    q,k=scaleq(q),scaleq(k)
  if vo_scale is not None:
    value=(value.astype(jnp.float32)*vo_scale).astype(dtype)
    local_o=(local_o.astype(jnp.float32)*vo_scale).astype(dtype)
  if ab.m_rope:
    def rope(y,name):
      r=standard_width
      return jnp.concatenate((y[...,:-r],att.apply_rotary_embedding(y[...,-r:],positions,name=name,embedding_dims=r)),axis=-1)
    q,k=rope(q,'query_rotary'),rope(k,'key_rotary')
  q=q/jnp.sqrt(att._attention_logit_scale_dim(q.shape[-1])).astype(dtype)
  window=t if is_global or att.sliding_window_size is None else min(t,int(att.sliding_window_size))
  if a._bam_splash_enabled(cfg,t):
    y=a._bam_splash_attention(q,k,value,segments,cfg,att.mesh,window=window,
                             seq_minor=a._bam_splash_seq_minor(cfg,False))
  elif att._query_chunk_size is not None:y,_=att._query_chunk_op(q,k,value,segments,window)
  else:y,_=att._attention_block(q,k,value,segments,q0=0,s0=0,window_size=window)
  content=y+local_o
  out=att.out_projection(x.shape[-1],content)
  if not att._has_write:return (out,M,None) if cfg.bam_pallas_core else (out,M)
  addr=att._write_address(x)
  wl=(jnp.full((b,t,n),jnp.inf,dtype) if ab.ungated_write('attention') else att.W_gw(x)+jnp.asarray(att.gw_b0,dtype))
  if cfg.bam_pallas_core:return out,M,(content,wl,addr)
  gate=jax.nn.sigmoid(wl)
  factors=(gate[...,None]*att._attention_write_content(content),address_norm(att,addr),gate)
  if defer_write:return out,M,factors
  dm=jnp.sum(factors[0][...,None]*factors[1][...,None,:],axis=-3)
  return out,M+dm

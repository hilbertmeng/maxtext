"""Residual Matrix Transformer in MaxText's MediumProp training runtime.

The residual stream is [batch, time, ResKey, ResVal].  This follows the
published RMT contractions and the open-source module layout, while the MLP,
optimizer, data and loss remain the matched MaxText backbone.

Selected Pallas path index (other modes are historical ablations):
  rmt_fused_attention_read -> rmt_pallas_attention_read (K1)
  rmt_fused_write_read_projection -> rmt_pallas_full_write_read (K2)
  minor_chunk64/minor_recompute -> rmt_pallas_full_write_read_minor (K2 backward)
  rmt_fused_projected_mlp_write -> rmt_pallas_projected_write (K3)
  rmt_rankh_write_mode != original -> rmt_pallas_rankh_write (new write ablations)
Full switch/helper index: experiments/bam_llama2_medium/rmt_rankh_write.md.
"""

import math

from flax import linen as nn
import jax
from jax import ad_checkpoint
import jax.numpy as jnp

import common_types
from layers import attentions, embeddings, initializers, linears, normalizations


RMT_DYNAMIC_HEALTH_NAMES = (
    *(f'{arm}_{stat}' for arm in ('q', 'k', 'v', 'o', 'mlp')
      for stat in ('dynamic_rms', 'static_rms', 'ratio')),
    *(f'{arm}_gate_{stat}' for arm in
      ('q', 'k', 'v', 'o', 'mlp', 'attn_write', 'mlp_write')
      for stat in ('mean', 'frac_gt_050')),
    *(f'{arm}_{part}_{stat}' for arm in ('attn_write', 'mlp_write')
      for part in ('first16', 'tail32') for stat in ('ratio', 'cosine')),
    'attn_input_first16_rms', 'attn_input_tail32_rms',
    'mlp_input_first16_rms', 'mlp_input_tail32_rms',
)

RMT_BOUNDARY_HEALTH_NAMES = (
    'dynamic_rms', 'static_rms', 'ratio', 'cosine',
    'gate_mean', 'gate_std', 'gate_frac_lt_005',
    'gate_frac_gt_050', 'gate_frac_gt_095',
)


def _boundary_health(dynamic, static, gate):
  dynamic, static, gate = (x.astype(jnp.float32) for x in (dynamic, static, gate))
  dynamic_rms, static_rms, ratio = _read_health(dynamic, static)
  cosine = jnp.mean(dynamic * static) / jnp.maximum(dynamic_rms * static_rms, 1e-12)
  return jnp.stack((dynamic_rms, static_rms, ratio, cosine,
                    jnp.mean(gate), jnp.std(gate), jnp.mean(gate < .05),
                    jnp.mean(gate > .5), jnp.mean(gate > .95)))


def dynamic_health_names(record_write_health=True, heads=16, key_dim=48):
  """Keep read/gate/state metrics when diagnostic write statistics are disabled."""
  return tuple(name.replace('first16', f'first{heads}').replace('tail32', f'tail{key_dim-heads}')
               for name in RMT_DYNAMIC_HEALTH_NAMES
               if record_write_health or not
               ('_write_' in name and name.endswith(('_ratio', '_cosine'))))


def _alibi_bias(num_heads, q0, q1, s0, s1, dtype=jnp.float32):
  """Published RMT slopes, in attention-head order."""
  slopes = jnp.geomspace(2.0 ** (-8.0 / num_heads), 2.0 ** -8, num_heads)
  distance = jnp.arange(q0, q1)[:, None] - jnp.arange(s0, s1)[None, :]
  return (-slopes[:, None, None] * distance[None]).astype(dtype)


def _rms(x):
  return jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))


def _read_health(dynamic, static):
  dynamic_rms, static_rms = _rms(dynamic), _rms(static)
  return (dynamic_rms, static_rms, dynamic_rms / jnp.maximum(static_rms, 1e-12))


def _write_health(dynamic, static):
  dynamic, static = dynamic.astype(jnp.float32), static.astype(jnp.float32)
  ratio = _rms(dynamic) / jnp.maximum(_rms(static), 1e-12)
  cosine = jnp.mean(dynamic * static) / jnp.maximum(_rms(dynamic) * _rms(static), 1e-12)
  return ratio, cosine


def _gate_health(gate):
  gate = gate.astype(jnp.float32)
  return jnp.mean(gate), jnp.mean(gate > 0.5)


def _init_gate_bias(opening):
  logit = math.log(opening / (1.0 - opening))
  return nn.initializers.constant(logit)


def _read_epsilon(cfg):
  return (cfg.normalization_layer_epsilon if cfg.bam_read_key_epsilon is None
          else cfg.bam_read_key_epsilon)


def _project_many(x,kernels,packed):
  if not packed:
    return tuple(jnp.einsum('btd,dr->btr',x,k.astype(x.dtype)) for k in kernels)
  widths=[k.shape[-1] for k in kernels]
  projected=jnp.einsum('btd,dr->btr',x,jnp.concatenate([k.astype(x.dtype) for k in kernels],axis=-1))
  projected=ad_checkpoint.checkpoint_name(projected,'rmt_dynamic_projection')
  cuts=[]
  offset=0
  for width in widths:
    cuts.append(projected[...,offset:offset+width])
    offset+=width
  return tuple(cuts)


class RMTDynamicQK(nn.Module):
  """BAM column-only shared rank-4 basis with independently gated Q/K routing."""
  config: common_types.Config

  @nn.compact
  def __call__(self, x, M, *, parameters_only=False):
    cfg=self.config
    heads=int(cfg.num_query_heads)
    address_dim=M.shape[-1]
    rank=4
    init=initializers.get_init_method(cfg.init_method)
    basis_kernel=self.param('basis_kernel',nn.with_logical_partitioning(init,('embed',None)),
                            (cfg.emb_dim,rank*address_dim),cfg.weight_dtype)
    basis_bias=self.param('basis_bias',nn.with_logical_partitioning(nn.initializers.zeros,(None,'kv')),
                          (rank,address_dim),cfg.weight_dtype)
    kernels=[basis_kernel]
    biases=[]
    for arm in ('q','k'):
      mix_kernel=self.param(f'{arm}_mix_kernel',nn.with_logical_partitioning(init,('embed',None)),
                            (cfg.emb_dim,heads*rank),cfg.weight_dtype)
      gate_kernel=self.param(f'{arm}_gate_kernel',nn.with_logical_partitioning(nn.initializers.zeros,('embed',None)),
                             (cfg.emb_dim,heads),cfg.weight_dtype)
      gate_bias=self.param(f'{arm}_gate_bias',nn.with_logical_partitioning(_init_gate_bias(.05),('q_heads',)),
                           (heads,),cfg.weight_dtype)
      kernels.extend((mix_kernel,gate_kernel));biases.append(gate_bias)
    if parameters_only:
      return tuple(z.astype(x.dtype) for z in
          (basis_kernel,basis_bias,kernels[1],kernels[2],biases[0],kernels[3],kernels[4],biases[1]))
    projected=_project_many(x,kernels,cfg.get_keys().get('rmt_pack_dynamic_projections',False))
    basis=projected[0].reshape(x.shape[:2]+(rank,address_dim))+basis_bias.astype(x.dtype)
    mixes=[projected[1+2*i].reshape(x.shape[:2]+(heads,rank)) for i in range(2)]
    gates=[jax.nn.sigmoid(projected[2+2*i]+biases[i].astype(x.dtype)) for i in range(2)]
    if cfg.get_keys().get('rmt_pallas_qk',False) and not self.is_initializing():
      from layers.rmt_pallas_qk import qk_read
      output=qk_read(jnp.swapaxes(M,-2,-1),basis,jnp.concatenate(mixes,axis=-2),
                     jnp.concatenate(gates,axis=-1),_read_epsilon(cfg),
                     tile=cfg.get_keys().get('rmt_pallas_qk_tile',32))
      results=(output[...,:heads,:],output[...,heads:,:])
    elif cfg.get_keys().get('rmt_pallas_qk_post',False) and not self.is_initializing():
      from layers.rmt_pallas_minor_qk import qk_post
      basis_read=ad_checkpoint.checkpoint_name(jnp.einsum('btvc,btrc->btrv',M,basis),'rmt_basis_read')
      output=qk_post(basis_read,basis,jnp.concatenate(mixes,axis=-2),
                     jnp.concatenate(gates,axis=-1),_read_epsilon(cfg),
                     tile=cfg.get_keys().get('rmt_pallas_qk_post_tile',128))
      results=(output[...,:heads,:],output[...,heads:,:])
    else:
      basis_read=ad_checkpoint.checkpoint_name(jnp.einsum('btvc,btrc->btrv',M,basis),'rmt_basis_read')
      basis_fp32=basis.astype(jnp.float32)
      gram=jnp.einsum('btrc,btsc->btrs',basis_fp32,basis_fp32)
      results=[]
      for mix,gate in zip(mixes,gates):
        mix_fp32=mix.astype(jnp.float32)
        norm2=jnp.einsum('btnr,btrs,btns->btn',mix_fp32,gram,mix_fp32)
        inverse_rms=jax.lax.rsqrt(norm2/address_dim+_read_epsilon(cfg))
        read=jnp.einsum('btrv,btnr->btnv',basis_read,mix)
        results.append(read*(inverse_rms.astype(read.dtype)*(.2*gate))[...,None])
    return results[0],results[1],gates[0],gates[1]


class RMTDynamicC8Read(nn.Module):
  """One dynamic key/read shared by destinations, with separate head gates."""

  config: common_types.Config
  destinations: int
  compress_state: bool = True

  @nn.compact
  def __call__(self, x, M, static_matrix=None, static_key=None, *, parameters_only=False):
    cfg = self.config
    pallas_joined = (static_matrix is not None and not self.is_initializing()
                     and cfg.get_keys().get('rmt_pallas_joined_read', False))
    heads = int(cfg.num_query_heads)
    key_dim = int(cfg.get_keys().get('rmt_dynamic_compression_dim', 8)) if self.compress_state else M.shape[-1]
    if parameters_only:
      if not self.compress_state or self.destinations != 1:
        raise ValueError('Fused MLP stage requires one compressed read destination')
      compression = self.param('compression', nn.with_logical_partitioning(nn.initializers.orthogonal(), ('v_factor', 'kv')),
                               (M.shape[-1], key_dim), cfg.weight_dtype)
      key_kernel = self.param('key_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                              (cfg.emb_dim, heads * key_dim), cfg.weight_dtype)
      gate_kernel = self.param('gate_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                               (cfg.emb_dim, heads), cfg.weight_dtype)
      gate_bias = self.param('gate_bias', nn.with_logical_partitioning(_init_gate_bias(.05), ('q_heads', None)),
                            (heads, 1), cfg.weight_dtype)
      return compression,key_kernel,gate_kernel,gate_bias
    if self.compress_state:
      compression = self.param('compression', nn.with_logical_partitioning(nn.initializers.orthogonal(), ('v_factor', 'kv')),
                               (M.shape[-1], key_dim), cfg.weight_dtype)
      if static_matrix is not None:
        leading = static_matrix.shape[-2] - M.shape[-1]
        padded = jnp.pad(compression.astype(M.dtype), ((leading, 0), (0, 0)))
        projection = jnp.concatenate((static_key, padded), axis=-1)
        if not pallas_joined:
          joined = jnp.einsum('btkv,kn->btnv', static_matrix, projection)
          static_read = joined[..., :static_key.shape[-1], :]
          compressed = jnp.swapaxes(joined[..., static_key.shape[-1]:, :], -2, -1)
      else:
        compressed = jnp.einsum('btvc,cr->btvr', M, compression.astype(M.dtype))
    else:
      compressed = M
    if self.compress_state and not pallas_joined:
      compressed=ad_checkpoint.checkpoint_name(compressed,'rmt_compressed_read')
    key_kernel = self.param('key_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                            (cfg.emb_dim, heads * key_dim), cfg.weight_dtype)
    gate_kernel = self.param('gate_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                             (cfg.emb_dim, heads * self.destinations), cfg.weight_dtype)
    gate_bias = self.param('gate_bias', nn.with_logical_partitioning(_init_gate_bias(.05), ('q_heads', None)),
                           (heads, self.destinations), cfg.weight_dtype)
    raw_key,logits = _project_many(x,(key_kernel,gate_kernel),cfg.get_keys().get('rmt_pack_dynamic_projections',False))
    raw_key = raw_key.reshape(x.shape[:2] + (heads, key_dim))
    pallas_c8=(cfg.get_keys().get('rmt_pallas_c8',False) and self.compress_state and not self.is_initializing())
    if not pallas_joined and not pallas_c8:
      key = normalizations.rms_norm(
          raw_key, dtype=x.dtype, epsilon=_read_epsilon(cfg),
          statistics_dtype=jnp.float32)
      read = jnp.einsum('btvc,btnc->btnv', compressed, key)
    logits = logits.reshape(x.shape[:2] + (heads, self.destinations))
    gates = jax.nn.sigmoid(logits + gate_bias.astype(x.dtype))
    if pallas_c8:
      if (self.destinations == 1 and self.name == 'dynamic_vo'
          and cfg.get_keys().get('rmt_pallas_v_only', False)):
        from layers.rmt_pallas_v_read import v_read as c8_read
      else:
        from layers.rmt_pallas_minor_read import c8_read
      values=c8_read(jnp.swapaxes(compressed,-2,-1),raw_key,gates,_read_epsilon(cfg),
                     tile=cfg.get_keys().get('rmt_pallas_c8_tile',128))
      reads=tuple(values[...,i,:] for i in range(self.destinations))
    elif pallas_joined:
      if cfg.get_keys().get('rmt_pallas_joined_layout')=='token_minor':
        from layers.rmt_pallas_minor_joined import joined_read
      else:
        from layers.rmt_pallas_joined import joined_read
      static_read, dynamic_reads = joined_read(
          static_matrix, raw_key, projection, gates, _read_epsilon(cfg),
          tile=cfg.get_keys().get('rmt_pallas_read_tile',64))
      reads = tuple(dynamic_reads[..., i, :] for i in range(self.destinations))
    else:
      reads = tuple(.2 * gates[..., i, None] * read for i in range(self.destinations))
    if static_matrix is not None:
      return reads, gates, static_read
    return reads, gates


class RMTDynamicWrite(nn.Module):
  """BAM GELU-LoRA address and normalized outer write into 32 or 48 RMT rows."""

  config: common_types.Config
  address_dim: int

  @nn.compact
  def __call__(self, x, data, matrix=None, static_key=None, padded_value_dim=None, *, address_only=False, parameters_only=False):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    bottleneck = int(cfg.get_keys().get('rmt_dynamic_write_bottleneck_dim', 256))
    init = initializers.get_init_method(cfg.init_method)
    down = self.param('address_down', nn.with_logical_partitioning(init, ('embed', None)),
                      (cfg.emb_dim, bottleneck), cfg.weight_dtype)
    up = self.param('address_up', nn.with_logical_partitioning(init, ('embed', None)),
                    (bottleneck, heads * self.address_dim), cfg.weight_dtype)
    up_bias = self.param('address_up_bias', nn.with_logical_partitioning(nn.initializers.zeros, ('q_heads', None)),
                         (heads, self.address_dim), cfg.weight_dtype)
    gate_kernel = self.param('gate_kernel', nn.with_logical_partitioning(init, ('embed', None)),
                             (cfg.emb_dim, heads), cfg.weight_dtype)
    gate_bias = self.param('gate_bias', nn.with_logical_partitioning(_init_gate_bias(.1), ('q_heads',)),
                           (heads,), cfg.weight_dtype)
    if parameters_only:
      return tuple(w.astype(x.dtype) for w in (down,up,up_bias,gate_kernel,gate_bias))
    if (self.name=='dynamic_mlp_write' and cfg.get_keys().get('rmt_fused_projected_mlp_write',False)
        and matrix is not None and not self.is_initializing()):
      if cfg.rmt_record_dynamic_health or self.address_dim!=48 or padded_value_dim:
        raise ValueError('Projected write fusion requires unpadded Full48 and health OFF')
      from layers.rmt_pallas_projected_write import projected_write
      output=projected_write(matrix,x,data,static_key,down.astype(x.dtype),up.astype(x.dtype),
          up_bias.astype(x.dtype),gate_kernel.astype(x.dtype),gate_bias.astype(x.dtype),
          cfg.normalization_layer_epsilon,
          forward_tile=cfg.get_keys().get('rmt_projected_write_forward_tile',128),
          reverse_tile=cfg.get_keys().get('rmt_projected_write_reverse_tile',32),
          write_mode=cfg.get_keys().get('rmt_rankh_write_mode','original'))
      return output,None
    hidden,gate_logits = _project_many(x,(down,gate_kernel),cfg.get_keys().get('rmt_pack_dynamic_projections',False))
    hidden = nn.gelu(hidden)
    address = jnp.einsum('btr,rd->btd', hidden, up.astype(x.dtype))
    address = address.reshape(x.shape[:2] + (heads, self.address_dim))
    address = ad_checkpoint.checkpoint_name(address + up_bias.astype(x.dtype),'rmt_dynamic_address')
    gate = jax.nn.sigmoid(gate_logits + gate_bias.astype(x.dtype))
    if address_only:
      return address,gate
    if matrix is not None and not self.is_initializing():
      if cfg.get_keys().get('rmt_pallas_write_layout','value_minor')=='token_minor':
        from layers.rmt_pallas_minor import write_residual
        return write_residual(matrix,address,data,gate,static_key,cfg.normalization_layer_epsilon,
                              tile=cfg.get_keys().get('rmt_pallas_tile',128),
                              key_contiguous=cfg.get_keys().get('rmt_pallas_key_contiguous',False),
                              backward=cfg.get_keys().get('rmt_pallas_write_backward','autodiff')),gate
      from layers.rmt_pallas import write_residual
      return write_residual(matrix, address, data, gate, static_key,
                            cfg.normalization_layer_epsilon,
                            tile=cfg.get_keys().get('rmt_pallas_tile',16),
                            forward_jax=cfg.get_keys().get('rmt_pallas_write_forward_jax',False)), gate
    raw_data = data
    address = normalizations.rms_norm(
        address, dtype=address.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    data = normalizations.rms_norm(
        data, dtype=data.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    if padded_value_dim is not None:
      data = jnp.pad(data, ((0,0),(0,0),(0,0),(0,padded_value_dim-data.shape[-1])))
    write = jnp.einsum('btnk,btnv->btkv', gate[..., None] * address, data)
    if matrix is not None:
      # Initialization must also run on the CPU without a TPU-only custom call.
      return matrix + jnp.einsum('btnv,nk->btkv', raw_data, static_key) + write, gate
    return write, gate


class RMTVectorNormParameters(nn.Module):
  """Retrieve the existing RMSNorm scale under its unchanged parameter scope."""
  config: common_types.Config

  @nn.compact
  def __call__(self):
    cfg=self.config
    initializer=nn.initializers.ones if cfg.direct_scale else nn.initializers.zeros
    scale=self.param('scale',nn.with_logical_partitioning(initializer,('norm',)),
                     (cfg.emb_dim,),cfg.weight_dtype).astype(cfg.dtype)
    return scale if cfg.direct_scale else scale+1


class MatrixRMSNorm(nn.Module):
  """RMT source RMSNorm over both residual-matrix axes, with full-size gain."""

  config: common_types.Config

  @nn.compact
  def __call__(self, matrix):
    cfg = self.config
    gain = self.param('scale', nn.initializers.ones,
                      matrix.shape[-2:], cfg.weight_dtype)
    mean_square = jnp.mean(jnp.square(matrix.astype(jnp.float32)),
                           axis=(-2, -1), keepdims=True)
    normalized = matrix.astype(jnp.float32) * jax.lax.rsqrt(
        mean_square + cfg.normalization_layer_epsilon)
    return (normalized * gain.astype(jnp.float32)).astype(cfg.dtype)


class RMTLayer(nn.Module):
  """RMT layer with optional BAM-style dynamic reads and writes."""

  config: common_types.Config
  quant: object = None
  mlp_dim: int | None = None

  @nn.compact
  def __call__(self, matrix, segment_ids, positions, deterministic, layer_index):
    cfg = self.config
    scan_minor = cfg.get_keys().get('rmt_scan_token_minor', False)
    if scan_minor:
      matrix = matrix.transpose(0,3,1,2)
    unsupported = ('rmt_llf_enabled' , 'rmt_fetch_independent_o_key',
                   'rmt_single_outer_write', 'rmt_headwise_mlp',
                   'rmt_transposed_matrix_carry', 'rmt_static_mlp_read_pre_norm',
                   'rmt_dynamic_read_full_matrix',
                   'rmt_write_health_row_reduce', 'rmt_write_health_reuse_input_rms')
    if any(cfg.get_keys().get(key, False) for key in unsupported):
      raise ValueError('This RMT variant is ledger only; use its recorded runtime worktree')
    for key in ('rmt_dynamic_mlp_enabled', 'rmt_dynamic_mlp_read_enabled',
                'rmt_dynamic_mlp_write_enabled', 'rmt_static_write_enabled'):
      if not cfg.get_keys().get(key, True):
        raise ValueError(f'{key}=False is ledger only; use its recorded runtime worktree')
    if cfg.get_keys().get('rmt_write_contraction', 'dot') != 'dot':
      raise ValueError('Only the trained dot write contraction is merged')
    heads = int(cfg.num_query_heads)
    value_dim = int(cfg.head_dim)
    key_dim = int(cfg.rmt_reskey_dim)
    assert cfg.emb_dim == heads * value_dim
    dynamic = bool(getattr(cfg, 'rmt_dynamic_enabled', False))
    joined_read = bool(cfg.get_keys().get('rmt_join_static_compression', False))
    if joined_read and not dynamic:
      raise ValueError('Joined static/compression reads require dynamic RMT')
    pallas_write = bool(cfg.get_keys().get('rmt_pallas_write', False))
    if pallas_write and (not dynamic or cfg.rmt_dynamic_write_rows != 48
                         or cfg.rmt_record_dynamic_health):
      raise ValueError('Pallas write prototype requires dynamic Full48 and health OFF')
    padded_value_dim = cfg.get_keys().get('rmt_pad_value_dim', 0)
    if padded_value_dim and (pallas_write or joined_read or cfg.rmt_record_dynamic_health
                             or not cfg.rmt_vector_pre_norm):
      raise ValueError('Padded carry prototype requires standalone vector-norm health-OFF path')
    dynamic_mlp_read_enabled = dynamic
    dynamic_mlp_write_enabled = dynamic
    dynamic_o_enabled = bool(cfg.get_keys().get('rmt_dynamic_o_enabled', True))
    rope_qk_dim = int(cfg.get_keys().get('rmt_rope_qk_dim', 0))
    vector_pre_norm = bool(cfg.get_keys().get('rmt_vector_pre_norm', False))
    if vector_pre_norm and not dynamic:
      raise ValueError('RMT vector pre-norm requires the dynamic arm')
    if rope_qk_dim and (not dynamic or rope_qk_dim % 2 or rope_qk_dim >= value_dim):
      raise ValueError('RMT RoPE Q/K requires a dynamic arm and an even proper subspace')
    write_rows = int(getattr(cfg, 'rmt_dynamic_write_rows', 32)) if dynamic else 0
    if dynamic and (key_dim <= heads or write_rows not in (key_dim - heads, key_dim)):
      raise ValueError('Dynamic RMT requires tail-row or full-row writes and nonempty tail')
    key_init = nn.initializers.normal(key_dim ** -0.5)
    write_init = nn.initializers.normal(heads ** -0.5 / math.sqrt(2 * cfg.num_decoder_layers))

    attn_in = (matrix if vector_pre_norm else
               MatrixRMSNorm(cfg, name='attn_norm')(matrix))
    qkv_key = self.param('qkv_key', key_init, (3, heads, key_dim), cfg.weight_dtype)
    fused_attention=bool(cfg.get_keys().get('rmt_fused_attention_read',False))
    if fused_attention and (not dynamic or not vector_pre_norm or dynamic_o_enabled
                            or joined_read or padded_value_dim or rope_qk_dim!=18
                            or cfg.rmt_record_dynamic_health):
      raise ValueError('Complete attention fusion requires NoO vector norm, RoPE18 and health OFF')
    fused_attention=fused_attention and not self.is_initializing()
    read_start=heads
    if fused_attention:
      proxy=attn_in[...,:heads,:].reshape(attn_in.shape[:2]+(cfg.emb_dim,))
      scale=RMTVectorNormParameters(cfg,name='attn_vector_norm')()
      tail=jnp.swapaxes(attn_in[...,heads:,:],-2,-1)
      bw,bb,qm,qg,qb,km,kg,kb=RMTDynamicQK(cfg,name='dynamic_qk')(proxy,tail,parameters_only=True)
      compression,vk,vg,vb=RMTDynamicC8Read(cfg,destinations=1,name='dynamic_vo')(
          proxy,tail,parameters_only=True)
      rope_kernels=[]
      for arm in ('q','k'):
        rope_kernels.append(self.param(f'{arm}_rope_kernel',
            nn.with_logical_partitioning(initializers.get_init_method(cfg.init_method),('embed',None)),
            (cfg.emb_dim,heads*rope_qk_dim),cfg.weight_dtype).astype(cfg.dtype))
      packed=jnp.concatenate((bw,qm,km,qg,kg,vk.astype(cfg.dtype),vg.astype(cfg.dtype),*rope_kernels),axis=1)
      gate_bias=jnp.concatenate((qb,kb,vb[:,0].astype(cfg.dtype)))
      from layers.rmt_pallas_attention_read import attention_read
      qkv,attn_x=attention_read(attn_in,qkv_key.reshape(3*heads,key_dim).astype(cfg.dtype),
          compression.astype(cfg.dtype),scale,packed,bb,gate_bias,positions,
          cfg.normalization_layer_epsilon,_read_epsilon(cfg),rope_qk_dim,
          cfg.rope_min_timescale,cfg.rope_max_timescale,
          forward_tile=cfg.get_keys().get('rmt_attention_read_forward_tile',128),
          reverse_tile=cfg.get_keys().get('rmt_attention_read_reverse_tile',128),
          save_small=cfg.get_keys().get('rmt_attention_save_small',False))
      query,key,value=qkv[...,:heads,:],qkv[...,heads:2*heads,:],qkv[...,2*heads:,:]
    if not joined_read and not fused_attention:
      qkv = jnp.einsum('btkv,ank->abtnv', attn_in, qkv_key.astype(cfg.dtype))
      query, key, value = (qkv[i][..., :value_dim] for i in range(3))
    if dynamic and not fused_attention:
      attn_x = attn_in[..., :heads, :value_dim].reshape(attn_in.shape[:2] + (cfg.emb_dim,))
      if vector_pre_norm:
        attn_x = normalizations.get_rmsnorm('attn_vector_norm', cfg)(attn_x)
      read_start = heads
      attn_M = jnp.swapaxes(attn_in[..., read_start:, :], -2, -1)
      dynamic_q, dynamic_k, q_gate, k_gate = RMTDynamicQK(
          cfg, name='dynamic_qk')(attn_x, attn_M)
      vo_module = RMTDynamicC8Read(cfg, destinations=2 if dynamic_o_enabled else 1,
                                  name='dynamic_vo')
      if joined_read:
        vo_reads, vo_gates, qkv = vo_module(
            attn_x, attn_M, attn_in, qkv_key.astype(cfg.dtype).reshape(3 * heads, key_dim).T)
        qkv = qkv.reshape(attn_in.shape[:2] + (3, heads, value_dim))
        query, key, value = (qkv[:, :, i] for i in range(3))
      else:
        vo_reads, vo_gates = vo_module(attn_x, attn_M)
      dynamic_q, dynamic_k = dynamic_q[..., :value_dim], dynamic_k[..., :value_dim]
      dynamic_v = vo_reads[0][..., :value_dim]
      if dynamic_o_enabled:
        dynamic_o = vo_reads[1][..., :value_dim]
      static_q, static_k, static_v = query, key, value
      query = query + dynamic_q
      key = key + dynamic_k
      value = value + dynamic_v
      if rope_qk_dim:
        rope_qk = []
        for arm in ('q', 'k'):
          kernel = self.param(
              f'{arm}_rope_kernel',
              nn.with_logical_partitioning(initializers.get_init_method(cfg.init_method),
                                           ('embed', None)),
              (cfg.emb_dim, heads * rope_qk_dim), cfg.weight_dtype)
          projected = jnp.einsum('btd,dr->btr', attn_x, kernel.astype(attn_x.dtype))
          projected = projected.reshape(attn_x.shape[:2] + (heads, rope_qk_dim))
          rope_qk.append(embeddings.RotaryEmbedding(
              min_timescale=cfg.rope_min_timescale,
              max_timescale=cfg.rope_max_timescale,
              embedding_dims=rope_qk_dim,
              fprop_dtype=cfg.dtype,
              rope_half=False,
              name=f'{arm}_rope')(projected, positions))
        query = jnp.concatenate((query[..., :-rope_qk_dim], rope_qk[0]), axis=-1)
        key = jnp.concatenate((key[..., :-rope_qk_dim], rope_qk[1]), axis=-1)
    if not fused_attention:query = query / math.sqrt(value_dim)
    t = matrix.shape[1]
    chunk = int(cfg.query_chunk_size)
    assert t % chunk == 0
    outputs = []
    record_health = dynamic and bool(getattr(cfg, 'rmt_record_dynamic_health', False))
    for q0 in range(0, t, chunk):
      q1 = q0 + chunk
      source = jnp.arange(q1)[None, :]
      target = jnp.arange(q0, q1)[:, None]
      valid = (source <= target)[None]
      if segment_ids is not None:
        valid &= (segment_ids[:, q0:q1, None] == segment_ids[:, None, :q1])
      if cfg.get_keys().get('rmt_remat_policy','full')=='attention_only':
        # Same attention equations and chunks. Limit recomputation to the
        # large quadratic intermediates; retain linear-size matrix-flow work.
        def attention_chunk(q,k,v,segments):
          # Store the shared full Q/K/V once, not every overlapping prefix
          # or quadratic boolean mask as separate checkpoint arguments.
          source=jnp.arange(q1)[None,:]
          target=jnp.arange(q0,q1)[:,None]
          mask=(source<=target)[None]
          if segments is not None:
            mask &= (segments[:,q0:q1,None]==segments[:,None,:q1])
          return attentions._attention_op(q[:,q0:q1],k[:,:q1],v[:,:q1],mask,
              float32_logits=cfg.float32_logits if rope_qk_dim else True,
              additive_bias=(None if rope_qk_dim else _alibi_bias(heads,q0,q1,0,q1)))[0]
        y=jax.checkpoint(attention_chunk,prevent_cse=True)(query,key,value,segment_ids)
      else:
        y, alpha = attentions._attention_op(
            query[:, q0:q1], key[:, :q1], value[:, :q1], valid,
            float32_logits=cfg.float32_logits if rope_qk_dim else True,
            additive_bias=(None if rope_qk_dim else
                           _alibi_bias(heads, q0, q1, 0, q1)))
      outputs.append(y)
    head_output = jnp.concatenate(outputs, axis=1).astype(cfg.dtype)
    if dynamic:
      static_head_output = head_output
      if dynamic_o_enabled:
        head_output = head_output + dynamic_o
    if cfg.get_keys().get('rmt_remat_policy','full') in ('save_dense_state','save_state','save_state_mlp','save_state_dynamic'):
      head_output=ad_checkpoint.checkpoint_name(head_output,'rmt_attention_head')
    attn_write = self.param('attn_write_key', write_init,
                            (heads, key_dim), cfg.weight_dtype)
    write_data = (jnp.pad(head_output, ((0,0),(0,0),(0,0),(0,padded_value_dim-value_dim)))
                  if padded_value_dim else head_output)
    fused_stage = bool(cfg.get_keys().get('rmt_fused_write_mlp_read', False))
    if fused_stage and (not dynamic or not vector_pre_norm or write_rows != 48
                        or joined_read or padded_value_dim or cfg.rmt_record_dynamic_health):
      raise ValueError('Fused write/read requires Full48 vector norm, unpadded state, health OFF')
    if fused_stage and not self.is_initializing():
      full_projection=bool(cfg.get_keys().get('rmt_fused_write_read_projection',False))
      write_parameters=RMTDynamicWrite(cfg,write_rows,name='dynamic_attn_write')(
          attn_x,head_output,address_only=not full_projection,parameters_only=full_projection)
      mlp_read=self.param('mlp_read_key',key_init,(key_dim,heads),cfg.weight_dtype)
      scale=RMTVectorNormParameters(cfg,name='mlp_vector_norm')()
      compression,wk,wg,bias=RMTDynamicC8Read(cfg,destinations=1,name='dynamic_mlp_read')(
          attn_x,jnp.swapaxes(matrix[...,heads:,:],-2,-1),parameters_only=True)
      read_parameters=(mlp_read.astype(cfg.dtype),compression.astype(cfg.dtype),scale,
                       wk.astype(cfg.dtype),wg.astype(cfg.dtype),bias[...,0].astype(cfg.dtype))
      if full_projection:
        from layers.rmt_pallas_full_write_read import full_write_read
        matrix,vector,mlp_x=full_write_read(
            matrix,attn_x,head_output,attn_write.astype(cfg.dtype),*write_parameters,*read_parameters,
            cfg.normalization_layer_epsilon,_read_epsilon(cfg),
            forward_tile=cfg.get_keys().get('rmt_fused_write_read_tile',128),
            reverse_tile=cfg.get_keys().get('rmt_fused_write_read_backward_tile',32),
            reverse_mode=cfg.get_keys().get('rmt_full_middle_reverse_mode','baseline'),
            save_native_outputs=cfg.get_keys().get('rmt_save_middle_native_outputs',False),
            write_mode=cfg.get_keys().get('rmt_rankh_write_mode','original'))
      else:
        from layers.rmt_pallas_write_read import write_mlp_read
        address,attn_write_gate=write_parameters
        matrix,vector,mlp_x=write_mlp_read(
            matrix,address,head_output,attn_write_gate,attn_write.astype(cfg.dtype),*read_parameters,
            cfg.normalization_layer_epsilon,_read_epsilon(cfg),
            tile=cfg.get_keys().get('rmt_fused_write_read_tile',128),
            buffers=cfg.get_keys().get('rmt_fused_write_read_buffers',1),
            backward_tile=cfg.get_keys().get('rmt_fused_write_read_backward_tile',0),
            backward_compute_tile=cfg.get_keys().get('rmt_fused_write_read_compute_tile',0))
      if cfg.get_keys().get('rmt_remat_policy','full') in ('save_dense_state','save_state','save_state_mlp','save_state_dynamic'):
        matrix=ad_checkpoint.checkpoint_name(matrix,'rmt_mlp_matrix')
    else:
      static_attn_write = jnp.einsum(
          'btnv,nk->btkv', write_data, attn_write.astype(cfg.dtype))
      if dynamic:
        dynamic_attn_write, attn_write_gate = RMTDynamicWrite(
            cfg, write_rows, name='dynamic_attn_write')(
                attn_x, head_output, matrix if pallas_write else None,
                attn_write.astype(cfg.dtype) if pallas_write else None,
                padded_value_dim or None)
        if write_rows != key_dim:
          dynamic_attn_write = jnp.pad(dynamic_attn_write,
                                       ((0, 0), (0, 0), (heads, 0), (0, 0)))
        matrix = dynamic_attn_write if pallas_write else matrix + static_attn_write + dynamic_attn_write
      elif not dynamic:
        matrix = matrix + static_attn_write

      if cfg.get_keys().get('rmt_remat_policy','full') in ('save_dense_state','save_state','save_state_mlp','save_state_dynamic'):
        matrix=ad_checkpoint.checkpoint_name(matrix,'rmt_mlp_matrix')
      mlp_in = (matrix if vector_pre_norm else
                MatrixRMSNorm(cfg, name='mlp_norm')(matrix))
      mlp_read = self.param('mlp_read_key', key_init,
                            (key_dim, heads), cfg.weight_dtype)
      if not joined_read:
        vector = jnp.einsum('btkv,kn->btnv', mlp_in, mlp_read.astype(cfg.dtype))
        vector = vector[..., :value_dim]
        static_mlp_read = vector
      if dynamic_mlp_read_enabled or dynamic_mlp_write_enabled:
        mlp_x = mlp_in[..., :heads, :value_dim].reshape(mlp_in.shape[:2] + (cfg.emb_dim,))
        if vector_pre_norm:
          mlp_x = normalizations.get_rmsnorm('mlp_vector_norm', cfg)(mlp_x)
      if dynamic_mlp_read_enabled:
        mlp_M = jnp.swapaxes(mlp_in[..., read_start:, :], -2, -1)
        mlp_read_module = RMTDynamicC8Read(cfg, destinations=1, name='dynamic_mlp_read')
        if joined_read:
          (dynamic_mlp_read,), mlp_read_gate, vector = mlp_read_module(
              mlp_x, mlp_M, mlp_in, mlp_read.astype(cfg.dtype))
          static_mlp_read = vector
        else:
          (dynamic_mlp_read,), mlp_read_gate = mlp_read_module(mlp_x, mlp_M)
        vector = vector + dynamic_mlp_read[..., :value_dim]
    if cfg.get_keys().get('rmt_save_middle_outputs',False) and not cfg.get_keys().get('rmt_save_middle_native_outputs',False):
      vector=ad_checkpoint.checkpoint_name(vector,'rmt_middle_vector')
      mlp_x=ad_checkpoint.checkpoint_name(mlp_x,'rmt_middle_proxy')
    vector = vector.reshape(vector.shape[:2] + (cfg.emb_dim,))
    vector = linears.MlpBlock(
        config=cfg, intermediate_dim=cfg.mlp_dim if self.mlp_dim is None else self.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=initializers.get_init_method(cfg.init_method),
        quant=self.quant, name='mlp')(vector, deterministic=deterministic)
    vector = vector.reshape(vector.shape[:2] + (heads, value_dim))
    mlp_write = self.param('mlp_write_key', write_init,
                           (heads, key_dim), cfg.weight_dtype)
    write_data = (jnp.pad(vector, ((0,0),(0,0),(0,0),(0,padded_value_dim-value_dim)))
                  if padded_value_dim else vector)
    static_mlp_write = jnp.einsum(
        'btnv,nk->btkv', write_data, mlp_write.astype(cfg.dtype))
    if dynamic_mlp_write_enabled:
      dynamic_mlp_write, mlp_write_gate = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_mlp_write')(
              mlp_x, vector, matrix if pallas_write else None,
              mlp_write.astype(cfg.dtype) if pallas_write else None,
              padded_value_dim or None)
      if write_rows != key_dim:
        dynamic_mlp_write = jnp.pad(dynamic_mlp_write,
                                    ((0, 0), (0, 0), (heads, 0), (0, 0)))
      matrix = dynamic_mlp_write if pallas_write else matrix + static_mlp_write + dynamic_mlp_write
    elif not dynamic_mlp_write_enabled:
      matrix = matrix + static_mlp_write
    health = None
    if dynamic and getattr(cfg, 'rmt_record_dynamic_health', False):
      # Keep the health schema identical for matched Full48/NoO comparisons.
      measured_o = dynamic_o if dynamic_o_enabled else jnp.zeros_like(static_head_output)
      measured_o_gate = (vo_gates[..., 1] if dynamic_o_enabled
                         else jnp.zeros_like(vo_gates[..., 0]))
      if not dynamic_mlp_read_enabled:
        dynamic_mlp_read = jnp.zeros_like(static_mlp_read)
        mlp_read_gate = jnp.zeros_like(vo_gates[..., :1])
      if not dynamic_mlp_write_enabled:
        dynamic_mlp_write = jnp.zeros_like(static_mlp_write)
        mlp_write_gate = jnp.zeros_like(attn_write_gate)
      reads = ((dynamic_q, static_q), (dynamic_k, static_k),
               (dynamic_v, static_v), (measured_o, static_head_output),
               (dynamic_mlp_read, static_mlp_read))
      gates = (q_gate, k_gate, vo_gates[..., 0], measured_o_gate,
               mlp_read_gate[..., 0], attn_write_gate, mlp_write_gate)
      values = [v for pair in reads for v in _read_health(*pair)]
      values.extend(v for gate in gates for v in _gate_health(gate))
      record_write_health = cfg.get_keys().get('rmt_record_write_health', True)
      if record_write_health:
        writes = ((dynamic_attn_write, static_attn_write),
                  (dynamic_mlp_write, static_mlp_write))
        values.extend(v for dyn, stat in writes for part in
                      (slice(None, heads), slice(heads, None))
                      for v in _write_health(dyn[..., part, :], stat[..., part, :]))
      values.extend((_rms(attn_in[..., :heads, :]), _rms(attn_in[..., heads:, :]),
                     _rms(mlp_in[..., :heads, :]), _rms(mlp_in[..., heads:, :])))
      assert len(values) == len(dynamic_health_names(record_write_health))
      health = jnp.stack(values)
      if not cfg.get_keys().get('rmt_block_scan', False):
        self.sow('intermediates', 'rmt_dynamic_health', health)
    if scan_minor:
      matrix = matrix.transpose(0,2,3,1)
    return matrix, (
        health if cfg.get_keys().get('rmt_block_scan', False) else None)


class RMTBlock(nn.Module):
  """Three all-local matrix layers, preserving the trained 18-layer scan tree."""

  config: common_types.Config
  quant: object = None

  @nn.compact
  def __call__(self, matrix, segment_ids, positions, deterministic, block_index):
    cfg = self.config
    widths = cfg.rmt_mlp_dim_by_block
    health = []
    Layer = nn.remat(RMTLayer, prevent_cse=True, static_argnums=(4,))
    for offset in range(3):
      matrix, stats = Layer(
          cfg, quant=self.quant, mlp_dim=int(widths[offset]),
          name=f'layer_{offset}')(
              matrix, segment_ids, positions, deterministic, 3 * block_index + offset)
      if stats is not None:
        health.append(stats)
    if health:
      self.sow('intermediates', 'rmt_dynamic_health', jnp.stack(health))
    return matrix, None


class RMTDecoder(nn.Module):
  """Native matrix-only RMT stream with the shared MaxText LM loss head."""

  config: common_types.Config
  shared_embedding: nn.Module
  deep_embedding: nn.Module
  mesh: common_types.Mesh
  quant: object = None

  @nn.compact
  def __call__(self, *, decoder_input_tokens, decoder_positions,
               decoder_segment_ids, decoder_target_tokens,
               decoder_target_mask, deterministic, model_mode):
    from layers import models  # Avoid an import cycle during Transformer setup.
    cfg = self.config
    if cfg.deep_embed_init == 'outside' or cfg.mtp_num_layers:
      raise ValueError('RMT comparison does not support deep embedding or MTP')
    if model_mode != common_types.MODEL_MODE_TRAIN:
      raise ValueError('RMT comparison currently supports training only')
    heads = int(cfg.num_query_heads)
    value_dim = int(cfg.head_dim)
    key_dim = int(cfg.rmt_reskey_dim)
    embedding = self.shared_embedding(decoder_input_tokens.astype(jnp.int32))
    embedding = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        embedding, deterministic=deterministic).astype(cfg.dtype)
    embedded_heads = embedding.reshape(embedding.shape[:2] + (heads, value_dim))
    seed_key = self.param('seed_key', nn.initializers.normal(heads ** -0.5),
                          (heads, key_dim), cfg.weight_dtype)
    matrix = jnp.einsum('btnv,nk->btkv', embedded_heads, seed_key.astype(cfg.dtype))
    if cfg.get_keys().get('rmt_dynamic_embedding_write', False):
      # Same learned content and R256 GELU address as EmbeddingBamWrite,
      # with the native RMT address/content axes (48, 75).
      data = linears.DenseGeneral(
          features=(heads, value_dim), axis=-1,
          kernel_init=initializers.get_init_method(cfg.init_method),
          kernel_axes=('embed', 'q_heads', 'v_factor'), dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype, name='embedding_write_content',
          quant=self.quant, matmul_precision=cfg.matmul_precision,
          use_bias=False)(embedding)
      dynamic_seed, seed_gate = RMTDynamicWrite(
          cfg, address_dim=key_dim, name='dynamic_embedding_write')(embedding, data)
      if cfg.get_keys().get('rmt_record_dynamic_health', False):
        self.sow('intermediates', 'rmt_embedding_health',
                 _boundary_health(dynamic_seed, matrix, seed_gate))
      matrix = matrix + dynamic_seed
    padded_value_dim = cfg.get_keys().get('rmt_pad_value_dim', 0)
    if padded_value_dim:
      if padded_value_dim < value_dim:raise ValueError('Padded value dimension must cover all values')
      matrix = jnp.pad(matrix, ((0,0),(0,0),(0,0),(0,padded_value_dim-value_dim)))
    block_scan = cfg.get_keys().get('rmt_block_scan', False)
    scan_minor = cfg.get_keys().get('rmt_scan_token_minor', False)
    if block_scan and scan_minor:
      raise ValueError('Token-minor carry requires direct layer scan')
    if block_scan and cfg.num_decoder_layers % 3:
      raise ValueError('RMT block scan requires a multiple of three layers')
    policy_name=cfg.get_keys().get('rmt_remat_policy','full')
    if policy_name not in ('full','save_dense','save_dense_state','save_state','save_state_mlp','save_state_dynamic','attention_only'):raise ValueError(f'Unknown RMT remat policy: {policy_name}')
    policy=(jax.checkpoint_policies.dots_with_no_batch_dims_saveable if policy_name in ('save_dense','save_dense_state') else None)
    if policy_name in ('save_dense_state','save_state','save_state_mlp','save_state_dynamic'):
      names=('rmt_attention_head','rmt_mlp_matrix')
      if cfg.get_keys().get('rmt_save_middle_outputs',False):
        names+=('rmt_middle_vector','rmt_middle_proxy','rmt_middle_residual_matrix')
      if policy_name=='save_state_mlp':names+=('mlpwi_0','mlpwi_1','mlpwo')
      if policy_name=='save_state_dynamic':names+=('rmt_dynamic_projection','rmt_basis_read','rmt_compressed_read','rmt_dynamic_address')
      named=jax.checkpoint_policies.save_only_these_names(*names)
      policy=(jax.checkpoint_policies.save_from_both_policies(policy,named)
              if policy_name=='save_dense_state' else named)
    if block_scan and policy_name!='full':raise ValueError('Selective remat requires direct layer scan')
    Layer = (RMTBlock if block_scan else RMTLayer if policy_name=='attention_only' else
             nn.remat(RMTLayer,prevent_cse=True,static_argnums=(4,),policy=policy))
    scan_length = cfg.num_decoder_layers // 3 if block_scan else cfg.num_decoder_layers
    ScanLayer = nn.scan(
        Layer,
        variable_axes={'params': cfg.param_scan_axis, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': cfg.enable_dropout},
        in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, 0),
        length=scan_length,
        unroll=int(cfg.scan_layers_unroll),
        metadata_params={nn.PARTITION_NAME: 'layers'})
    if scan_minor:
      matrix = matrix.transpose(0,2,3,1)
    matrix, _ = ScanLayer(cfg, quant=self.quant, name='layers')(
        matrix, decoder_segment_ids, decoder_positions, deterministic,
        jnp.arange(scan_length))
    if scan_minor:
      matrix = matrix.transpose(0,3,1,2)
    if padded_value_dim:matrix = matrix[..., :value_dim]
    matrix = MatrixRMSNorm(cfg, name='final_matrix_norm')(matrix)
    final_read = self.param('final_read_key', nn.initializers.normal(key_dim ** -0.5),
                            (key_dim, heads), cfg.weight_dtype)
    hidden = jnp.einsum('btkv,kn->btnv', matrix, final_read.astype(cfg.dtype))
    if cfg.get_keys().get('rmt_dynamic_unembedding_read', False):
      # Final full-matrix norm stays; the first16 proxy additionally has the
      # same learned vector pre-norm as the middle-layer dynamic routes.
      x = matrix[..., :heads, :].reshape(matrix.shape[:2] + (cfg.emb_dim,))
      x = normalizations.get_rmsnorm('unembedding_vector_norm', cfg)(x)
      dynamic_state = jnp.swapaxes(matrix[..., heads:, :], -2, -1)
      (dynamic_read,), read_gates = RMTDynamicC8Read(
          cfg, destinations=1,
          compress_state=not cfg.get_keys().get('rmt_dynamic_unembedding_direct_read', False),
          name='dynamic_unembedding_read')(x, dynamic_state)
      if cfg.get_keys().get('rmt_record_dynamic_health', False):
        self.sow('intermediates', 'rmt_unembedding_health',
                 _boundary_health(dynamic_read, hidden, read_gates[..., 0]))
      hidden = hidden + dynamic_read
    hidden = hidden.reshape(hidden.shape[:2] + (cfg.emb_dim,))
    head = models.OutputHead(config=cfg, shared_embedding=self.shared_embedding,
                             mesh=self.mesh, quant=self.quant, name='lm_head')
    return head(hidden, decoder_target_tokens, decoder_target_mask,
                cfg.loss_chunk_size, deterministic)

  def logits_from_hidden_states(self, hidden_states, deterministic=True,
                                mtp_layer=False):
    raise ValueError('RMT matrix state must be read before producing logits')

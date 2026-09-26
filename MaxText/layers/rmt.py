"""Residual Matrix Transformer in MaxText's MediumProp training runtime.

The residual stream is [batch, time, ResKey, ResVal].  This follows the
published RMT contractions and the open-source module layout, while the MLP,
optimizer, data and loss remain the matched MaxText backbone.
"""

import math

from flax import linen as nn
import jax
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


def dynamic_health_names(record_write_health=True):
  """Keep read/gate/state metrics when diagnostic write statistics are disabled."""
  return tuple(name for name in RMT_DYNAMIC_HEALTH_NAMES
               if record_write_health or not
               ('_write_' in name and name.endswith(('_ratio', '_cosine'))))


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


def _row_reduced_write_health(dynamic, reference, *, return_reference_rms=False):
  """Compute identical partition statistics without slicing full matrix tensors."""
  dynamic, reference = dynamic.astype(jnp.float32), reference.astype(jnp.float32)
  # Leave only K rows before splitting first16/tail32. All large-array
  # products/reductions can fuse; no full B*T*K*V partition copies are needed.
  axes = (0, 1, 3)
  dynamic_ms = jnp.mean(jnp.square(dynamic), axis=axes)
  reference_ms = jnp.mean(jnp.square(reference), axis=axes)
  cross = jnp.mean(dynamic * reference, axis=axes)
  values, reference_rmss = [], []
  for part in (slice(None, 16), slice(16, None)):
    dynamic_rms = jnp.sqrt(jnp.mean(dynamic_ms[part]))
    reference_rms = jnp.sqrt(jnp.mean(reference_ms[part]))
    reference_rmss.append(reference_rms)
    values.extend((dynamic_rms / jnp.maximum(reference_rms, 1e-12),
                   jnp.mean(cross[part]) /
                   jnp.maximum(dynamic_rms * reference_rms, 1e-12)))
  return ((tuple(values), tuple(reference_rmss)) if return_reference_rms
          else tuple(values))


def _dynamic_outer_write(address, data, method):
  """Equivalent write contractions for matched full-step lowering diagnostics."""
  if method == 'dot':
    return jnp.einsum('btnk,btnv->btkv', address, data)
  if method == 'dot_transposed':
    return jnp.swapaxes(jnp.einsum('btnv,btnk->btvk', data, address), -2, -1)
  if method == 'mul_reduce':
    # Match the dot's FP32 accumulator rather than reducing products in BF16.
    product = (address.astype(jnp.float32)[..., :, :, None]
               * data.astype(jnp.float32)[..., :, None, :])
    return jnp.sum(product, axis=-3).astype(data.dtype)
  raise ValueError(f'Unsupported RMT write contraction: {method}')


def _factorized_write_health(dynamic_address, static_address, data):
  """Measure two outer writes without materializing either component matrix."""
  # Small per-token head Grams lower poorly as padded batched TPU dots.
  # Fused multiply/reduce keeps FP32 statistics without materialized writes.
  data = data.astype(jnp.float32)
  data_gram = jnp.sum(data[..., :, None, :] * data[..., None, :, :], axis=-1)
  values = []
  for part in (slice(None, 16), slice(16, None)):
    dyn = dynamic_address[..., part].astype(jnp.float32)
    stat = static_address[..., part].astype(jnp.float32)
    size = dyn.shape[-1] * data.shape[-1]
    dynamic_gram = jnp.sum(dyn[..., :, None, :] * dyn[..., None, :, :], axis=-1)
    static_gram = jnp.einsum('nk,mk->nm', stat, stat)
    cross_gram = jnp.sum(stat[:, None, :] * dyn[..., None, :, :], axis=-1)
    dynamic_ms = jnp.maximum(jnp.mean(jnp.sum(dynamic_gram * data_gram, axis=(-2, -1))) / size, 0.)
    static_ms = jnp.maximum(jnp.mean(jnp.sum(static_gram * data_gram, axis=(-2, -1))) / size, 0.)
    cross = jnp.mean(jnp.sum(cross_gram * data_gram, axis=(-2, -1))) / size
    dynamic_rms, static_rms = jnp.sqrt(dynamic_ms), jnp.sqrt(static_ms)
    values.extend((dynamic_rms / jnp.maximum(static_rms, 1e-12),
                   cross / jnp.maximum(dynamic_rms * static_rms, 1e-12)))
  return tuple(values)


def _gate_health(gate):
  gate = gate.astype(jnp.float32)
  return jnp.mean(gate), jnp.mean(gate > 0.5)


def _init_gate_bias(opening):
  logit = math.log(opening / (1.0 - opening))
  return nn.initializers.constant(logit)


def _read_epsilon(cfg):
  return (cfg.normalization_layer_epsilon if cfg.bam_read_key_epsilon is None
          else cfg.bam_read_key_epsilon)


class RMTDynamicQK(nn.Module):
  """BAM column-only shared rank-4 basis with independently gated Q/K routing."""

  config: common_types.Config

  @nn.compact
  def __call__(self, x, M):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    address_dim = M.shape[-1]
    rank = 4
    init = initializers.get_init_method(cfg.init_method)
    basis_kernel = self.param('basis_kernel', nn.with_logical_partitioning(init, ('embed', None)),
                              (cfg.emb_dim, rank * address_dim), cfg.weight_dtype)
    basis_bias = self.param('basis_bias', nn.with_logical_partitioning(nn.initializers.zeros, (None, 'kv')),
                            (rank, address_dim), cfg.weight_dtype)
    basis = jnp.einsum('btd,dr->btr', x, basis_kernel.astype(x.dtype))
    basis = basis.reshape(x.shape[:2] + (rank, address_dim))
    basis = basis + basis_bias.astype(x.dtype)
    basis_read = jnp.einsum('btvc,btrc->btrv', M, basis)
    basis_fp32 = basis.astype(jnp.float32)
    gram = jnp.einsum('btrc,btsc->btrs', basis_fp32, basis_fp32)
    results, gates = [], []
    for arm in ('q', 'k'):
      mix_kernel = self.param(f'{arm}_mix_kernel', nn.with_logical_partitioning(init, ('embed', None)),
                              (cfg.emb_dim, heads * rank), cfg.weight_dtype)
      mix = jnp.einsum('btd,dr->btr', x, mix_kernel.astype(x.dtype))
      mix = mix.reshape(x.shape[:2] + (heads, rank))
      gate_kernel = self.param(f'{arm}_gate_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                               (cfg.emb_dim, heads), cfg.weight_dtype)
      gate_bias = self.param(f'{arm}_gate_bias', nn.with_logical_partitioning(_init_gate_bias(.05), ('q_heads',)),
                             (heads,), cfg.weight_dtype)
      gate = jax.nn.sigmoid(
          jnp.einsum('btd,dn->btn', x, gate_kernel.astype(x.dtype))
          + gate_bias.astype(x.dtype))
      # Same effective-key RMS as BAM's rank-4 Gram route, after pre-RMS bias.
      mix_fp32 = mix.astype(jnp.float32)
      norm2 = jnp.einsum('btnr,btrs,btns->btn', mix_fp32, gram, mix_fp32)
      inverse_rms = jax.lax.rsqrt(norm2 / address_dim + _read_epsilon(cfg))
      read = jnp.einsum('btrv,btnr->btnv', basis_read, mix)
      results.append(read * (inverse_rms.astype(read.dtype) * (.2 * gate))[..., None])
      gates.append(gate)
    return results[0], results[1], gates[0], gates[1]


class RMTDynamicC8Read(nn.Module):
  """One C8 key/read shared by destinations, with separate per-head gates."""

  config: common_types.Config
  destinations: int
  fetch_output: bool = False
  independent_output_key: bool = False

  @nn.compact
  def __call__(self, x, M):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    compression = self.param('compression', nn.with_logical_partitioning(nn.initializers.orthogonal(), ('v_factor', 'kv')),
                             (M.shape[-1], 8), cfg.weight_dtype)
    compressed = jnp.einsum('btvc,cr->btvr', M, compression.astype(M.dtype))
    key_kernel = self.param('key_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                            (cfg.emb_dim, heads * 8), cfg.weight_dtype)
    raw_key = jnp.einsum('btd,dr->btr', x, key_kernel.astype(x.dtype))
    raw_key = raw_key.reshape(x.shape[:2] + (heads, 8))
    key = normalizations.rms_norm(
        raw_key, dtype=x.dtype, epsilon=_read_epsilon(cfg),
        statistics_dtype=jnp.float32)
    read = jnp.einsum('btvc,btnc->btnv', compressed, key)
    gate_kernel = self.param('gate_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
                             (cfg.emb_dim, heads * self.destinations), cfg.weight_dtype)
    gate_bias = self.param('gate_bias', nn.with_logical_partitioning(_init_gate_bias(.05), ('q_heads', None)),
                           (heads, self.destinations), cfg.weight_dtype)
    logits = jnp.einsum('btd,dr->btr', x, gate_kernel.astype(x.dtype))
    logits = logits.reshape(x.shape[:2] + (heads, self.destinations))
    gates = jax.nn.sigmoid(logits + gate_bias.astype(x.dtype))
    if self.fetch_output:
      if self.destinations != 2:
        raise ValueError('Fetched O requires separate V/O gates')
      output_key = key
      if self.independent_output_key:
        output_kernel = self.param(
            'o_key_kernel', nn.with_logical_partitioning(nn.initializers.zeros, ('embed', None)),
            (cfg.emb_dim, heads * 8), cfg.weight_dtype)
        output_key = jnp.einsum('btd,dr->btr', x, output_kernel.astype(x.dtype))
        output_key = normalizations.rms_norm(
            output_key.reshape(x.shape[:2] + (heads, 8)), dtype=x.dtype,
            epsilon=_read_epsilon(cfg), statistics_dtype=jnp.float32)
      # O reads the fetched state later. Do not compute an unused local O read.
      return (.2 * gates[..., 0, None] * read, gates, compressed, output_key)
    return tuple(.2 * gates[..., i, None] * read for i in range(self.destinations)), gates


class RMTDynamicWrite(nn.Module):
  """BAM GELU-LoRA address and normalized outer write into 32 or 48 RMT rows."""

  config: common_types.Config
  address_dim: int

  @nn.compact
  def __call__(self, x, data, static_address=None):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    bottleneck = 256
    init = initializers.get_init_method(cfg.init_method)
    down = self.param('address_down', nn.with_logical_partitioning(init, ('embed', None)),
                      (cfg.emb_dim, bottleneck), cfg.weight_dtype)
    up = self.param('address_up', nn.with_logical_partitioning(init, ('embed', None)),
                    (bottleneck, heads * self.address_dim), cfg.weight_dtype)
    up_bias = self.param('address_up_bias', nn.with_logical_partitioning(nn.initializers.zeros, ('q_heads', None)),
                         (heads, self.address_dim), cfg.weight_dtype)
    hidden = nn.gelu(jnp.einsum('btd,dr->btr', x, down.astype(x.dtype)))
    address = jnp.einsum('btr,rd->btd', hidden, up.astype(x.dtype))
    address = address.reshape(x.shape[:2] + (heads, self.address_dim))
    address = address + up_bias.astype(x.dtype)
    gate_kernel = self.param('gate_kernel', nn.with_logical_partitioning(init, ('embed', None)),
                             (cfg.emb_dim, heads), cfg.weight_dtype)
    gate_bias = self.param('gate_bias', nn.with_logical_partitioning(_init_gate_bias(.1), ('q_heads',)),
                           (heads,), cfg.weight_dtype)
    gate = jax.nn.sigmoid(
        jnp.einsum('btd,dn->btn', x, gate_kernel.astype(x.dtype))
        + gate_bias.astype(x.dtype))
    address = normalizations.rms_norm(
        address, dtype=address.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    if static_address is not None:
      inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(data.astype(jnp.float32)),
                                         axis=-1, keepdims=True)
                                 + cfg.normalization_layer_epsilon)
      dynamic_address = ((gate[..., None] * address).astype(jnp.float32)
                         * inverse_rms).astype(data.dtype)
      if dynamic_address.shape[-1] != static_address.shape[-1]:
        dynamic_address = jnp.pad(dynamic_address, ((0, 0), (0, 0), (0, 0), (16, 0)))
      combined_address = static_address.astype(data.dtype) + dynamic_address
      write = _dynamic_outer_write(combined_address, data,
                                   cfg.get_keys().get('rmt_write_contraction', 'dot'))
      health = (_factorized_write_health(dynamic_address, static_address.astype(data.dtype), data)
                if (cfg.get_keys().get('rmt_record_dynamic_health', False)
                    and cfg.get_keys().get('rmt_record_write_health', True)) else ())
      return write, gate, health
    data = normalizations.rms_norm(
        data, dtype=data.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    write = _dynamic_outer_write(gate[..., None] * address, data,
                                 cfg.get_keys().get('rmt_write_contraction', 'dot'))
    return write, gate


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
  is_fetch: bool = False
  mlp_dim: int | None = None

  @nn.compact
  def __call__(self, matrix, segment_ids, positions, deterministic, layer_index):
    cfg = self.config
    transposed_carry = cfg.get_keys().get('rmt_transposed_matrix_carry', False)
    if transposed_carry:
      matrix = jnp.swapaxes(matrix, -2, -1)
    heads = int(cfg.num_query_heads)
    value_dim = int(cfg.head_dim)
    key_dim = int(cfg.rmt_reskey_dim)
    assert cfg.emb_dim == heads * value_dim
    dynamic = bool(getattr(cfg, 'rmt_dynamic_enabled', False))
    dynamic_mlp = dynamic and bool(cfg.get_keys().get('rmt_dynamic_mlp_enabled', True))
    dynamic_mlp_read_enabled = dynamic and bool(
        cfg.get_keys().get('rmt_dynamic_mlp_read_enabled', dynamic_mlp))
    dynamic_mlp_write_enabled = dynamic and bool(
        cfg.get_keys().get('rmt_dynamic_mlp_write_enabled', dynamic_mlp))
    single_outer_write = bool(cfg.get_keys().get('rmt_single_outer_write', False))
    static_write_enabled = bool(cfg.get_keys().get('rmt_static_write_enabled', True))
    if single_outer_write and not (dynamic and dynamic_mlp_write_enabled):
      raise ValueError('Single-outer write requires dynamic attention and MLP writes')
    if not static_write_enabled and (single_outer_write or not dynamic or not dynamic_mlp_write_enabled):
      raise ValueError('Dynamic-only writes require dynamic attention/MLP and no combined static write')
    dynamic_o_enabled = bool(cfg.get_keys().get('rmt_dynamic_o_enabled', True))
    dynamic_full_read = bool(cfg.get_keys().get('rmt_dynamic_read_full_matrix', False))
    rope_qk_dim = int(cfg.get_keys().get('rmt_rope_qk_dim', 0))
    vector_pre_norm = bool(cfg.get_keys().get('rmt_vector_pre_norm', False))
    if vector_pre_norm and not dynamic:
      raise ValueError('RMT vector pre-norm requires the dynamic arm')
    if rope_qk_dim and (not dynamic or rope_qk_dim % 2 or rope_qk_dim >= value_dim):
      raise ValueError('RMT RoPE Q/K requires a dynamic arm and an even proper subspace')
    write_rows = int(getattr(cfg, 'rmt_dynamic_write_rows', 32)) if dynamic else 0
    if dynamic and (key_dim != 48 or write_rows not in (32, 48)):
      raise ValueError('RMT K48 dynamic branch requires 32 or 48 write rows')
    key_init = nn.initializers.normal(key_dim ** -0.5)
    write_init = nn.initializers.normal(heads ** -0.5 / math.sqrt(2 * cfg.num_decoder_layers))

    attn_in = (matrix if vector_pre_norm else
               MatrixRMSNorm(cfg, name='attn_norm')(matrix))
    qkv_key = self.param('qkv_key', key_init, (3, heads, key_dim), cfg.weight_dtype)
    qkv = jnp.einsum('btkv,ank->abtnv', attn_in, qkv_key.astype(cfg.dtype))
    query, key, value = qkv[0], qkv[1], qkv[2]
    if dynamic:
      attn_x = attn_in[..., :heads, :].reshape(attn_in.shape[:2] + (cfg.emb_dim,))
      if vector_pre_norm:
        attn_x = normalizations.get_rmsnorm('attn_vector_norm', cfg)(attn_x)
      read_start = 0 if dynamic_full_read else heads
      attn_M = jnp.swapaxes(attn_in[..., read_start:, :], -2, -1)
      dynamic_q, dynamic_k, q_gate, k_gate = RMTDynamicQK(
          cfg, name='dynamic_qk')(attn_x, attn_M)
      if self.is_fetch:
        if not dynamic_o_enabled:
          raise ValueError('LLF fetch requires dynamic O')
        dynamic_v, vo_gates, fetch_state, output_key = RMTDynamicC8Read(
            cfg, destinations=2, fetch_output=True,
            independent_output_key=cfg.get_keys().get('rmt_fetch_independent_o_key', False),
            name='dynamic_vo')(attn_x, attn_M)
        mix_kernel = self.param(
            'fetch_head_mix_kernel', nn.with_logical_partitioning(
                initializers.get_init_method(cfg.init_method), ('embed', 'q_heads')),
            (cfg.emb_dim, heads), cfg.weight_dtype)
        mix_bias = self.param('fetch_head_mix_bias', nn.with_logical_partitioning(
            nn.initializers.zeros, ('q_heads',)), (heads,), cfg.weight_dtype)
        mix_logits = jnp.einsum('btd,dn->btn', attn_x, mix_kernel.astype(attn_x.dtype))
        mix_logits = mix_logits + mix_bias.astype(attn_x.dtype)
        mix_weights = attentions._dynamic_bam_fetch_mix_weights(
            mix_logits, cfg.dtype, rms_epsilon=_read_epsilon(cfg))
      else:
        vo_reads, vo_gates = RMTDynamicC8Read(
            cfg, destinations=2 if dynamic_o_enabled else 1,
            name='dynamic_vo')(attn_x, attn_M)
        dynamic_v = vo_reads[0]
        if dynamic_o_enabled:
          dynamic_o = vo_reads[1]
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
    query = query / math.sqrt(value_dim)
    t = matrix.shape[1]
    chunk = int(cfg.query_chunk_size)
    assert t % chunk == 0
    outputs = []
    fetched_outputs = []
    fetch_route_sums = []
    record_health = dynamic and bool(getattr(cfg, 'rmt_record_dynamic_health', False))
    for q0 in range(0, t, chunk):
      q1 = q0 + chunk
      source = jnp.arange(q1)[None, :]
      target = jnp.arange(q0, q1)[:, None]
      valid = (source <= target)[None]
      if segment_ids is not None:
        valid &= (segment_ids[:, q0:q1, None] == segment_ids[:, None, :q1])
      y, alpha = attentions._attention_op(
          query[:, q0:q1], key[:, :q1], value[:, :q1], valid,
          float32_logits=cfg.float32_logits if rope_qk_dim else True,
          additive_bias=(None if rope_qk_dim else
                         attentions._alibi_bias(heads, q0, q1, 0, q1)))
      outputs.append(y)
      if self.is_fetch:
        fetched = attentions._bam_fetch_op(
            alpha, fetch_state[:, :q1], mix_weights[:, q0:q1], source == target,
            diagonal_one=True, return_route=record_health)
        if record_health:
          fetched, raw_route, route = fetched
          health_valid = jnp.broadcast_to(valid, route.shape)
          if segment_ids is not None:
            health_valid &= segment_ids[:, q0:q1, None] != 0
          fetch_route_sums.append(attentions._bam_fetch_route_sums(
              raw_route, route, health_valid, source == target))
        fetched_read = jnp.einsum(
            'btvc,btnc->btnv', fetched, output_key[:, q0:q1])
        fetched_outputs.append(.2 * vo_gates[:, q0:q1, :, 1, None] * fetched_read)
    if self.is_fetch:
      dynamic_o = jnp.concatenate(fetched_outputs, axis=1).astype(cfg.dtype)
      if record_health:
        self.sow('intermediates', 'rmt_fetch_route_sums',
                 jnp.sum(jnp.stack(fetch_route_sums), axis=0))
    head_output = jnp.concatenate(outputs, axis=1).astype(cfg.dtype)
    if dynamic:
      static_head_output = head_output
      if dynamic_o_enabled:
        head_output = head_output + dynamic_o
    attn_residual = matrix
    if static_write_enabled:
      attn_write = self.param('attn_write_key', write_init,
                              (heads, key_dim), cfg.weight_dtype)
    elif self.is_initializing():
      # Preserve the parent's later parameter seeds without retaining this key.
      self.make_rng('params')
    if single_outer_write:
      combined_attn_write, attn_write_gate, attn_write_health = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_attn_write')(attn_x, head_output, attn_write)
      matrix = matrix + combined_attn_write
    elif static_write_enabled:
      static_attn_write = jnp.einsum(
          'btnv,nk->btkv', head_output, attn_write.astype(cfg.dtype))
    if dynamic and not single_outer_write:
      dynamic_attn_write, attn_write_gate = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_attn_write')(attn_x, head_output)
      if write_rows == 32:
        dynamic_attn_write = jnp.pad(dynamic_attn_write,
                                     ((0, 0), (0, 0), (16, 0), (0, 0)))
      if static_write_enabled:
        matrix = matrix + static_attn_write + dynamic_attn_write
      else:
        matrix = matrix + dynamic_attn_write
    elif not dynamic:
      matrix = matrix + static_attn_write

    mlp_in = (matrix if vector_pre_norm else
              MatrixRMSNorm(cfg, name='mlp_norm')(matrix))
    mlp_read = self.param('mlp_read_key', key_init,
                          (key_dim, heads), cfg.weight_dtype)
    vector = jnp.einsum('btkv,kn->btnv', mlp_in, mlp_read.astype(cfg.dtype))
    static_mlp_read = vector
    if dynamic_mlp_read_enabled or dynamic_mlp_write_enabled:
      mlp_x = mlp_in[..., :heads, :].reshape(mlp_in.shape[:2] + (cfg.emb_dim,))
      if vector_pre_norm:
        mlp_x = normalizations.get_rmsnorm('mlp_vector_norm', cfg)(mlp_x)
    if dynamic_mlp_read_enabled:
      mlp_M = jnp.swapaxes(mlp_in[..., read_start:, :], -2, -1)
      (dynamic_mlp_read,), mlp_read_gate = RMTDynamicC8Read(
          cfg, destinations=1, name='dynamic_mlp_read')(mlp_x, mlp_M)
      vector = vector + dynamic_mlp_read
    headwise_mlp = cfg.get_keys().get('rmt_headwise_mlp', False)
    if not headwise_mlp:
      vector = vector.reshape(vector.shape[:2] + (cfg.emb_dim,))
    if cfg.get_keys().get('rmt_static_mlp_read_pre_norm', False):
      if dynamic_mlp_read_enabled or headwise_mlp:
        raise ValueError('Static MLP read pre-norm requires the static MLP route')
      vector = normalizations.get_rmsnorm('mlp_read_vector_norm', cfg)(vector)
    vector = linears.MlpBlock(
        config=cfg, intermediate_dim=cfg.mlp_dim if self.mlp_dim is None else self.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=initializers.get_init_method(cfg.init_method),
        quant=self.quant, headwise=headwise_mlp, name='mlp')(vector, deterministic=deterministic)
    if not headwise_mlp:
      vector = vector.reshape(vector.shape[:2] + (heads, value_dim))
    mlp_residual = matrix
    if static_write_enabled:
      mlp_write = self.param('mlp_write_key', write_init,
                             (heads, key_dim), cfg.weight_dtype)
    elif self.is_initializing():
      self.make_rng('params')
    if single_outer_write:
      combined_mlp_write, mlp_write_gate, mlp_write_health = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_mlp_write')(mlp_x, vector, mlp_write)
      matrix = matrix + combined_mlp_write
    elif static_write_enabled:
      static_mlp_write = jnp.einsum(
          'btnv,nk->btkv', vector, mlp_write.astype(cfg.dtype))
    if dynamic_mlp_write_enabled and not single_outer_write:
      dynamic_mlp_write, mlp_write_gate = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_mlp_write')(mlp_x, vector)
      if write_rows == 32:
        dynamic_mlp_write = jnp.pad(dynamic_mlp_write,
                                    ((0, 0), (0, 0), (16, 0), (0, 0)))
      if static_write_enabled:
        matrix = matrix + static_mlp_write + dynamic_mlp_write
      else:
        matrix = matrix + dynamic_mlp_write
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
      input_health = None
      reuse_input_rms = cfg.get_keys().get('rmt_write_health_reuse_input_rms', False)
      if reuse_input_rms and not (record_write_health and vector_pre_norm
                                  and not static_write_enabled and not single_outer_write
                                  and cfg.get_keys().get('rmt_write_health_row_reduce', False)):
        raise ValueError('Write/reference RMS reuse requires pure VectorNorm row health')
      if record_write_health:
        if single_outer_write:
          values.extend(attn_write_health + mlp_write_health)
        else:
          # Without static writes, report amplitude/alignment against the residual.
          writes = ((dynamic_attn_write, static_attn_write if static_write_enabled else attn_residual),
                    (dynamic_mlp_write, static_mlp_write if static_write_enabled else mlp_residual))
          if cfg.get_keys().get('rmt_write_health_row_reduce', False):
            if reuse_input_rms:
              # Under VectorNorm with pure dynamic writes the references are
              # exactly attn_in/mlp_in. Reuse their two partition RMSs rather
              # than slicing/reducing the same large matrices a second time.
              input_health = []
              for dyn, stat in writes:
                stats, reference_rmss = _row_reduced_write_health(
                    dyn, stat, return_reference_rms=True)
                values.extend(stats)
                input_health.extend(reference_rmss)
            else:
              values.extend(v for dyn, stat in writes
                            for v in _row_reduced_write_health(dyn, stat))
          else:
            values.extend(v for dyn, stat in writes for part in
                          (slice(None, 16), slice(16, None))
                          for v in _write_health(dyn[..., part, :], stat[..., part, :]))
      values.extend(input_health if input_health is not None else
                    (_rms(attn_in[..., :16, :]), _rms(attn_in[..., 16:, :]),
                     _rms(mlp_in[..., :16, :]), _rms(mlp_in[..., 16:, :])))
      assert len(values) == len(dynamic_health_names(record_write_health))
      health = jnp.stack(values)
      if not cfg.get_keys().get('rmt_block_scan', False):
        self.sow('intermediates', 'rmt_dynamic_health', health)
    return (jnp.swapaxes(matrix, -2, -1) if transposed_carry else matrix), (
        health if cfg.get_keys().get('rmt_block_scan', False) else None)


class RMTBlock(nn.Module):
  """Three matrix-residual layers, optionally ending with a fetched-O layer."""

  config: common_types.Config
  quant: object = None

  @nn.compact
  def __call__(self, matrix, segment_ids, positions, deterministic, block_index):
    cfg = self.config
    widths = cfg.rmt_mlp_dim_by_block
    health = []
    Layer = nn.remat(RMTLayer, prevent_cse=True, static_argnums=(4,))
    for offset in range(3):
      fetch = offset == 2 and cfg.get_keys().get('rmt_llf_enabled', False)
      matrix, stats = Layer(
          cfg, quant=self.quant, is_fetch=fetch, mlp_dim=int(widths[offset]),
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
    transposed_carry = cfg.get_keys().get('rmt_transposed_matrix_carry', False)
    if transposed_carry:
      matrix = jnp.swapaxes(matrix, -2, -1)
    block_scan = cfg.get_keys().get('rmt_block_scan', False)
    if block_scan and cfg.num_decoder_layers % 3:
      raise ValueError('RMT block scan requires a multiple of three layers')
    Layer = RMTBlock if block_scan else nn.remat(RMTLayer, prevent_cse=True, static_argnums=(4,))
    scan_length = cfg.num_decoder_layers // 3 if block_scan else cfg.num_decoder_layers
    ScanLayer = nn.scan(
        Layer,
        variable_axes={'params': cfg.param_scan_axis, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': cfg.enable_dropout},
        in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, 0),
        length=scan_length,
        unroll=int(cfg.scan_layers_unroll),
        metadata_params={nn.PARTITION_NAME: 'layers'})
    matrix, _ = ScanLayer(cfg, quant=self.quant, name='layers')(
        matrix, decoder_segment_ids, decoder_positions, deterministic,
        jnp.arange(scan_length))
    if transposed_carry:
      matrix = jnp.swapaxes(matrix, -2, -1)
    matrix = MatrixRMSNorm(cfg, name='final_matrix_norm')(matrix)
    final_read = self.param('final_read_key', nn.initializers.normal(key_dim ** -0.5),
                            (key_dim, heads), cfg.weight_dtype)
    hidden = jnp.einsum('btkv,kn->btnv', matrix, final_read.astype(cfg.dtype))
    hidden = hidden.reshape(hidden.shape[:2] + (cfg.emb_dim,))
    head = models.OutputHead(config=cfg, shared_embedding=self.shared_embedding,
                             mesh=self.mesh, quant=self.quant, name='lm_head')
    return head(hidden, decoder_target_tokens, decoder_target_mask,
                cfg.loss_chunk_size, deterministic)

  def logits_from_hidden_states(self, hidden_states, deterministic=True,
                                mtp_layer=False):
    raise ValueError('RMT matrix state must be read before producing logits')

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
from layers import attentions, initializers, linears, normalizations


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
    return tuple(.2 * gates[..., i, None] * read for i in range(self.destinations)), gates


class RMTDynamicWrite(nn.Module):
  """BAM GELU-LoRA address and normalized outer write into 32 or 48 RMT rows."""

  config: common_types.Config
  address_dim: int

  @nn.compact
  def __call__(self, x, data):
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
    data = normalizations.rms_norm(
        data, dtype=data.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    address = normalizations.rms_norm(
        address, dtype=address.dtype, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=jnp.float32)
    write = jnp.einsum('btnk,btnv->btkv', gate[..., None] * address, data)
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

  @nn.compact
  def __call__(self, matrix, segment_ids, deterministic, layer_index):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    value_dim = int(cfg.head_dim)
    key_dim = int(cfg.rmt_reskey_dim)
    assert cfg.emb_dim == heads * value_dim
    dynamic = bool(getattr(cfg, 'rmt_dynamic_enabled', False))
    dynamic_o_enabled = bool(cfg.get_keys().get('rmt_dynamic_o_enabled', True))
    write_rows = int(getattr(cfg, 'rmt_dynamic_write_rows', 32)) if dynamic else 0
    if dynamic and (key_dim != 48 or write_rows not in (32, 48)):
      raise ValueError('RMT K48 dynamic branch requires 32 or 48 write rows')
    key_init = nn.initializers.normal(key_dim ** -0.5)
    write_init = nn.initializers.normal(heads ** -0.5 / math.sqrt(2 * cfg.num_decoder_layers))

    attn_in = MatrixRMSNorm(cfg, name='attn_norm')(matrix)
    qkv_key = self.param('qkv_key', key_init, (3, heads, key_dim), cfg.weight_dtype)
    qkv = jnp.einsum('btkv,ank->abtnv', attn_in, qkv_key.astype(cfg.dtype))
    query, key, value = qkv[0], qkv[1], qkv[2]
    if dynamic:
      attn_x = attn_in[..., :heads, :].reshape(attn_in.shape[:2] + (cfg.emb_dim,))
      attn_M = jnp.swapaxes(attn_in[..., heads:, :], -2, -1)
      dynamic_q, dynamic_k, q_gate, k_gate = RMTDynamicQK(
          cfg, name='dynamic_qk')(attn_x, attn_M)
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
    query = query / math.sqrt(value_dim)
    t = matrix.shape[1]
    chunk = int(cfg.query_chunk_size)
    assert t % chunk == 0
    outputs = []
    for q0 in range(0, t, chunk):
      q1 = q0 + chunk
      source = jnp.arange(q1)[None, :]
      target = jnp.arange(q0, q1)[:, None]
      valid = (source <= target)[None]
      if segment_ids is not None:
        valid &= (segment_ids[:, q0:q1, None] == segment_ids[:, None, :q1])
      y, _ = attentions._attention_op(
          query[:, q0:q1], key[:, :q1], value[:, :q1], valid,
          float32_logits=True,
          additive_bias=attentions._alibi_bias(heads, q0, q1, 0, q1))
      outputs.append(y)
    head_output = jnp.concatenate(outputs, axis=1).astype(cfg.dtype)
    if dynamic:
      static_head_output = head_output
      if dynamic_o_enabled:
        head_output = head_output + dynamic_o
    attn_write = self.param('attn_write_key', write_init,
                            (heads, key_dim), cfg.weight_dtype)
    static_attn_write = jnp.einsum(
        'btnv,nk->btkv', head_output, attn_write.astype(cfg.dtype))
    if dynamic:
      dynamic_attn_write, attn_write_gate = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_attn_write')(attn_x, head_output)
      if write_rows == 32:
        dynamic_attn_write = jnp.pad(dynamic_attn_write,
                                     ((0, 0), (0, 0), (16, 0), (0, 0)))
      matrix = matrix + static_attn_write + dynamic_attn_write
    else:
      matrix = matrix + static_attn_write

    mlp_in = MatrixRMSNorm(cfg, name='mlp_norm')(matrix)
    mlp_read = self.param('mlp_read_key', key_init,
                          (key_dim, heads), cfg.weight_dtype)
    vector = jnp.einsum('btkv,kn->btnv', mlp_in, mlp_read.astype(cfg.dtype))
    if dynamic:
      static_mlp_read = vector
      mlp_x = mlp_in[..., :heads, :].reshape(mlp_in.shape[:2] + (cfg.emb_dim,))
      mlp_M = jnp.swapaxes(mlp_in[..., heads:, :], -2, -1)
      (dynamic_mlp_read,), mlp_read_gate = RMTDynamicC8Read(
          cfg, destinations=1, name='dynamic_mlp_read')(mlp_x, mlp_M)
      vector = vector + dynamic_mlp_read
    vector = vector.reshape(vector.shape[:2] + (cfg.emb_dim,))
    vector = linears.MlpBlock(
        config=cfg, intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=initializers.get_init_method(cfg.init_method),
        quant=self.quant, name='mlp')(vector, deterministic=deterministic)
    vector = vector.reshape(vector.shape[:2] + (heads, value_dim))
    mlp_write = self.param('mlp_write_key', write_init,
                           (heads, key_dim), cfg.weight_dtype)
    static_mlp_write = jnp.einsum(
        'btnv,nk->btkv', vector, mlp_write.astype(cfg.dtype))
    if dynamic:
      dynamic_mlp_write, mlp_write_gate = RMTDynamicWrite(
          cfg, write_rows, name='dynamic_mlp_write')(mlp_x, vector)
      if write_rows == 32:
        dynamic_mlp_write = jnp.pad(dynamic_mlp_write,
                                    ((0, 0), (0, 0), (16, 0), (0, 0)))
      matrix = matrix + static_mlp_write + dynamic_mlp_write
      if getattr(cfg, 'rmt_record_dynamic_health', False):
        # Keep the health schema identical for matched Full48/NoO comparisons.
        measured_o = dynamic_o if dynamic_o_enabled else jnp.zeros_like(static_head_output)
        measured_o_gate = (vo_gates[..., 1] if dynamic_o_enabled
                           else jnp.zeros_like(vo_gates[..., 0]))
        reads = ((dynamic_q, static_q), (dynamic_k, static_k),
                 (dynamic_v, static_v), (measured_o, static_head_output),
                 (dynamic_mlp_read, static_mlp_read))
        gates = (q_gate, k_gate, vo_gates[..., 0], measured_o_gate,
                 mlp_read_gate[..., 0], attn_write_gate, mlp_write_gate)
        writes = ((dynamic_attn_write, static_attn_write),
                  (dynamic_mlp_write, static_mlp_write))
        values = [v for pair in reads for v in _read_health(*pair)]
        values.extend(v for gate in gates for v in _gate_health(gate))
        values.extend(v for dyn, stat in writes for part in
                      (slice(None, 16), slice(16, None))
                      for v in _write_health(dyn[..., part, :], stat[..., part, :]))
        values.extend((_rms(attn_in[..., :16, :]), _rms(attn_in[..., 16:, :]),
                       _rms(mlp_in[..., :16, :]), _rms(mlp_in[..., 16:, :])))
        assert len(values) == len(RMT_DYNAMIC_HEALTH_NAMES)
        self.sow('intermediates', 'rmt_dynamic_health', jnp.stack(values))
    else:
      matrix = matrix + static_mlp_write
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
    del decoder_positions
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
    Layer = nn.remat(RMTLayer, prevent_cse=True, static_argnums=(3,))
    ScanLayer = nn.scan(
        Layer,
        variable_axes={'params': cfg.param_scan_axis, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': cfg.enable_dropout},
        in_axes=(nn.broadcast, nn.broadcast, 0),
        length=cfg.num_decoder_layers,
        unroll=int(cfg.scan_layers_unroll),
        metadata_params={nn.PARTITION_NAME: 'layers'})
    matrix, _ = ScanLayer(cfg, quant=self.quant, name='layers')(
        matrix, decoder_segment_ids, deterministic,
        jnp.arange(cfg.num_decoder_layers))
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

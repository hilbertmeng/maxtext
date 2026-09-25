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
from layers import attentions, initializers, linears


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
  """One static-key RMT attention and static-adapter SwiGLU FFN layer."""

  config: common_types.Config
  quant: object = None

  @nn.compact
  def __call__(self, matrix, segment_ids, deterministic, layer_index):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    value_dim = int(cfg.head_dim)
    key_dim = int(cfg.rmt_reskey_dim)
    assert cfg.emb_dim == heads * value_dim
    key_init = nn.initializers.normal(key_dim ** -0.5)
    write_init = nn.initializers.normal(heads ** -0.5 / math.sqrt(2 * cfg.num_decoder_layers))

    attn_in = MatrixRMSNorm(cfg, name='attn_norm')(matrix)
    qkv_key = self.param('qkv_key', key_init, (3, heads, key_dim), cfg.weight_dtype)
    qkv = jnp.einsum('btkv,ank->abtnv', attn_in, qkv_key.astype(cfg.dtype))
    query, key, value = qkv[0], qkv[1], qkv[2]
    query = query / math.sqrt(value_dim)
    t = matrix.shape[1]
    chunk = int(cfg.query_chunk_size)
    assert t % chunk == 0
    outputs = []
    for q0 in range(0, t, chunk):
      q1 = q0 + chunk
      source = jnp.arange(t)[None, :]
      target = jnp.arange(q0, q1)[:, None]
      valid = (source <= target)[None]
      if segment_ids is not None:
        valid &= (segment_ids[:, q0:q1, None] == segment_ids[:, None, :])
      y, _ = attentions._attention_op(
          query[:, q0:q1], key, value, valid, float32_logits=True,
          additive_bias=attentions._alibi_bias(heads, q0, q1, 0, t))
      outputs.append(y)
    head_output = jnp.concatenate(outputs, axis=1).astype(cfg.dtype)
    attn_write = self.param('attn_write_key', write_init,
                            (heads, key_dim), cfg.weight_dtype)
    matrix = matrix + jnp.einsum(
        'btnv,nk->btkv', head_output, attn_write.astype(cfg.dtype))

    mlp_in = MatrixRMSNorm(cfg, name='mlp_norm')(matrix)
    mlp_read = self.param('mlp_read_key', key_init,
                          (key_dim, heads), cfg.weight_dtype)
    vector = jnp.einsum('btkv,kn->btnv', mlp_in, mlp_read.astype(cfg.dtype))
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
    matrix = matrix + jnp.einsum(
        'btnv,nk->btkv', vector, mlp_write.astype(cfg.dtype))
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

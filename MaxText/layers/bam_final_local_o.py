"""Generalized local-O exit from the final, unshifted BAM matrix carry."""

import math
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from layers import attentions, initializers, linears, normalizations


class FinalLocalO(nn.Module):
  """One extra local-O read and its own output projection before ln_f.

  The ordinary layers keep their fused Pallas paths. This single column read
  uses the existing native primitive and bridges v8t's [V,K] matrix layout.
  It consumes the final carry itself, without another temporal M-shift.
  """

  config: Any
  quant: Any = None

  @nn.compact
  def __call__(self, residual, matrix):
    cfg = self.config
    n, k, v = cfg.num_query_heads, cfg.bam_k, cfg.bam_v
    if cfg.emb_dim != n * k or cfg.head_dim != k:
      raise ValueError('Final localO requires D = heads * K = heads * head_dim')
    if cfg.bam_pallas_core:
      if cfg.bam_pallas_body != 'v8t':
        raise ValueError('Final localO supports the v8t Pallas carry layout')
      matrix = jnp.swapaxes(matrix, -1, -2)
    if matrix.shape != residual.shape[:-1] + (k, v):
      raise ValueError('Final localO received an incompatible matrix carry')
    bam_init = initializers.get_bam_init(cfg)
    zero = initializers.contant_dense_init(0.)

    def parameter(name, shape, axes, value=0.):
      return self.param(name, nn.with_logical_partitioning(
          nn.initializers.constant(value), axes), shape, cfg.weight_dtype)

    def dense(name, x, features, axes, init=bam_init, axis=-1):
      return linears.DenseGeneral(
          features=features, axis=axis, use_bias=False, kernel_init=init,
          kernel_axes=axes, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
          quant=self.quant, matmul_precision=cfg.matmul_precision, name=name)(x)

    x = normalizations.get_decoder_norm('input_norm', cfg)(residual)
    hidden = nn.gelu(dense('read_down', x,
                          cfg.bam_local_full_m_read_bottleneck_dim, ('embed', None)))
    raw_key = dense('read_up', hidden, (n, v), ('embed', 'q_heads', 'kv'))
    raw_key += parameter('pre_rms_bias', (n, v), ('q_heads', 'kv')).astype(cfg.dtype)
    epsilon = (cfg.bam_fetched_read_key_epsilon or cfg.bam_read_key_epsilon
               or cfg.normalization_layer_epsilon)
    opening = cfg.bam_fetched_read_gate_init or cfg.bam_read_gate_init
    if opening is None:
      opening = math.sqrt(epsilon) / cfg.bam_read_key_scale
    logits = dense('dynamic_gate', x, (n, 1), ('embed', 'q_heads', None), zero)
    logits += parameter('dynamic_gate_bias', (n, 1), ('q_heads', None),
                        math.log(opening / (1. - opening))).astype(cfg.dtype)
    arm = attentions._BamReadArm(
        name='f', k_dim=k, v_dim=v, num_heads=n, read_side='col', prune_row=True,
        key_scale=cfg.bam_read_key_scale, rms_epsilon=epsilon,
        rms_statistics_dtype={'float32': jnp.float32, 'activation': cfg.dtype}[
            cfg.bam_read_rms_statistics_dtype],
        implementation=cfg.bam_read_implementation)
    dynamic, _ = attentions.bam_read(matrix, raw_key, arm, gate_logits=logits)
    static_key = parameter('static_o_key', (v, n), ('v_factor', 'q_heads')).astype(cfg.dtype)
    static = jnp.einsum('btkv,vn->btnk', matrix, static_key)
    static_opening = cfg.bam_static_read_gate_init
    static_logits = dense('static_gate', x, n, ('embed', 'q_heads'), zero)
    static_logits += parameter('static_gate_bias', (n,), ('q_heads',),
        math.log(static_opening / (1.1 - static_opening))).astype(cfg.dtype)
    static *= (1.1 * jax.nn.sigmoid(static_logits) / static_opening)[..., None]
    # Reuse the existing signed-scalar health exporter and no-WD rule.
    scale = parameter('common_vo_read_scale', (1,), (None,),
                      1. / math.sqrt(1. + 2. * cfg.num_decoder_layers))[0]
    read = ((dynamic + static).astype(jnp.float32) * scale).astype(cfg.dtype)
    update = dense('out', read, cfg.emb_dim, ('q_heads', 'kv', 'embed'),
                   initializers.get_model_init(cfg, residual_output=True), axis=(-2, -1))
    return residual + update

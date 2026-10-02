"""Gated full-matrix column read into BAM's final vector residual."""
import math
from flax import linen as nn
import jax
import jax.numpy as jnp
from layers import initializers, linears, normalizations

HEALTH_NAMES = ('matrix_raw_rms', 'matrix_norm_rms', 'read_pre_gate_rms',
                'read_rms', 'residual_rms', 'read_over_residual',
                'gate_mean', 'gate_std', 'gate_frac_lt_005',
                'gate_frac_gt_050', 'gate_frac_gt_095',
                'matrix_scale_mean', 'matrix_scale_rms')


class BamDynamicUnembedding(nn.Module):
  config: object
  quant: object = None

  @nn.compact
  def __call__(self, residual, matrix):
    cfg = self.config
    heads = int(cfg.num_query_heads)
    if heads * matrix.shape[-2] != residual.shape[-1]:
      raise ValueError('Direct unembedding requires heads*bam_k == emb_dim')
    scale = self.param('matrix_scale', nn.with_logical_partitioning(
        nn.initializers.ones, ('k_factor', 'v_factor')),
        matrix.shape[-2:], cfg.weight_dtype)
    raw = matrix.astype(jnp.float32)
    matrix_norm = (raw * jax.lax.rsqrt(jnp.mean(raw ** 2, axis=(-2, -1),
        keepdims=True) + cfg.normalization_layer_epsilon)
        * scale.astype(jnp.float32)).astype(cfg.dtype)
    query = normalizations.get_rmsnorm('query_norm', cfg)(residual)
    key = linears.DenseGeneral(
        features=(heads, matrix.shape[-1]), axis=-1, use_bias=False,
        kernel_init=initializers.contant_dense_init(0.0),
        kernel_axes=('embed', 'q_heads', 'v_factor'), dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype, quant=self.quant,
        matmul_precision=cfg.matmul_precision, name='read_key')(query)
    key = normalizations.rms_norm(key, dtype=cfg.dtype,
        epsilon=(cfg.normalization_layer_epsilon if cfg.bam_read_key_epsilon is None
                 else cfg.bam_read_key_epsilon), statistics_dtype=jnp.float32)
    logits = linears.DenseGeneral(
        features=(heads,), axis=-1, use_bias=False,
        kernel_init=initializers.contant_dense_init(0.0), kernel_axes=('embed', 'q_heads'),
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant,
        matmul_precision=cfg.matmul_precision, name='read_gate')(query)
    opening = float(cfg.bam_unembedding_gate_init)
    bias = self.param('gate_bias', nn.with_logical_partitioning(
        nn.initializers.constant(math.log(opening / (1.0 - opening))),
        ('q_heads',)), (heads,), cfg.weight_dtype)
    gates = jax.nn.sigmoid(logits + bias.astype(cfg.dtype))
    read = jnp.einsum('btkv,btnv->btnk', matrix_norm, key)
    # Deliberately no fixed amplitude coefficient: the gate alone controls gain.
    gated = gates[..., None] * read
    update = gated.reshape(residual.shape)
    if cfg.bam_record_concat_health:
      def rms(z):return jnp.sqrt(jnp.mean(z.astype(jnp.float32) ** 2))
      g = gates.astype(jnp.float32)
      read_rms, residual_rms = rms(update), rms(residual)
      self.sow('intermediates', 'health', jnp.stack((
          rms(matrix), rms(matrix_norm), rms(read), read_rms, residual_rms,
          read_rms / jnp.maximum(residual_rms, 1e-12), jnp.mean(g), jnp.std(g),
          jnp.mean(g < .05), jnp.mean(g > .5), jnp.mean(g > .95),
          jnp.mean(scale.astype(jnp.float32)), rms(scale))))
    return residual + update

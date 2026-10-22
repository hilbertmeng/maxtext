"""Selective three-source Mudd routes for BAM values and final readout."""

from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from layers import initializers, linears, normalizations


def tail_start(config):
  return config.num_decoder_layers - 3 * int(getattr(config, 'bam_mudd_tail_blocks', 0))


def uses_anchors(config):
  return bool(getattr(config, 'bam_mudd_tail_blocks', 0)
              or getattr(config, 'bam_mudd_final_output', False))


def is_value_layer(config, layer_index):
  count = int(getattr(config, 'bam_mudd_tail_qkvm_last_layers', 0))
  return (bool(getattr(config, 'bam_mudd_tail_blocks', 0))
          and layer_index >= tail_start(config)
          and (layer_index % 3 == 2 or (count > 0 and layer_index >= config.num_decoder_layers - count)))


def skips_mlp_write(config, layer_index):
  return (bool(getattr(config, 'bam_mudd_tail_blocks', 0))
          and getattr(config, 'bam_mudd_tail_skip_mlp_write', True)
          and layer_index >= tail_start(config) and layer_index % 3 == 1)


def restores_mlp_write(config, layer_index):
  return (bool(getattr(config, 'bam_mudd_tail_blocks', 0))
          and not getattr(config, 'bam_mudd_tail_skip_mlp_write', True)
          and layer_index >= tail_start(config) and layer_index % 3 == 1)


def skips_attention_write(config, layer_index):
  count = int(getattr(config, 'bam_mudd_tail_skip_attention_write_last_layers', 0))
  return (bool(getattr(config, 'bam_mudd_tail_blocks', 0)) and count > 0
          and layer_index >= config.num_decoder_layers - count)


def is_qk_layer(config, layer_index):
  if not (bool(getattr(config, 'bam_mudd_tail_residual_qk', False))
          and is_value_layer(config, layer_index)):
    return False
  count = getattr(config, 'bam_mudd_tail_qk_last_layers', None)
  return count is None or layer_index >= config.num_decoder_layers - count


def is_mlp_route_layer(config, layer_index):
  return bool(getattr(config, 'bam_mudd_tail_residual_mlp', False)) and is_value_layer(config, layer_index)


class ThreeSourceRouter(nn.Module):
  """Mudd's signed coefficients over [current, embedding, complete L0 output]."""

  config: Any
  mesh: Any
  quant: Any = None
  hidden_expand: int = 1

  @nn.compact
  def __call__(self, current, anchors):
    cfg = self.config
    if len(anchors) != 2 or any(x.shape != current.shape for x in anchors):
      raise ValueError('three-source Mudd requires two residual anchors of the current shape')
    # Preserve the existing Mudd rounding, including its strict next multiple.
    hidden = 3 * self.hidden_expand
    if cfg.dynamic_dense_hidden_round:
      hidden = (hidden // 64 + 1) * 64
    x = normalizations.get_rmsnorm('pre_dense_proj1_norm', cfg, scale_init=None)(current)
    x = linears.DenseGeneral(
        hidden, kernel_init=initializers.nd_dense_init(1., 'fan_in', 'normal'),
        kernel_axes=('embed', None), dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        quant=self.quant, use_bias=False, name='dynamic_dense_conn1')(x)
    x = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)(x)
    coefficients = linears.DenseGeneral(
        3, kernel_init=initializers.contant_dense_init(0.), kernel_axes=('kv', None),
        dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant,
        use_bias=False, name='dynamic_dense_conn2')(x)
    bias = self.param('mix_bias', nn.with_logical_partitioning(
        lambda key, shape, dtype: jnp.asarray([1., 0., 0.], dtype), (None,)),
        (3,), cfg.weight_dtype)
    coefficients = coefficients + jnp.asarray(bias, coefficients.dtype)
    return sum(coefficients[..., i:i + 1] * source
               for i, source in enumerate((current, *anchors)))

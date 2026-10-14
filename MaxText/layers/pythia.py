"""Pythia / GPT-NeoX parallel-residual decoder on MaxText's attention backend.

Reference: EleutherAI/pythia and transformers GPTNeoXLayer. Model dimensions,
training schedule and data remain independent of this architectural recipe.
"""
from typing import Any
from flax import linen as nn
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from layers import attentions, initializers, linears, normalizations, quantizations
from pythia_config import rotary_width


class PythiaDecoderLayer(nn.Module):
  config: Any
  mesh: Any
  quant: Any = None
  sliding_window_size: int | None = None

  @nn.compact
  def __call__(self, inputs, decoder_segment_ids, decoder_positions,
               decoder_input_tokens, deep_embedding, deterministic, model_mode,
               eos_sum=None, is_global=None, hids=None, M_in=None):
    # Match the current Decoder's scanned/unscanned call contract. Pythia has no
    # matrix stream, deep embeddings, history composition or local layer schedule.
    del deep_embedding, is_global, M_in
    cfg = self.config
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    init = initializers.get_model_init(cfg)
    out_init = initializers.get_model_init(cfg, residual_output=True)
    attn_input = normalizations.get_decoder_norm("input_layernorm", cfg)(inputs)
    mlp_input = normalizations.get_decoder_norm("post_attention_layernorm", cfg)(inputs)
    attention = attentions.Attention(
        config=cfg, mesh=self.mesh, quant=self.quant, name="self_attention",
        num_query_heads=cfg.num_query_heads, num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim, max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
        kernel_init=init, output_kernel_init=out_init, output_bias=True,
        rotary_dim=rotary_width(cfg.head_dim, cfg.pythia_rotary_fraction, cfg.pythia_rotary_dim),
        dropout_rate=cfg.dropout_rate, float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits, kv_quant=quantizations.configure_kv_quant(cfg),
        sliding_window_size=cfg.max_target_length,
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(','))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(','))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(','))),
        reshape_q=cfg.reshape_q, use_ragged_attention=cfg.use_ragged_attention,
        ragged_block_size=cfg.ragged_block_size)
    attn_output = attention(
        attn_input, attn_input, decoder_positions,
        decoder_segment_ids=decoder_segment_ids, deterministic=deterministic,
        model_mode=model_mode, decoder_input_tokens=decoder_input_tokens,
        eos_sum=eos_sum)
    mlp_output = linears.MlpBlock(
        config=cfg, intermediate_dim=cfg.mlp_dim, activations=cfg.mlp_activations,
        kernel_init=init, output_kernel_init=out_init, use_bias=True,
        intermediate_dropout_rate=cfg.dropout_rate, dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype, quant=self.quant, name="mlp")(
            mlp_input, deterministic=deterministic)
    output = inputs + attn_output + mlp_output
    output = nn.with_logical_constraint(
        output, ("activation_batch", "activation_norm_length", "activation_embed"))
    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(output))
      self.sow("intermediates", "activation_stdev", jnp.std(output))
    return (output, None) if cfg.scan_layers else (output, hids)

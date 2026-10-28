"""Native write reference, including separately gated MHA/LocalO contents."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SplitAttentionWrite(NamedTuple):
  """Token-major factors; both contents share RMS(mha+local_o) and address."""
  mha: jax.Array
  local_o: jax.Array
  mha_logits: jax.Array
  local_o_logits: jax.Array
  address: jax.Array


def rms(value, epsilon):
  value = value.astype(jnp.float32)
  return jnp.sqrt(jnp.mean(value * value, axis=-1, keepdims=True) + epsilon)


def split_content(group, epsilon):
  """Return (g_a*a+g_o*o)/RMS(a+o), plus three compact health scalars.

  Normalization precedes neither gate separately nor the gated mixture. Keep
  gradients through the original, ungated sum's shared denominator.
  """
  dtype = group.mha.dtype
  ga = jax.nn.sigmoid(group.mha_logits.astype(jnp.float32)).astype(dtype)
  go = jax.nn.sigmoid(group.local_o_logits.astype(jnp.float32)).astype(dtype)
  denominator = rms(group.mha + group.local_o, epsilon)
  a = ga[..., None].astype(jnp.float32) * group.mha.astype(jnp.float32) / denominator
  o = go[..., None].astype(jnp.float32) * group.local_o.astype(jnp.float32) / denominator
  ratio = jnp.sqrt(jnp.mean(o * o)) / jnp.maximum(jnp.sqrt(jnp.mean(a * a)), 1e-12)
  stats = jnp.stack((jnp.mean(ga.astype(jnp.float32)),
                     jnp.mean(go.astype(jnp.float32)), ratio))
  return (a + o).astype(dtype), stats


def native_write(m, groups, *, epsilon):
  """Pure JAX write for v8t M[B,T,V,K]; normal triples or SplitAttentionWrite.

  One address contraction per group; the existing MLP writer remains an ordinary
  triple. The returned optional stats describe normalized content amplitudes,
  not the norms of two separately materialized matrix updates.
  """
  dtype = m.dtype
  acc = m.astype(jnp.float32)
  stats = None
  for group in groups:
    if isinstance(group, SplitAttentionWrite):
      content, stats = split_content(group, epsilon)
      address = group.address
    else:
      content, logits, address = group
      normalized = (content.astype(jnp.float32) / rms(content, epsilon)).astype(dtype)
      content = (jax.nn.sigmoid(logits.astype(jnp.float32)).astype(dtype)[..., None]
                 * normalized).astype(dtype)
    address = (address.astype(jnp.float32) / rms(address, epsilon)).astype(dtype)
    acc = acc + jnp.einsum('btnv,btnk->btvk', address.astype(jnp.float32),
                           content.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  return acc.astype(dtype), stats


def record_health_metrics(output_metrics, intermediate_outputs, config):
  """Three scalars per live writer; no head/coordinate dumps or dead final writer."""
  decoder = intermediate_outputs['intermediates']['decoder']
  layers_tree = decoder['layers']
  block = layers_tree.get('sub_0', layers_tree)['block']
  source = block if 'dual_write_stats' in block else block['self_attention']
  values = source['dual_write_stats'][0]
  layers = config.num_decoder_layers - int(getattr(config, 'bam_final_local_layer_count', 0) or 0)
  for layer in range(layers):
    for index, name in enumerate(('mha_gate_mean', 'local_o_gate_mean', 'local_o_over_mha_content_rms')):
      output_metrics['scalar'][f'bam/dual_write/layer_{layer:03d}/{name}'] = values[layer, index]

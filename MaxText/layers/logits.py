"""Vocabulary projection precision and compact, mask-aware logit health."""

import jax
import jax.numpy as jnp


HEALTH_SAMPLE_KEY = "logits_health_samples"
HEALTH_METRIC_NAMES = ("logits/common_offset_abs_p95", "logits/centered_std_median")


@jax.custom_vjp
def bf16_dot_fp32_output(hidden, kernel):
  """Keep FP32 accumulator output, with native BF16 backward dot operands.

  Inputs must already be BF16. Respect the caller's default matmul precision;
  this changes the output dtype, not the model's global precision policy.
  """
  if hidden.dtype != jnp.bfloat16 or kernel.dtype != jnp.bfloat16:
    raise TypeError("bf16_dot_fp32_output requires BF16 hidden states and kernel")
  return jnp.matmul(hidden, kernel, preferred_element_type=jnp.float32)


def _bf16_dot_fwd(hidden, kernel):
  return bf16_dot_fp32_output(hidden, kernel), (hidden, kernel)


def _bf16_dot_bwd(residual, gradient):
  hidden, kernel = residual
  gradient = gradient.astype(jnp.bfloat16)
  dh = jnp.tensordot(gradient, kernel, axes=((-1,), (-1,)))
  token_axes = tuple(range(hidden.ndim - 1))
  dw = jnp.tensordot(gradient, hidden, axes=(token_axes, token_axes)).T
  return dh, dw


bf16_dot_fp32_output.defvjp(_bf16_dot_fwd, _bf16_dot_bwd)


def health_sample_positions(batch, sequence_length, sample_size):
  """One fixed jittered position per global stratum, independent of chunks.

  Plain evenly spaced flattened indices can all land at token zero when batch
  and sequence lengths are powers of two. Jitter also samples sequence interiors.
  """
  total = batch * sequence_length
  count = min(sample_size, total)
  positions = []
  for i in range(count):
    lo, hi = i * total // count, (i + 1) * total // count
    positions.append(lo + ((1103515245 * i + 12345) % (hi - lo)))
  return positions


def chunk_health_samples(logits, target_mask, start, sequence_length, sample_size):
  """Sample fixed stratified global token positions, independent of chunks.

  Return only (row mean, centered row std, validity), never vocabulary arrays.
  The mask excludes padding. Samples cover the logical global batch; JAX's
  partitioner handles vocabulary reductions and gathers of the small summaries.
  """
  batch, length, _ = logits.shape
  selected = health_sample_positions(batch, sequence_length, sample_size)
  selected = [i for i in selected if start <= i % sequence_length < start + length]
  if not selected:
    return jnp.zeros((0, 3), jnp.float32)
  b = jnp.asarray([i // sequence_length for i in selected], jnp.int32)
  t = jnp.asarray([i % sequence_length - start for i in selected], jnp.int32)
  sampled = jax.lax.stop_gradient(logits[b, t]).astype(jnp.float32)
  mean = jnp.mean(sampled, axis=-1)
  centered = sampled - mean[..., None]
  std = jnp.sqrt(jnp.mean(centered * centered, axis=-1))
  valid = target_mask[b, t] > 0
  return jnp.stack((mean, std, valid.astype(jnp.float32)), axis=-1)


def summarize_health(samples):
  """Compute global sample quantiles once, including all accumulated batches.

  No valid sampled tokens gives NaN (missing evidence), not a fabricated zero.
  """
  samples = jax.lax.stop_gradient(samples).astype(jnp.float32)
  means = jnp.where(samples[..., 2] > 0, jnp.abs(samples[..., 0]), jnp.nan)
  stds = jnp.where(samples[..., 2] > 0, samples[..., 1], jnp.nan)
  return jnp.stack((jnp.nanquantile(means.reshape(-1), .95),
                    jnp.nanmedian(stds.reshape(-1))))


def reduce_microbatch_aux(aux):
  """Keep logit samples across GA; preserve existing sums for other aux leaves."""
  def reduce(path, value):
    if any(getattr(key, "key", None) == HEALTH_SAMPLE_KEY for key in path):
      return value.reshape((-1, value.shape[-1]))
    return jnp.sum(value, axis=0)
  return jax.tree_util.tree_map_with_path(reduce, aux)


def validate_config(keys):
  """Reject incompatible precision settings instead of silently ignoring them."""
  for name in ("logits_fp32_output_bf16_backward", "record_logits_health_metrics"):
    if type(keys[name]) is not bool:
      raise ValueError(f"{name} must be a bool")
  size = keys["logits_health_sample_size"]
  if type(size) is not int or size < 1:
    raise ValueError("logits_health_sample_size must be a positive integer")
  if keys["logits_fp32_output_bf16_backward"]:
    if jnp.dtype(keys["dtype"]) != jnp.dtype(jnp.bfloat16):
      raise ValueError("logits_fp32_output_bf16_backward requires dtype=bfloat16")
    if keys["logits_dot_in_fp32"]:
      raise ValueError("logits_fp32_output_bf16_backward cannot be combined with logits_dot_in_fp32")

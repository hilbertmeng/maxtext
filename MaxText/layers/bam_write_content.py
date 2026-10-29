"""Fixed write-content denominators; None preserves the original content RMSNorm."""
import math

def divisor(config, kind):
  value = getattr(config, 'bam_' + kind + '_write_content_divisor', None)
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
    raise ValueError('Write content divisor must be positive and finite: ' + kind)
  return float(value)


def fixed_content(content, tau):
  """Use the calibrated FP32 denominator, then round the output once to model dtype."""
  import jax.numpy as jnp
  return (content.astype(jnp.float32) / jnp.asarray(tau, jnp.float32)).astype(content.dtype)


def prepare_mlp_content(content, config):
  """Optionally normalize the complete content vector, before splitting write heads."""
  if not getattr(config, 'bam_mlp_write_content_global_rmsnorm', False):
    return content
  if divisor(config, 'mlp') != 1.0:
    raise ValueError('Global MLP write RMSNorm requires divisor1 to avoid a second per-head normalization')
  import jax.numpy as jnp
  from layers import initializers, normalizations
  statistics_dtype = (jnp.float32 if config.bam_write_rms_statistics_dtype == 'float32'
                      else content.dtype)
  return normalizations.rms_norm(
      content, dtype=content.dtype, epsilon=initializers.get_bam_write_epsilon(config),
      statistics_dtype=statistics_dtype)

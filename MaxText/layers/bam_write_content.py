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


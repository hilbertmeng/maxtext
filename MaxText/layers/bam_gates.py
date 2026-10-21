"""BAM semantic gate activations and their constant-opening initial biases."""
import math
import jax
import jax.numpy as jnp


def softcap_silu(x):
  """MemT SoftCapSiLU with tau=2; retain the caller's activation dtype."""
  value = x.astype(jnp.float32)
  return (2.0 * jnp.tanh(jax.nn.silu(value) / 2.0)).astype(x.dtype)


def activation(name):
  if name == 'softcap_silu':
    return softcap_silu
  if name in ('sigmoid', 'softplus'):
    return getattr(jax.nn, name)
  raise ValueError(f'Unknown BAM gate activation: {name}')


def bias_for_opening(opening, name):
  """Invert the positive, increasing branch without a worker SciPy dependency."""
  if name == 'sigmoid':
    if not 0 < opening < 1:
      raise ValueError('sigmoid opening must lie in (0, 1)')
    return math.log(opening / (1.0 - opening))
  if name == 'softplus':
    return math.log(math.expm1(opening))
  if name != 'softcap_silu':
    raise ValueError(f'Unknown BAM gate activation: {name}')
  if not 0 < opening < 2:
    raise ValueError('SoftCapSiLU opening must lie in (0, 2)')
  target = 2.0 * math.atanh(opening / 2.0)
  lo, hi = 0.0, 2.0 * target + 1.0
  for _ in range(64):
    mid = (lo + hi) / 2.0
    if mid / (1.0 + math.exp(-mid)) < target:
      lo = mid
    else:
      hi = mid
  return (lo + hi) / 2.0

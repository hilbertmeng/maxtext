"""Diagnostic-only routing probe for BAM MLP outputs at matrix-write layers.

tap(): identity whose backward emits the per-(token, head-chunk) first-order effect
sum_d dL/dy * y for one consumer path (residual or matrix write).
route_mask(): host-provided per-layer routing fractions, so masks change per batch
without recompiling. Never enabled in training.
"""
from functools import partial
import threading
import jax
import jax.numpy as jnp
import numpy as np

STORE = {}
MASK = {}
_lock = threading.Lock()


def _emit(path, layer, effect):
  with _lock:
    STORE.setdefault((int(layer), path), []).append(np.asarray(effect, np.float32))


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def tap(y, layer, path):
  return y


def _tap_fwd(y, layer, path):
  return y, (y, layer)


def _tap_bwd(path, residual, grad):
  y, layer = residual
  effect = jnp.sum(grad.astype(jnp.float32) * y.astype(jnp.float32), axis=-1)
  jax.debug.callback(partial(_emit, path), layer, effect)
  return grad, None


tap.defvjp(_tap_fwd, _tap_bwd)


def route_mask(layer, shape):
  def lookup(l):
    return np.asarray(MASK[int(l)], np.float32).reshape(shape)
  return jax.pure_callback(lookup, jax.ShapeDtypeStruct(shape, jnp.float32), layer)

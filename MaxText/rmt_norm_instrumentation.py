"""Scalar-only forward/backward taps for standalone RMT diagnosis."""
from functools import partial
import json
import os
import threading
import jax
import jax.numpy as jnp

_lock = threading.Lock()


def _emit(tag, direction, layer, stats):
  record = dict(tag=tag, direction=direction, layer=int(layer),
                rms=float(stats[0]), l2=float(stats[1]), absmax=float(stats[2]))
  filename = os.environ.get('RMT_NORM_TAP_FILE')
  if filename:
    with _lock, open(filename, 'a') as f:
      f.write(json.dumps(record) + '\n')


def _record(x, layer, tag, direction):
  y = x.astype(jnp.float32)
  ss = jnp.sum(jnp.square(y))
  stats = jnp.stack((jnp.sqrt(ss / y.size), jnp.sqrt(ss), jnp.max(jnp.abs(y))))
  jax.debug.callback(partial(_emit, tag, direction), layer, stats)


def _forward(x, layer, tag):
  _record(x, layer, tag, 'forward')
  return x


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def tap(x, layer, tag):
  return _forward(x, layer, tag)


def _tap_fwd(x, layer, tag):
  return _forward(x, layer, tag), layer


def _tap_bwd(tag, layer, grad):
  _record(grad, layer, tag, 'backward')
  return grad, None


tap.defvjp(_tap_fwd, _tap_bwd)

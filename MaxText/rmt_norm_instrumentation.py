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
  if direction == 'backward' and tag.endswith('mlp_write/content_raw'):
    token_rms = jnp.sqrt(jnp.mean(y*y,axis=tuple(range(2,y.ndim)))).reshape(-1)
    def emit_token_grad(l, values):
      filename = os.environ.get('RMT_NORM_TAP_FILE')
      if filename:
        with _lock,open(filename,'a') as f:
          f.write(json.dumps(dict(tag=tag+'_token_gradient',direction=direction,
                                  layer=int(l),token_rms=__import__('numpy').asarray(values).tolist()))+'\n')
    jax.lax.cond(layer>=16,lambda _:jax.debug.callback(emit_token_grad,layer,token_rms),lambda _:None,None)


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


@jax.custom_vjp
def scale_gradient_only(x, factor):
  return x


def _scale_gradient_fwd(x, factor):
  return x, factor


def _scale_gradient_bwd(factor, grad):
  return grad * factor.astype(grad.dtype), None


scale_gradient_only.defvjp(_scale_gradient_fwd, _scale_gradient_bwd)

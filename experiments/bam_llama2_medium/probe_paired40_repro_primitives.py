"""CPU candidate checks, not a TPU trajectory or full-model equivalence test.

Run with the diagnostics skill's pinned CPU Python, PYTHONPATH=MaxText.
Historical reference: 0038e21; current reproduction: 28aefca.
"""
import json

import jax
import jax.numpy as jnp
from flax import linen as nn
from layers.normalizations import rms_norm


class Projection(nn.Module):
  @nn.compact
  def __call__(self):
    return self.param('kernel', nn.initializers.normal(.006), (128, 64))


class Wrapper(nn.Module):
  projection_name: str

  @nn.compact
  def __call__(self):
    return Projection(name=self.projection_name)()


def split_norm(x):
  return jnp.stack([
      rms_norm(x[..., side, :], dtype=x.dtype, epsilon=1e-4,
               axis=(-2, -1)) for side in range(2)], axis=-2)


def merged_norm(x):
  return rms_norm(x, dtype=x.dtype, epsilon=1e-4, axis=(-3, -1))


def difference(a, b):
  a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  return dict(max_abs=float(jnp.max(jnp.abs(a-b))),
              unequal_fraction=float(jnp.mean(a != b)))


def main():
  result = dict(backend=jax.default_backend(), jax=jax.__version__)
  key = jax.random.PRNGKey(9876)
  values = []
  for name in ('W_local_qk_packed', 'W_local_packed'):
    model = Wrapper(name)
    values.append(model.apply(model.init(key)))
  result['module_rename_random_initializer'] = difference(*values)
  result['rms'] = {}
  for dtype in (jnp.float32, jnp.bfloat16):
    x = jax.random.normal(key, (2, 64, 16, 2, 2), dtype=dtype)
    cotangent = jax.random.normal(jax.random.fold_in(key, 1), x.shape, dtype=dtype)
    f, g = jax.jit(split_norm), jax.jit(merged_norm)
    result['rms'][str(dtype)] = dict(
        forward=difference(f(x), g(x)),
        backward=difference(jax.jit(lambda z: jax.vjp(split_norm, z)[1](cotangent)[0])(x),
                            jax.jit(lambda z: jax.vjp(merged_norm, z)[1](cotangent)[0])(x)))
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()

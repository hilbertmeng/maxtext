"""Compare historical bf16-statistics BAM RMS with the fp32-statistics refactor."""

import jax
import jax.numpy as jnp
import numpy as np


EPSILON = 1e-6


def old_rms(x, axis):
  return x * jax.lax.rsqrt(
      jnp.mean(x ** 2, axis=axis, keepdims=True) + EPSILON)


def new_rms(x, axis):
  x_f32 = x.astype(jnp.float32)
  y = x_f32 * jax.lax.rsqrt(
      jnp.mean(x_f32 ** 2, axis=axis, keepdims=True) + EPSILON)
  return y.astype(x.dtype)


def relative_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def probe(name, shape, axis, seed):
  x_key, upstream_key = jax.random.split(jax.random.PRNGKey(seed))
  base = jax.random.normal(x_key, shape, jnp.bfloat16)
  upstream = jax.random.normal(upstream_key, shape, jnp.bfloat16)
  print(f"case={name} shape={shape} axis={axis}")
  for scale in (0.001, 0.01, 0.1, 1.0):
    x = base * jnp.asarray(scale, jnp.bfloat16)

    def objective(fn, value):
      y = fn(value, axis)
      return jnp.sum(y.astype(jnp.float32) * upstream.astype(jnp.float32)), y

    (old_value, old_y), old_grad = jax.jit(jax.value_and_grad(
        lambda value: objective(old_rms, value), has_aux=True))(x)
    (new_value, new_y), new_grad = jax.jit(jax.value_and_grad(
        lambda value: objective(new_rms, value), has_aux=True))(x)
    print(
        f"scale={scale:g}",
        f"out_rel_l2={relative_l2(new_y, old_y):.6g}",
        f"grad_rel_l2={relative_l2(new_grad, old_grad):.6g}",
        f"objective_delta={float(new_value-old_value):+.6g}",
        f"old_dtype={old_y.dtype}",
        f"new_dtype={new_y.dtype}",
    )


def main():
  probe("read_key", (4, 256, 32), -1, 1)
  probe("head_mix", (4, 256, 16, 2), -2, 2)
  probe("write_factor", (4, 256, 16, 32), -1, 3)
  probe("fetch_mix", (4, 256, 16), -1, 4)


if __name__ == "__main__":
  main()

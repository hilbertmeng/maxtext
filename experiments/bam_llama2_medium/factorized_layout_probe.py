"""TPU value/gradient check for bnt+transpose versus direct-btn expansion."""

import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np


def old_pair(args):
  y_u, y_v, row_mix, col_mix = args
  y_u = jnp.einsum("btk,btn->bntk", y_u, col_mix)
  y_v = jnp.einsum("btv,btn->bntv", y_v, row_mix)
  return jnp.concatenate((
      jnp.transpose(y_u, (0, 2, 1, 3)),
      jnp.transpose(y_v, (0, 2, 1, 3))), axis=-1)


def new_pair(args):
  y_u, y_v, row_mix, col_mix = args
  y_u = jnp.einsum("btk,btn->btnk", y_u, col_mix)
  y_v = jnp.einsum("btv,btn->btnv", y_v, row_mix)
  return jnp.concatenate((y_u, y_v), axis=-1)


def main():
  b, t, n, k, v = 4, 256, 16, 32, 32
  keys = jax.random.split(jax.random.PRNGKey(1234), 6)
  args = (
      jax.random.normal(keys[0], (b, t, k), jnp.bfloat16),
      jax.random.normal(keys[1], (b, t, v), jnp.bfloat16),
      jax.random.normal(keys[2], (b, t, n), jnp.bfloat16),
      jax.random.normal(keys[3], (b, t, n), jnp.bfloat16),
  )
  upstream = jax.random.normal(keys[4], (b, t, n, k + v), jnp.bfloat16)
  old = jax.jit(old_pair)(args)
  new = jax.jit(new_pair)(args)
  print("forward_array_equal", bool(np.array_equal(np.asarray(old), np.asarray(new))))
  print("forward_max_abs", float(jnp.max(jnp.abs(old - new))))

  old_value, old_grad = jax.jit(jax.value_and_grad(
      lambda values: jnp.sum(old_pair(values).astype(jnp.float32)
                             * upstream.astype(jnp.float32))))(args)
  new_value, new_grad = jax.jit(jax.value_and_grad(
      lambda values: jnp.sum(new_pair(values).astype(jnp.float32)
                             * upstream.astype(jnp.float32))))(args)
  print("objective_abs_diff", float(jnp.abs(old_value - new_value)))
  for index, (old_item, new_item) in enumerate(zip(old_grad, new_grad)):
    print(f"grad{index}_array_equal", bool(np.array_equal(
        np.asarray(old_item), np.asarray(new_item))))
    print(f"grad{index}_max_abs", float(jnp.max(jnp.abs(old_item - new_item))))

  for name, fn in (("old", old_pair), ("new", new_pair)):
    hlo = jax.jit(fn).lower(args).compile().as_text()
    print(name + "_hlo_sha256", hashlib.sha256(hlo.encode()).hexdigest())
    for opcode in ("transpose", "dot", "multiply", "fusion", "copy"):
      print(name + "_hlo_" + opcode, len(re.findall(r"\b" + opcode + r"\b", hlo)))


if __name__ == "__main__":
  main()

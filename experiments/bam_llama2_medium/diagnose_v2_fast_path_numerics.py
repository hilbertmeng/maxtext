"""Measure TPU bf16 numerical differences in V2's two equivalent fast paths."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp


def _write_dot(u1, u2):
  return jnp.einsum("btnk,btnv->btkv", u1, u2)


def _write_mul(u1, u2):
  return jnp.sum(u1[..., None] * u2[..., None, :], axis=-3)


def _write_mul_f32_reduce(u1, u2):
  product = u1[..., None] * u2[..., None, :]
  return jnp.sum(product, axis=-3, dtype=jnp.float32).astype(u1.dtype)


def _read_combined(alpha, matrix):
  diagonal = jnp.arange(alpha.shape[-1])
  off_diagonal = alpha.at[..., diagonal, diagonal].set(0)
  return jnp.einsum("bfts,bskv->bftkv", off_diagonal, matrix) + matrix[:, None]


def _read_diagonal_one(alpha, matrix):
  diagonal = jnp.arange(alpha.shape[-1])
  alpha = alpha.at[..., diagonal, diagonal].set(jnp.asarray(1, alpha.dtype))
  return jnp.einsum("bfts,bskv->bftkv", alpha, matrix)


def _error(actual, reference):
  actual = actual.astype(jnp.float32)
  reference = reference.astype(jnp.float32)
  delta = actual - reference
  ref_rms = jnp.sqrt(jnp.mean(reference * reference))
  return {
      "max_abs": jnp.max(jnp.abs(delta)),
      "rms": jnp.sqrt(jnp.mean(delta * delta)),
      "relative_rms": jnp.sqrt(jnp.mean(delta * delta)) / jnp.maximum(ref_rms, 1e-12),
      "different_fraction": jnp.mean(delta != 0),
  }


def main():
  b, t, n, k, v = 2, 512, 16, 32, 32
  keys = jax.random.split(jax.random.PRNGKey(8172), 5)
  u1 = jax.random.normal(keys[0], (b, t, n, k), dtype=jnp.float32).astype(jnp.bfloat16)
  u2 = jax.random.normal(keys[1], (b, t, n, v), dtype=jnp.float32).astype(jnp.bfloat16)
  upstream = jax.random.normal(keys[2], (b, t, k, v), dtype=jnp.float32).astype(jnp.bfloat16)
  logits = jax.random.normal(keys[3], (b, 1, t, t), dtype=jnp.float32).astype(jnp.bfloat16)
  alpha = jax.nn.softmax(jnp.tril(logits), axis=-1).astype(jnp.bfloat16)

  write_functions = {
      "mul_bf16_reduce": _write_mul,
      "mul_f32_reduce": _write_mul_f32_reduce,
  }
  dot = jax.jit(_write_dot)(u1, u2)
  result = {"device": str(jax.devices()[0]), "write": {}, "read": {}}
  for name, function in write_functions.items():
    actual = jax.jit(function)(u1, u2)
    value_error = _error(actual, dot)
    dot_grads = jax.grad(
        lambda left, right: jnp.mean(_write_dot(left, right) * upstream), (0, 1)
    )(u1, u2)
    actual_grads = jax.grad(
        lambda left, right: jnp.mean(function(left, right) * upstream), (0, 1)
    )(u1, u2)
    result["write"][name] = {
        "value": value_error,
        "grad_u1": _error(actual_grads[0], dot_grads[0]),
        "grad_u2": _error(actual_grads[1], dot_grads[1]),
    }

  matrix = dot
  combined = jax.jit(_read_combined)(alpha, matrix)
  diagonal_one = jax.jit(_read_diagonal_one)(alpha, matrix)
  result["read"]["diagonal_one_vs_combined"] = _error(diagonal_one, combined)

  combined_dot = jax.jit(_read_combined)(alpha, dot)
  combined_mul = jax.jit(_read_diagonal_one)(alpha, jax.jit(_write_mul)(u1, u2))
  result["read"]["combined_fast_vs_original"] = _error(combined_mul, combined_dot)

  result = jax.device_get(result)
  print(json.dumps(jax.tree.map(lambda x: x.item() if hasattr(x, "item") else x, result), indent=2))


if __name__ == "__main__":
  main()

"""Actual BamAttention new-fp32-RMS versus legacy-bf16-RMS TPU probe."""

import os
import sys

from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import common_types
import max_utils
import pyconfig
from layers import attentions, initializers, normalizations


def value(x):
  return x.value if hasattr(x, "value") else x


def with_value(template, new_value):
  return template.replace(value=new_value) if hasattr(template, "value") else new_value


def relative_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def legacy_rms_norm(x, *, dtype, epsilon=1e-6, axis=-1):
  return jnp.asarray(
      x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=axis, keepdims=True) + epsilon),
      dtype)


def main():
  run_name = "rms-module-equivalence"
  os.makedirs(run_name, exist_ok=True)
  cfg = pyconfig.initialize(
      [sys.argv[0], "MaxText/configs/base.yml"],
      exp_class="BamLlama2MediumDirectPLocR256Gelu",
      run_name=run_name,
      enable_checkpointing=False,
      per_device_batch_size=1.0,
      max_target_length=8,
      max_prefill_predict_length=4,
      base_emb_dim=512,
      base_num_query_heads=4,
      base_num_kv_heads=4,
      base_mlp_dim=512,
      base_num_decoder_layers=1,
      scan_layers=False,
      attention="dot_product",
  )
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  module = attentions.BamAttention(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      mesh=mesh,
      attention_kernel="dot_product",
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=0.0,
      kernel_init=initializers.get_init_method(cfg.init_method),
      float32_qk_product=cfg.float32_qk_product,
      float32_logits=cfg.float32_logits,
      attention_type=cfg.attention_type,
      layer_mode="local_qk+local_o+full",
      read_side="both",
      bam_k=cfg.bam_k,
      bam_v=cfg.bam_v,
      name="self_attention",
  )

  rngs = jax.random.split(jax.random.PRNGKey(321), 9)
  shape = (1, cfg.max_target_length, cfg.base_emb_dim)
  inputs = jax.random.normal(rngs[0], shape, cfg.dtype)
  positions = jnp.arange(cfg.max_target_length, dtype=jnp.int32)[None]
  segments = jnp.ones(shape[:2], dtype=jnp.int32)
  M = jax.random.normal(
      rngs[1], shape[:2] + (cfg.bam_k, cfg.bam_v), cfg.dtype)
  args = (inputs, inputs, positions, segments)
  kwargs = dict(
      deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN, M_in=M)
  variables = module.init(
      {"params": rngs[2], "aqt": rngs[3]}, *args, **kwargs)

  # Activate every runtime-key and gate path; production key kernels start at zero.
  mutable = unfreeze(variables)
  for name, key in (
      ("W_lq", rngs[4]), ("W_lk", rngs[5]),
      ("W_lq_gate", rngs[6]), ("W_lk_gate", rngs[7]),
      ("W_R", rngs[8]),
  ):
    old = mutable["params"][name]["kernel"]
    old_value = value(old)
    mutable["params"][name]["kernel"] = with_value(
        old, 0.02 * jax.random.normal(key, old_value.shape, old_value.dtype))
  variables = freeze(mutable)

  upstream_out = jax.random.normal(rngs[0], shape, jnp.float32)
  upstream_M = jax.random.normal(rngs[1], M.shape, jnp.float32)

  def run():
    def outputs(v):
      out, M_out = module.apply(v, *args, **kwargs)
      return out.astype(jnp.float32), M_out.astype(jnp.float32)

    def objective(v):
      out, M_out = outputs(v)
      return jnp.sum(out * upstream_out) + jnp.sum(M_out * upstream_M)

    out, M_out = jax.jit(outputs)(variables)
    objective_value, grads = jax.jit(jax.value_and_grad(objective))(variables)
    return out, M_out, objective_value, grads

  new_rms_norm = normalizations.rms_norm
  new = run()

  def selective_legacy(mode):
    def rms_norm(x, *, dtype, epsilon=1e-6, axis=-1):
      canonical_axis = axis if axis >= 0 else axis + x.ndim
      use_legacy = (
          mode == "all"
          or (mode == "head_mix" and canonical_axis == x.ndim - 2)
          or (mode == "write" and x.ndim == 4 and canonical_axis == x.ndim - 1)
          or (mode == "fetch_mix" and x.ndim == 3 and x.shape[-1] == cfg.num_query_heads)
          or (mode == "read_key" and canonical_axis == x.ndim - 1
              and (x.ndim >= 5 or (x.ndim == 3 and x.shape[-1] != cfg.num_query_heads)))
      )
      fn = legacy_rms_norm if use_legacy else new_rms_norm
      return fn(x, dtype=dtype, epsilon=epsilon, axis=axis)
    return rms_norm

  new_grads = flatten_dict(unfreeze(new[3]))
  for mode in ("all", "read_key", "head_mix", "write", "fetch_mix"):
    normalizations.rms_norm = selective_legacy(mode)
    jax.clear_caches()
    candidate = run()
    candidate_grads = flatten_dict(unfreeze(candidate[3]))
    compared = []
    for key, candidate_grad in candidate_grads.items():
      if key not in new_grads:
        continue
      new_grad = value(new_grads[key])
      candidate_grad = value(candidate_grad)
      if float(jnp.linalg.norm(candidate_grad.astype(jnp.float32))) < 1e-8:
        continue
      compared.append((relative_l2(new_grad, candidate_grad), "/".join(key)))
    compared.sort(reverse=True)
    print(
        f"mode={mode}",
        f"output_relative_l2={relative_l2(new[0], candidate[0]):.8g}",
        f"M_out_relative_l2={relative_l2(new[1], candidate[1]):.8g}",
        f"objective_abs_diff={float(jnp.abs(new[2] - candidate[2])):.8g}",
        f"gradient_max={compared[0]}",
    )
    print(f"mode={mode} gradient_top10={compared[:10]}")


if __name__ == "__main__":
  main()

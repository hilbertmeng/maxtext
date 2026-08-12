"""TPU value/gradient equivalence check for packed factorized LocalQK."""

import os
import sys

from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import common_types
import max_utils
import pyconfig
from layers import attentions, initializers


BASE = "BamLlama2MediumDirectPLocR256Gelu"
PACKED = "BamLlama2MediumDirectPLocR256GeluPackedLocalQK"


def value(x):
  return x.value if hasattr(x, "value") else x


def with_value(template, new_value):
  return template.replace(value=new_value) if hasattr(template, "value") else new_value


def config(exp_class):
  run_name = f"{exp_class}-equivalence"
  os.makedirs(run_name, exist_ok=True)
  return pyconfig.initialize(
      [sys.argv[0], "MaxText/configs/base.yml"],
      exp_class=exp_class,
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


def module(cfg, mesh):
  return attentions.BamAttention(
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


def pack_params(base_variables, packed_variables):
  """Copy common leaves and concatenate the six baseline projection kernels."""
  base = flatten_dict(unfreeze(base_variables))
  packed = flatten_dict(unfreeze(packed_variables))
  for key in packed:
    if key in base and value(packed[key]).shape == value(base[key]).shape:
      packed[key] = with_value(packed[key], value(base[key]))

  root = ("params",)
  segments = [
      value(base[root + ("W_lq", "kernel")]),
      value(base[root + ("W_lq_gate", "kernel")]),
      value(base[root + ("W_lq_head_mix", "kernel")]).reshape(
          value(base[root + ("W_lq_head_mix", "kernel")]).shape[0], -1),
      value(base[root + ("W_lk", "kernel")]),
      value(base[root + ("W_lk_gate", "kernel")]),
      value(base[root + ("W_lk_head_mix", "kernel")]).reshape(
          value(base[root + ("W_lk_head_mix", "kernel")]).shape[0], -1),
  ]
  key = root + ("W_local_qk_packed", "kernel")
  packed[key] = with_value(packed[key], jnp.concatenate(segments, axis=-1))
  key = root + ("W_lq_bias",)
  packed[key] = with_value(packed[key], value(base[root + ("W_lq", "bias")]))
  key = root + ("W_lk_bias",)
  packed[key] = with_value(packed[key], value(base[root + ("W_lk", "bias")]))
  return freeze(unflatten_dict(packed))


def pack_grads(base_grads):
  base = flatten_dict(unfreeze(base_grads))
  root = ("params",)
  packed = {
      key: value(value_) for key, value_ in base.items()
      if not any(name in key for name in (
          "W_lq", "W_lk", "W_lq_gate", "W_lk_gate",
          "W_lq_head_mix", "W_lk_head_mix"))
  }
  segments = [
      value(base[root + ("W_lq", "kernel")]),
      value(base[root + ("W_lq_gate", "kernel")]),
      value(base[root + ("W_lq_head_mix", "kernel")]).reshape(
          value(base[root + ("W_lq_head_mix", "kernel")]).shape[0], -1),
      value(base[root + ("W_lk", "kernel")]),
      value(base[root + ("W_lk_gate", "kernel")]),
      value(base[root + ("W_lk_head_mix", "kernel")]).reshape(
          value(base[root + ("W_lk_head_mix", "kernel")]).shape[0], -1),
  ]
  packed[root + ("W_local_qk_packed", "kernel")] = jnp.concatenate(
      segments, axis=-1)
  packed[root + ("W_lq_bias",)] = value(base[root + ("W_lq", "bias")])
  packed[root + ("W_lk_bias",)] = value(base[root + ("W_lk", "bias")])
  packed[root + ("W_lq_gate_b0",)] = value(base[root + ("W_lq_gate_b0",)])
  packed[root + ("W_lk_gate_b0",)] = value(base[root + ("W_lk_gate_b0",)])
  return packed


def relative_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def cosine(a, b):
  a = np.asarray(a, np.float32).reshape(-1)
  b = np.asarray(b, np.float32).reshape(-1)
  return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)


def main():
  base_cfg = config(BASE)
  packed_cfg = config(PACKED)
  mesh = jax.sharding.Mesh(
      max_utils.create_device_mesh(base_cfg), base_cfg.mesh_axes)
  base_module = module(base_cfg, mesh)
  packed_module = module(packed_cfg, mesh)

  rngs = jax.random.split(jax.random.PRNGKey(123), 8)
  batch, length, embed = 1, base_cfg.max_target_length, base_cfg.base_emb_dim
  inputs = jax.random.normal(rngs[0], (batch, length, embed), base_cfg.dtype)
  positions = jnp.arange(length, dtype=jnp.int32)[None]
  segments = jnp.ones((batch, length), dtype=jnp.int32)
  M = jax.random.normal(
      rngs[1], (batch, length, base_cfg.bam_k, base_cfg.bam_v),
      base_cfg.dtype)
  args = (inputs, inputs, positions, segments)
  kwargs = dict(
      deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN, M_in=M)
  base_vars = base_module.init(
      {"params": rngs[2], "aqt": rngs[3]}, *args, **kwargs)
  packed_vars = packed_module.init(
      {"params": rngs[2], "aqt": rngs[3]}, *args, **kwargs)

  def base_output(variables):
    return base_module.apply(variables, *args, **kwargs)[0].astype(jnp.float32)

  def packed_output(variables):
    return packed_module.apply(variables, *args, **kwargs)[0].astype(jnp.float32)

  native_base_out = jax.jit(base_output)(base_vars)
  native_packed_out = jax.jit(packed_output)(packed_vars)
  upstream = jax.random.normal(rngs[0], native_base_out.shape, jnp.float32)
  _, native_base_grad = jax.jit(jax.value_and_grad(
      lambda variables: jnp.sum(base_output(variables) * upstream)))(base_vars)
  _, native_packed_grad = jax.jit(jax.value_and_grad(
      lambda variables: jnp.sum(packed_output(variables) * upstream)))(packed_vars)

  native_base = flatten_dict(unfreeze(base_vars))
  native_packed = flatten_dict(unfreeze(packed_vars))
  native_bg = flatten_dict(unfreeze(native_base_grad))
  native_pg = flatten_dict(unfreeze(native_packed_grad))
  root = ("params",)
  key_width = base_cfg.bam_k + base_cfg.bam_v
  mix_width = 2 * base_cfg.num_query_heads
  packed_kernel = value(native_packed[root + ("W_local_qk_packed", "kernel")])
  packed_grad = value(native_pg[root + ("W_local_qk_packed", "kernel")])
  split_points = (
      key_width, key_width + 2, key_width + 2 + mix_width,
      2 * key_width + 2 + mix_width,
      2 * key_width + 4 + mix_width)
  _, _, packed_q_mix, _, _, packed_k_mix = jnp.split(
      packed_kernel, split_points, axis=-1)
  packed_q_grad, _, _, packed_k_grad, _, _ = jnp.split(
      packed_grad, split_points, axis=-1)
  for side, packed_mix, packed_key_grad in (
      ("q", packed_q_mix, packed_q_grad),
      ("k", packed_k_mix, packed_k_grad),
  ):
    baseline_mix = value(native_base[
        root + (f"W_l{side}_head_mix", "kernel")]).reshape(
            packed_mix.shape)
    baseline_key_grad = value(native_bg[
        root + (f"W_l{side}", "kernel")])
    print(f"native_{side}_mix_rms_base", float(jnp.sqrt(jnp.mean(
        baseline_mix.astype(jnp.float32) ** 2))))
    print(f"native_{side}_mix_rms_packed", float(jnp.sqrt(jnp.mean(
        packed_mix.astype(jnp.float32) ** 2))))
    print(f"native_{side}_mix_cosine", cosine(packed_mix, baseline_mix))
    print(f"native_{side}_key_grad_norm_ratio", float(
        jnp.linalg.norm(packed_key_grad) / jnp.linalg.norm(baseline_key_grad)))
    print(f"native_{side}_key_grad_cosine", cosine(
        packed_key_grad, baseline_key_grad))
  print("native_forward_max_abs", float(jnp.max(jnp.abs(
      native_base_out - native_packed_out))))

  # Make LocalQK active while retaining identical non-LocalQK parameters.
  mutable = unfreeze(base_vars)
  for name, key in (("W_lq", rngs[4]), ("W_lk", rngs[5])):
    old = mutable["params"][name]["kernel"]
    old_value = value(old)
    mutable["params"][name]["kernel"] = with_value(
        old, 0.02 * jax.random.normal(key, old_value.shape, old_value.dtype))
  for name, key in (("W_lq_gate", rngs[6]), ("W_lk_gate", rngs[7])):
    old = mutable["params"][name]["kernel"]
    old_value = value(old)
    mutable["params"][name]["kernel"] = with_value(
        old, 0.02 * jax.random.normal(key, old_value.shape, old_value.dtype))
  base_vars = freeze(mutable)
  packed_vars = pack_params(base_vars, packed_vars)

  base_out = jax.jit(base_output)(base_vars)
  packed_out = jax.jit(packed_output)(packed_vars)
  print("forward_max_abs", float(jnp.max(jnp.abs(base_out - packed_out))))
  print("forward_relative_l2", relative_l2(packed_out, base_out))

  base_value, base_grad = jax.jit(jax.value_and_grad(
      lambda variables: jnp.sum(base_output(variables) * upstream)))(base_vars)
  packed_value, packed_grad = jax.jit(jax.value_and_grad(
      lambda variables: jnp.sum(packed_output(variables) * upstream)))(packed_vars)
  print("objective_abs_diff", float(jnp.abs(base_value - packed_value)))

  expected = pack_grads(base_grad)
  actual = {key: value(value_) for key, value_ in flatten_dict(
      unfreeze(packed_grad)).items()}
  compared = []
  for key, expected_value in expected.items():
    if key not in actual or actual[key].shape != expected_value.shape:
      continue
    compared.append((relative_l2(actual[key], expected_value), "/".join(key)))
  compared.sort(reverse=True)
  print("gradient_max_relative_l2", compared[0][0], compared[0][1])
  print("gradient_top5", compared[:5])


if __name__ == "__main__":
  main()

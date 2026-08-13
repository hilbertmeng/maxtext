"""TPU value/gradient equivalence check for BAM C256 implementation variants."""

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


MODES = (
    "legacy",
    "no_remat",
    "deferred_read",
    "diag_correction",
    "optimized",
)


def value(x):
  return x.value if hasattr(x, "value") else x


def with_value(template, new_value):
  return template.replace(value=new_value) if hasattr(template, "value") else new_value


def config(mode):
  run_name = f"qchunk-{mode}-equivalence"
  os.makedirs(run_name, exist_ok=True)
  return pyconfig.initialize(
      [sys.argv[0], "MaxText/configs/base.yml"],
      exp_class="BamV2QChunk256SixLayerProfile",
      run_name=run_name,
      enable_checkpointing=False,
      per_device_batch_size=1.0,
      max_target_length=8,
      max_prefill_predict_length=4,
      query_chunk_size=4,
      base_emb_dim=512,
      base_num_query_heads=4,
      base_num_kv_heads=4,
      base_mlp_dim=512,
      base_num_decoder_layers=1,
      scan_layers=False,
      bam_query_chunk_implementation=mode,
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
      attention_kernel="dot_product_chunk",
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=0.0,
      kernel_init=initializers.get_init_method(cfg.init_method),
      float32_qk_product=cfg.float32_qk_product,
      float32_logits=cfg.float32_logits,
      attention_type=cfg.attention_type,
      layer_mode="local_qk+full",
      read_side="both",
      bam_k=cfg.bam_k,
      bam_v=cfg.bam_v,
      name="self_attention",
  )


def relative_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def main():
  configs = {mode: config(mode) for mode in MODES}
  reference_cfg = configs["legacy"]
  mesh = jax.sharding.Mesh(
      max_utils.create_device_mesh(reference_cfg), reference_cfg.mesh_axes)
  modules = {mode: module(cfg, mesh) for mode, cfg in configs.items()}

  rngs = jax.random.split(jax.random.PRNGKey(73), 6)
  batch, length, embed = 1, reference_cfg.max_target_length, reference_cfg.base_emb_dim
  inputs = jax.random.normal(rngs[0], (batch, length, embed), reference_cfg.dtype)
  positions = jnp.arange(length, dtype=jnp.int32)[None]
  # Exercise the packed-data segment mask, including a segment boundary inside a chunk.
  segments = jnp.array([[1, 1, 1, 2, 2, 2, 2, 2]], dtype=jnp.int32)
  M = jax.random.normal(
      rngs[1], (batch, length, reference_cfg.bam_k, reference_cfg.bam_v),
      reference_cfg.dtype)
  args = (inputs, inputs, positions, segments)
  kwargs = dict(
      deterministic=True, model_mode=common_types.MODEL_MODE_TRAIN, M_in=M)
  variables = modules["legacy"].init(
      {"params": rngs[2], "aqt": rngs[3]}, *args, **kwargs)

  # W_R is zero-initialized in training. Activate it so fetched-read values and
  # gradients, rather than only the dormant MHA starting point, are compared.
  mutable = unfreeze(variables)
  flat = flatten_dict(mutable)
  for path, leaf in list(flat.items()):
    if path[-2:] == ("W_R", "kernel"):
      array = value(leaf)
      flat[path] = with_value(
          leaf, 0.02 * jax.random.normal(rngs[4], array.shape, array.dtype))
  variables = freeze(unflatten_dict(flat))
  upstream = jax.random.normal(
      rngs[5], (batch, length, embed), jnp.float32)

  def output(mode, params):
    return modules[mode].apply(params, *args, **kwargs)[0].astype(jnp.float32)

  results = {}
  for mode in MODES:
    objective = lambda params: jnp.sum(output(mode, params) * upstream)
    out = jax.jit(lambda params: output(mode, params))(variables)
    loss, grad = jax.jit(jax.value_and_grad(objective))(variables)
    results[mode] = (out, loss, flatten_dict(unfreeze(grad)))

  reference_out, reference_loss, reference_grad = results["legacy"]
  for mode in MODES:
    out, loss, grad = results[mode]
    grad_errors = [
        relative_l2(value(grad[path]), value(reference_grad[path]))
        for path in reference_grad
        if path in grad and value(grad[path]).shape == value(reference_grad[path]).shape
    ]
    print(
        mode,
        "forward_max_abs", float(jnp.max(jnp.abs(out - reference_out))),
        "forward_relative_l2", relative_l2(out, reference_out),
        "objective_abs_diff", float(jnp.abs(loss - reference_loss)),
        "gradient_max_relative_l2", max(grad_errors),
    )
if __name__ == "__main__":
  main()

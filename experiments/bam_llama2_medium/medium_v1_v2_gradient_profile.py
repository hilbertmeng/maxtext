"""Paired initialization-gradient profile for Medium BAM V1/V2.

Run this standalone diagnostic at each milestone's recorded commit.  It keeps
the full 24-layer model, uses the same Pile-eval batch, and writes only scalar
parameter/gradient/update statistics; no instrumentation is added to BAM.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import maxtext_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


_LAYER_RE = re.compile(r"(?:layers?|decoder_layer|layer)[_/-]?(\d+)")


def _path_name(path):
  return "/" + "/".join(str(part) for part in path)


def _global_l2(tree):
  return jnp.sqrt(sum(
      jnp.sum(jnp.square(value.astype(jnp.float32)))
      for value in jax.tree.leaves(tree)))


def _group_name(name):
  if name.endswith("/W_R/kernel"):
    return "fetched_W_R"
  if name.endswith("/W_R_gate/kernel"):
    return "fetched_gate_kernel"
  if name.endswith("/W_R_gate_b0"):
    return "fetched_gate_bias"
  if name.endswith("/W_local_qk_packed/kernel"):
    return "local_qk_packed"
  if "/W_lq" in name or "/W_lk" in name or "/local_qk" in name:
    return "local_qk_other"
  if name.endswith("/query/kernel"):
    return "mha_query"
  if name.endswith("/key/kernel"):
    return "mha_key"
  if name.endswith("/value/kernel"):
    return "mha_value"
  if name.endswith("/out/kernel"):
    return "mha_out"
  if "/P_loc_down/kernel" in name:
    return "write_P_loc_down"
  if "/P_loc_up/kernel" in name or "/P_loc/kernel" in name:
    return "write_P_loc"
  if "/W_g" in name or "/write" in name and "gate" in name.lower():
    return "write_gate"
  if "mlp" in name.lower() and name.endswith("/kernel"):
    return "mlp_kernel"
  if "embed" in name.lower():
    return "embedding"
  if name.endswith("/scale"):
    return "norm_scale"
  return "other"


def _aggregate_stats(flat_tree, bam_k):
  grouped = {}
  layerwise = {}
  for path, value in flat_tree.items():
    name = _path_name(path)
    group = _group_name(name)
    grouped.setdefault(group, []).append(value)
    if group == "fetched_W_R":
      grouped.setdefault("fetched_W_R_row", []).append(value[..., :bam_k])
      grouped.setdefault("fetched_W_R_col", []).append(value[..., bam_k:])
    match = _LAYER_RE.search(name)
    if match and group in {
        "fetched_W_R", "mha_query", "mha_key", "mha_value", "mha_out",
        "local_qk_packed", "local_qk_other", "write_P_loc_down",
        "write_P_loc", "write_gate",
    }:
      key = f"layer_{int(match.group(1)):02d}/{group}"
      layerwise.setdefault(key, []).append(value)
      if group == "fetched_W_R":
        layerwise.setdefault(
            f"layer_{int(match.group(1)):02d}/fetched_W_R_row", []).append(
                value[..., :bam_k])
        layerwise.setdefault(
            f"layer_{int(match.group(1)):02d}/fetched_W_R_col", []).append(
                value[..., bam_k:])

  def combine(values):
    count = sum(value.size for value in values)
    square = sum(jnp.sum(jnp.square(value.astype(jnp.float32)))
                 for value in values)
    absolute = sum(jnp.sum(jnp.abs(value.astype(jnp.float32)))
                   for value in values)
    maximum = jnp.max(jnp.stack([
        jnp.max(jnp.abs(value.astype(jnp.float32))) for value in values]))
    return {
        "count": jnp.asarray(count, jnp.int32),
        "l2": jnp.sqrt(square),
        "rms": jnp.sqrt(square / count),
        "mean_abs": absolute / count,
        "max_abs": maximum,
        "sum_sq": square,
    }

  return (
      {name: combine(values) for name, values in grouped.items()},
      {name: combine(values) for name, values in layerwise.items()},
  )


def _tree_metrics(tree, bam_k):
  return _aggregate_stats(flatten_dict(tree), bam_k)


def _host_tree_to_json(value):
  if isinstance(value, dict):
    return {key: _host_tree_to_json(child) for key, child in value.items()}
  array = np.asarray(value)
  return array.item() if array.ndim == 0 else array.tolist()


def _json_tree(value):
  # One batched device-to-host transfer avoids thousands of scalar TPU RPCs.
  return _host_tree_to_json(jax.device_get(value))


def _batch_fingerprint(batch):
  inputs = np.asarray(jax.device_get(batch["inputs"]), dtype=np.int32)
  return {
      "shape": list(inputs.shape),
      "sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
      "first_tokens": inputs.reshape(-1)[:32].tolist(),
  }


def _variant_state(state, variant):
  """Construct an initialization ablation without touching BAM production code."""
  if variant == "baseline":
    return state
  if variant not in {
      "wr_normal006", "gate0005", "wr_normal006_gate0005",
  }:
    raise ValueError(f"Unknown BAM_GRAD_VARIANT: {variant}")

  initialize_w_r = "wr_normal006" in variant
  gate_opening = 0.0005 if "gate0005" in variant else None
  normal_key = jax.random.PRNGKey(20260831)
  def replace(path, value):
    names = tuple(getattr(part, "key", str(part)) for part in path)
    if initialize_w_r and names[-2:] == ("W_R", "kernel"):
      seed = int(hashlib.sha256("/".join(names).encode()).hexdigest()[:8], 16)
      key = jax.random.fold_in(normal_key, seed)
      return (0.006 * jax.random.normal(
          key, value.shape, dtype=jnp.float32)).astype(value.dtype)
    if gate_opening is not None and names[-1] == "W_R_gate_b0":
      logit = np.log(gate_opening / (1.0 - gate_opening))
      return jnp.full_like(value, logit)
    return value
  return state.replace(params=jax.tree_util.tree_map_with_path(
      replace, state.params))


def run(config):
  output_path = Path(os.environ.get(
      "BAM_GRAD_OUTPUT", "/tmp/medium_bam_gradient_profile.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  trace_steps = int(os.environ.get("BAM_GRAD_STEPS", "3"))
  variants = tuple(filter(None, os.environ.get(
      "BAM_GRAD_VARIANTS", "baseline").split(",")))
  if len(variants) > 1 and trace_steps != 1:
    raise ValueError("Multi-variant initialization profiles require BAM_GRAD_STEPS=1")

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  batches = [next(eval_iterator) for _ in range(trace_steps)]

  def diagnostic_step(state, batch, rng):
    grad_fn = jax.value_and_grad(train.loss_fn, argnums=4, has_aux=True)
    (loss, _), raw_grads = grad_fn(
        model, config, batch, rng, state.params, is_train=True)
    raw_norm = _global_l2(raw_grads)
    clipped = maxtext_utils.apply_gradient_clipping(
        raw_grads, state, config.gradient_clipping_threshold)
    clipped_norm = _global_l2(clipped)
    raw_groups, raw_layers = _tree_metrics(raw_grads, config.bam_k)
    clipped_groups, _ = _tree_metrics(clipped, config.bam_k)
    param_groups, _ = _tree_metrics(state.params, config.bam_k)
    new_state = state.apply_gradients(grads=clipped)
    updates = jax.tree_util.tree_map(
        lambda new, old: new - old, new_state.params, state.params)
    update_groups, update_layers = _tree_metrics(updates, config.bam_k)
    return {
        "loss": loss,
        "raw_grad_norm": raw_norm,
        "clipped_grad_norm": clipped_norm,
        "clip_multiplier": clipped_norm / jnp.maximum(raw_norm, 1e-30),
        "raw_grad_groups": raw_groups,
        "raw_grad_layers": raw_layers,
        "clipped_grad_groups": clipped_groups,
        "param_groups": param_groups,
        "update_groups": update_groups,
        "update_layers": update_layers,
    }

  step_fn = jax.jit(diagnostic_step)
  report = {
      "metadata": {
          "exp_class": config.exp_class,
          "bam_k": config.bam_k,
          "bam_v": config.bam_v,
          "bam_abs_v_compression_dim": getattr(
              config, "bam_abs_v_compression_dim", None),
          "num_layers": config.num_decoder_layers,
          "sequence_length": config.max_target_length,
          "per_device_batch_size": config.per_device_batch_size,
          "gradient_clipping_threshold": config.gradient_clipping_threshold,
          "jax_version": jax.__version__,
          "git_commit": os.environ.get("BAM_GRAD_GIT_COMMIT", ""),
          "batch": _batch_fingerprint(batches[0]),
      },
      "steps": {},
  }
  if len(variants) == 1:
    state = _variant_state(state, variants[0])
    report["metadata"]["variant"] = variants[0]
    for step, batch in enumerate(batches):
      rng = jax.random.fold_in(init_rng, step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics = step_fn(state, batch, rng)
      report["steps"][str(step)] = _json_tree(metrics)
      print(
          f"BAM_GRAD variant={variants[0]} step={step} "
          f"loss={report['steps'][str(step)]['loss']:.8f} "
          f"raw={report['steps'][str(step)]['raw_grad_norm']:.6f} "
          f"clip={report['steps'][str(step)]['clip_multiplier']:.6f}",
          flush=True)
  else:
    report["metadata"]["variants"] = variants
    report["variants"] = {}
    for variant in variants:
      variant_state = _variant_state(state, variant)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics = step_fn(variant_state, batches[0], init_rng)
      report["variants"][variant] = _json_tree(metrics)
      print(
          f"BAM_GRAD variant={variant} step=0 "
          f"loss={report['variants'][variant]['loss']:.8f} "
          f"raw={report['variants'][variant]['raw_grad_norm']:.6f} "
          f"clip={report['variants'][variant]['clip_multiplier']:.6f}",
          flush=True)

  report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_GRAD_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

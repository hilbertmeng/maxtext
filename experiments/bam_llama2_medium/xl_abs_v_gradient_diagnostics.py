"""Trace the first BAM updates while varying fetched-M AbsV width.

This diagnostic stays outside ``BamAttention``.  It reports per-parameter and
per-coordinate gradient/update norms, splits fetched ``W_R`` into its row/col
keys, and measures the fetched read added to the standard attention output.
Use a short, shallow XL config so C=8/16/32 can be compared on one v6e chip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from absl import app
from flax.core import freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import max_utils
import maxtext_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train
from xl_abs_v_width_diagnostics import _stack_captures


_GROUP_SUFFIXES = {
    "W_R": "/W_R/kernel",
    "W_R_gate": "/W_R_gate/kernel",
    "W_R_gate_b0": "/W_R_gate_b0",
    "abs_v_projection": "/abs_v_cache_projection",
    "fetch_head_mix": "/fetch_head_mix/kernel",
    "P_loc_down": "/P_loc_down/kernel",
    "P_loc_up": "/P_loc_up/kernel",
    "local_qk_packed": "/W_local_qk_packed/kernel",
    "query": "/query/kernel",
    "key": "/key/kernel",
    "value": "/value/kernel",
    "out": "/out/kernel",
}


def _capture_fetch(module, method_name):
  del module
  return method_name in ("_read_fetched_m", "_query_chunk_op")


def _tree_group_stats(tree, bam_k):
  flat = flatten_dict(tree)
  groups = {name: [] for name in _GROUP_SUFFIXES}
  groups.update({"W_R_row": [], "W_R_col": []})
  for path, value in flat.items():
    name = "/" + "/".join(path)
    for group, suffix in _GROUP_SUFFIXES.items():
      if name.endswith(suffix):
        groups[group].append(value)
    if name.endswith(_GROUP_SUFFIXES["W_R"]):
      groups["W_R_row"].append(value[..., :bam_k])
      groups["W_R_col"].append(value[..., bam_k:])

  result = {}
  for name, values in groups.items():
    if not values:
      continue
    count = sum(value.size for value in values)
    square = sum(jnp.sum(jnp.square(value.astype(jnp.float32)))
                 for value in values)
    absolute = sum(jnp.sum(jnp.abs(value.astype(jnp.float32)))
                   for value in values)
    maximum = jnp.max(jnp.stack([
        jnp.max(jnp.abs(value.astype(jnp.float32))) for value in values]))
    l2 = jnp.sqrt(square)
    result[name] = {
        "count": jnp.asarray(count, jnp.int32),
        "l2": l2,
        "rms": jnp.sqrt(square / count),
        "mean_abs": absolute / count,
        "max_abs": maximum,
    }
  return result


def _global_l2(tree):
  return jnp.sqrt(sum(
      jnp.sum(jnp.square(value.astype(jnp.float32)))
      for value in jax.tree.leaves(tree)))


def _set_fetched_gate_init(params, initial_gate):
  """Replace only fetched-read gate biases in an initialized parameter tree."""
  if not 0.0 < initial_gate < 1.0:
    raise ValueError(f"invalid fetched gate initialization: {initial_gate}")
  flat = flatten_dict(unfreeze(params))
  matches = [path for path in flat if path[-1] == "W_R_gate_b0"]
  if not matches:
    raise ValueError("no fetched W_R_gate_b0 parameters found")
  logit = np.log(initial_gate / (1.0 - initial_gate))
  for path in matches:
    flat[path] = jnp.full_like(flat[path], logit)
  return freeze(unflatten_dict(flat))


def _capture_read_metrics(model, config, params, batch, rng, read_v_dim):
  rng1, params_rng = jax.random.split(rng)
  (_, _, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": params_rng},
      mutable=["intermediates"],
      capture_intermediates=_capture_fetch,
  )
  captured = _stack_captures(collections, config.num_decoder_layers)
  read = captured["read"].astype(jnp.float32)
  y_std = captured["y_std"].astype(jnp.float32)
  mask = (batch["targets_segmentation"] != 0).astype(jnp.float32)
  mask = mask[None, ..., None, None]

  def energy(x):
    square = jnp.sum(jnp.square(x) * mask, axis=tuple(range(1, x.ndim)))
    return square

  col = read[..., :config.bam_k]
  row = read[..., config.bam_k:config.bam_k + read_v_dim]
  read_square = energy(read)
  std_square = energy(y_std)
  return {
      "read_to_std": jnp.sqrt(read_square / jnp.maximum(std_square, 1e-12)),
      "col_to_std": jnp.sqrt(energy(col) / jnp.maximum(std_square, 1e-12)),
      "row_to_std": jnp.sqrt(energy(row) / jnp.maximum(std_square, 1e-12)),
      "read_rms": jnp.sqrt(jnp.mean(jnp.square(read), axis=(1, 2, 3, 4))),
      "std_rms": jnp.sqrt(jnp.mean(jnp.square(y_std), axis=(1, 2, 3, 4))),
  }


def _json_tree(value):
  if isinstance(value, dict):
    return {key: _json_tree(child) for key, child in value.items()}
  array = np.asarray(jax.device_get(value))
  return array.item() if array.ndim == 0 else array.tolist()


def run(config):
  output_path = Path(os.environ.get(
      "BAM_ABSV_GRAD_OUTPUT", "/tmp/xl_abs_v_gradient_diagnostics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  trace_steps = int(os.environ.get("BAM_ABSV_GRAD_STEPS", "10"))
  capture_steps = {
      int(step) for step in os.environ.get(
          "BAM_ABSV_GRAD_CAPTURE_STEPS", "0,1,2,5,10").split(",")
      if step
  }
  read_v_dim = config.bam_abs_v_compression_dim or config.bam_v
  fetched_gate_init = os.environ.get("BAM_ABSV_GRAD_FETCH_GATE_INIT")

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  if fetched_gate_init is not None:
    state = state.replace(params=_set_fetched_gate_init(
        state.params, float(fetched_gate_init)))
  batches = [next(eval_iterator) for _ in range(trace_steps + 1)]

  def diagnostic_step(state, batch, rng):
    grad_fn = jax.value_and_grad(train.loss_fn, argnums=4, has_aux=True)
    (loss, _), raw_grads = grad_fn(
        model, config, batch, rng, state.params, is_train=True)
    raw_norm = _global_l2(raw_grads)
    clipped = maxtext_utils.apply_gradient_clipping(
        raw_grads, state, config.gradient_clipping_threshold)
    new_state = state.apply_gradients(grads=clipped)
    updates = jax.tree.map(
        lambda new, old: new - old, new_state.params, state.params)
    metrics = {
        "loss": loss,
        "raw_grad_norm": raw_norm,
        "clip_multiplier": jnp.minimum(
            1.0, config.gradient_clipping_threshold / jnp.maximum(raw_norm, 1e-12)),
        "raw_grads": _tree_group_stats(raw_grads, config.bam_k),
        "params_before": _tree_group_stats(state.params, config.bam_k),
        "updates": _tree_group_stats(updates, config.bam_k),
        "params_after": _tree_group_stats(new_state.params, config.bam_k),
    }
    return new_state, metrics

  step_fn = jax.jit(diagnostic_step)
  capture_fn = jax.jit(lambda params, batch, rng: _capture_read_metrics(
      model, config, params, batch, rng, read_v_dim))
  report = {
      "metadata": {
          "exp_class": config.exp_class,
          "bam_k": config.bam_k,
          "bam_v": config.bam_v,
          "read_v_dim": read_v_dim,
          "num_layers": config.num_decoder_layers,
          "sequence_length": config.max_target_length,
          "gradient_clipping_threshold": config.gradient_clipping_threshold,
          "fetched_gate_init_override": (
              None if fetched_gate_init is None else float(fetched_gate_init)),
      },
      "steps": {},
      "captures": {},
  }
  for step in range(trace_steps + 1):
    rng = jax.random.fold_in(init_rng, step)
    if step in capture_steps:
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        captured = capture_fn(state.params, batches[step], rng)
      report["captures"][str(step)] = _json_tree(captured)
    if step == trace_steps:
      break
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics = step_fn(state, batches[step], rng)
    report["steps"][str(step)] = _json_tree(metrics)
    print(
        f"BAM_ABSV_GRAD step={step} loss={report['steps'][str(step)]['loss']:.8f} "
        f"raw={report['steps'][str(step)]['raw_grad_norm']:.6f} "
        f"clip={report['steps'][str(step)]['clip_multiplier']:.6f}",
        flush=True)

  report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_ABSV_GRAD_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

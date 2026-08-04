"""Read-only loss ablation for one full-read W_R fetch slice.

The checkpoint is restored once and never written.  All scales use the same eval
batch and the same compiled forward.  No BAM diagnostic activations are captured.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from absl import app
from flax.core import FrozenDict, freeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


def _find_wr_kernel(params: Any, layer: int) -> tuple[tuple[str, ...], jax.Array]:
  layer_name = f"layers_{layer}"
  matches = [
      (path, value)
      for path, value in flatten_dict(params).items()
      if layer_name in path and len(path) >= 2 and path[-2:] == ("W_R", "kernel")
  ]
  if len(matches) != 1:
    paths = ["/".join(path) for path, _ in matches]
    raise ValueError(f"Expected one layer-{layer} W_R kernel, found {len(matches)}: {paths}")
  return matches[0]


def _scale_fetch(params: Any, path: tuple[str, ...], fetch: int, scale: float) -> Any:
  """Return a new pytree; the restored checkpoint arrays remain untouched."""
  flat = dict(flatten_dict(params))
  kernel = flat[path]
  if fetch < 0 or fetch >= kernel.shape[-2]:
    raise ValueError(f"fetch={fetch} outside W_R shape {kernel.shape}")
  flat[path] = kernel.at[..., fetch, :].multiply(scale)
  updated = unflatten_dict(flat)
  return freeze(updated) if isinstance(params, FrozenDict) else updated


def _forward(model, params, batch, rng):
  dropout_rng, params_rng = jax.random.split(rng)
  xent, correct, _ = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng},
  )
  mask = batch["targets_segmentation"] != 0
  weights = jnp.sum(mask, axis=-1)
  sequence_loss = jnp.sum(xent * mask, axis=-1) / jnp.maximum(weights, 1)
  total_weights = jnp.sum(weights)
  return {
      "loss": jnp.sum(xent * mask) / jnp.maximum(total_weights, 1),
      "accuracy": correct / jnp.maximum(total_weights, 1),
      "sequence_loss": sequence_loss,
      "sequence_weights": weights,
  }


def _delta_summary(delta: np.ndarray) -> dict[str, Any]:
  delta = np.asarray(delta, np.float64)
  return {
      "mean": float(np.mean(delta)),
      "std": float(np.std(delta)),
      "min": float(np.min(delta)),
      "p25": float(np.percentile(delta, 25)),
      "median": float(np.median(delta)),
      "p75": float(np.percentile(delta, 75)),
      "max": float(np.max(delta)),
      "improved_count": int(np.sum(delta < 0)),
      "worsened_count": int(np.sum(delta > 0)),
      "unchanged_count": int(np.sum(delta == 0)),
  }


def run(config) -> None:
  started = time.perf_counter()
  if not config.bam_enabled or config.bam_diagnostics:
    raise ValueError("Requires bam_enabled=True and bam_diagnostics=False")
  if not config.only_eval:
    raise ValueError("Ablation is inference-only; pass only_eval=True")

  layer = int(os.environ.get("BAM_ABLATION_LAYER", "17"))
  fetch = int(os.environ.get("BAM_ABLATION_FETCH", "1"))
  scales = [float(value) for value in os.environ.get("BAM_ABLATION_SCALES", "1,0.5,0.25,0").split(",")]
  if 1.0 not in scales:
    raise ValueError("BAM_ABLATION_SCALES must include 1 as the paired baseline")
  output_path = Path(os.environ.get("BAM_ABLATION_OUTPUT", "/tmp/bam_wr_ablation.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )
  wr_path, wr_kernel = _find_wr_kernel(state.params, layer)
  if wr_kernel.shape[-2] != config.bam_n_f:
    raise ValueError(f"W_R fetch axis mismatch: shape={wr_kernel.shape}, bam_n_f={config.bam_n_f}")
  setup_seconds = time.perf_counter() - started

  data_started = time.perf_counter()
  batch = next(eval_data_iterator)
  jax.block_until_ready(batch)
  data_seconds = time.perf_counter() - data_started
  inputs_host = np.asarray(jax.device_get(batch["inputs"]))
  sequence_hashes = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in inputs_host]
  sampled_hashes = [hashlib.sha256(row[::32].tobytes()).hexdigest()[:12] for row in inputs_host]

  compiled_forward = jax.jit(lambda params, batch, rng: _forward(model, params, batch, rng))
  results = {}
  execution_order = []
  for run_index, scale in enumerate(scales):
    scaled_params = state.params if scale == 1.0 else _scale_fetch(state.params, wr_path, fetch, scale)
    run_rng = jax.random.fold_in(init_rng, 0)
    scale_started = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      metrics = compiled_forward(scaled_params, batch, run_rng)
    metrics = jax.device_get(jax.block_until_ready(metrics))
    elapsed = time.perf_counter() - scale_started
    key = f"{scale:g}"
    execution_order.append(key)
    results[key] = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "sequence_loss": np.asarray(metrics["sequence_loss"]).tolist(),
        "sequence_weights": np.asarray(metrics["sequence_weights"]).astype(int).tolist(),
        "seconds": elapsed,
        "run_index": run_index,
    }
    print(
        f"BAM_WR_ABLATION scale={key} loss={results[key]['loss']:.8f} "
        f"accuracy={results[key]['accuracy']:.8f} seconds={elapsed:.3f}",
        flush=True,
    )

  baseline = np.asarray(results["1"]["sequence_loss"], np.float64)
  baseline_loss = results["1"]["loss"]
  for key, result in results.items():
    delta = np.asarray(result["sequence_loss"], np.float64) - baseline
    result["loss_delta_vs_1"] = float(result["loss"] - baseline_loss)
    result["relative_loss_delta_vs_1"] = float((result["loss"] - baseline_loss) / baseline_loss)
    result["sequence_loss_delta_vs_1"] = delta.tolist()
    result["sequence_delta_summary"] = _delta_summary(delta)

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "layer": layer,
          "fetch": fetch,
          "wr_path": "/".join(wr_path),
          "wr_shape": list(wr_kernel.shape),
          "execution_order": execution_order,
          "eval_batch_size": int(inputs_host.shape[0]),
          "valid_tokens": int(sum(results["1"]["sequence_weights"])),
          "eval_shuffle_buffer_size": config.eval_shuffle_buffer_size,
          "data_shuffle_seed": config.data_shuffle_seed,
          "sequence_hashes": sequence_hashes,
          "sampled_sequence_hashes": sampled_hashes,
          "setup_seconds": setup_seconds,
          "data_seconds": data_seconds,
          "total_seconds": time.perf_counter() - started,
          "checkpoint_mutated": False,
      },
      "results": results,
  }
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"BAM_WR_ABLATION_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Read-only FetchRank diagnostics on fixed Pile eval batches.

Restores one rank>1 checkpoint, compares its original loss with route mixer
parameters tied to their mean/first/second route, and reports per-layer route
divergence.  The restored checkpoint is never mutated.
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


def _fetch_mixer_paths(params: Any) -> list[tuple[tuple[str, ...], jax.Array]]:
  matches = []
  for path, value in flatten_dict(params).items():
    if len(path) >= 2 and path[-2] == "fetch_head_mix" and path[-1] in ("kernel", "bias"):
      if value.shape[-2] <= 1:
        raise ValueError(f"Fetch mixer has no route axis: {'/'.join(path)} {value.shape}")
      matches.append((path, value))
  if not matches:
    raise ValueError("No fetch_head_mix kernel/bias parameters found")
  return matches


def _tie_routes(params: Any, mode: str) -> Any:
  flat = dict(flatten_dict(params))
  for path, value in _fetch_mixer_paths(params):
    if mode == "mean":
      tied = jnp.mean(value, axis=-2, keepdims=True)
    elif mode == "first":
      tied = value[..., :1, :]
    elif mode == "second":
      tied = value[..., 1:2, :]
    else:
      raise ValueError(f"Unknown tie mode: {mode}")
    flat[path] = jnp.broadcast_to(tied, value.shape)
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
  total_weights = jnp.sum(weights)
  return {
      "loss": jnp.sum(xent * mask) / jnp.maximum(total_weights, 1),
      "sequence_loss": jnp.sum(xent * mask, axis=-1) / jnp.maximum(weights, 1),
      "sequence_weights": weights,
      "accuracy": correct / jnp.maximum(total_weights, 1),
  }


def _delta_summary(delta: np.ndarray) -> dict[str, Any]:
  return {
      "mean": float(np.mean(delta)),
      "std": float(np.std(delta)),
      "p25": float(np.percentile(delta, 25)),
      "median": float(np.median(delta)),
      "p75": float(np.percentile(delta, 75)),
      "min": float(np.min(delta)),
      "max": float(np.max(delta)),
      "improved_count": int(np.sum(delta < 0)),
      "worsened_count": int(np.sum(delta > 0)),
  }


def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
  a = np.asarray(a, np.float64).reshape(-1)
  b = np.asarray(b, np.float64).reshape(-1)
  a_rms = np.sqrt(np.mean(a * a))
  b_rms = np.sqrt(np.mean(b * b))
  denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
  return {
      "cosine": float(np.dot(a, b) / denom),
      "route0_rms": float(a_rms),
      "route1_rms": float(b_rms),
      "difference_rms": float(np.sqrt(np.mean(np.square(a - b)))),
      "difference_to_mean_rms": float(
          np.sqrt(np.mean(np.square(a - b))) / max((a_rms + b_rms) / 2, 1e-30)),
  }


def _parameter_route_stats(params: Any) -> dict[str, Any]:
  output = {}
  for path, value in _fetch_mixer_paths(params):
    host = np.asarray(jax.device_get(value))
    name = "/".join(path)
    output[name] = {"shape": list(host.shape), "aggregate": _pair_stats(
        np.take(host, 0, axis=-2), np.take(host, 1, axis=-2))}
    if host.ndim >= 4 and host.shape[0] > 1:
      output[name]["per_layer"] = [
          _pair_stats(np.take(layer, 0, axis=-2), np.take(layer, 1, axis=-2))
          for layer in host
      ]
  return output


def run(config) -> None:
  started = time.perf_counter()
  if not config.bam_enabled or int(config.bam_fetch_rank) <= 1:
    raise ValueError("Requires a BAM FetchRank>1 configuration")
  if config.bam_diagnostics or not config.only_eval:
    raise ValueError("Use bam_diagnostics=False and only_eval=True")

  output_path = Path(os.environ.get(
      "BAM_FETCH_RANK_OUTPUT", "/tmp/bam_fetch_rank_diagnostics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  num_batches = int(os.environ.get("BAM_FETCH_RANK_BATCHES", "1"))

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  setup_seconds = time.perf_counter() - started

  variants = {
      "original": state.params,
      "tied_mean": _tie_routes(state.params, "mean"),
      "tied_first": _tie_routes(state.params, "first"),
      "tied_second": _tie_routes(state.params, "second"),
  }
  compiled_forward = jax.jit(
      lambda params, batch, rng: _forward(model, params, batch, rng))
  collected = {name: {"loss": [], "sequence_loss": [], "sequence_weights": []}
               for name in variants}
  sequence_hashes = []
  execution_seconds = {name: 0.0 for name in variants}

  for batch_index in range(num_batches):
    batch = next(eval_data_iterator)
    jax.block_until_ready(batch)
    tokens = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in tokens)
    for name, params in variants.items():
      run_rng = jax.random.fold_in(init_rng, batch_index)
      variant_started = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics = compiled_forward(params, batch, run_rng)
      metrics = jax.device_get(jax.block_until_ready(metrics))
      execution_seconds[name] += time.perf_counter() - variant_started
      collected[name]["loss"].append(float(metrics["loss"]))
      collected[name]["sequence_loss"].extend(
          np.asarray(metrics["sequence_loss"], np.float64).tolist())
      collected[name]["sequence_weights"].extend(
          np.asarray(metrics["sequence_weights"], np.int64).tolist())

  baseline = np.asarray(collected["original"]["sequence_loss"], np.float64)
  results = {}
  for name, values in collected.items():
    sequence_loss = np.asarray(values["sequence_loss"], np.float64)
    weights = np.asarray(values["sequence_weights"], np.float64)
    loss = float(np.sum(sequence_loss * weights) / np.sum(weights))
    delta = sequence_loss - baseline
    results[name] = {
        "loss": loss,
        "loss_delta_vs_original": float(loss - (
            np.sum(baseline * weights) / np.sum(weights))),
        "sequence_delta_summary": _delta_summary(delta),
        "seconds": execution_seconds[name],
    }
    print(
        f"BAM_FETCH_RANK variant={name} loss={loss:.8f} "
        f"dloss={results[name]['loss_delta_vs_original']:+.8f}", flush=True)

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "exp_class": config.exp_class,
          "fetch_rank": int(config.bam_fetch_rank),
          "batches": num_batches,
          "sequences": len(sequence_hashes),
          "valid_tokens": int(sum(collected["original"]["sequence_weights"])),
          "sequence_hashes": sequence_hashes,
          "setup_seconds": setup_seconds,
          "total_seconds": time.perf_counter() - started,
          "checkpoint_mutated": False,
      },
      "parameter_route_stats": _parameter_route_stats(state.params),
      "results": results,
  }
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"BAM_FETCH_RANK_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

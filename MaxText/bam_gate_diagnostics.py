"""Measure learned BAM write/read gate usage on shuffled Pile eval examples.

This runner restores a checkpoint read-only and captures only the DenseGeneral
outputs needed to reconstruct gates.  Statistics stay outside attention code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


_LAYER_RE = re.compile(r"layers_(\d+)")
_CAPTURE_NAMES = frozenset((
    "W_gw",
    "W_R_gate",
    "W_lq_gate",
    "W_lk_gate",
    "W_lq_head_mix",
    "W_lk_head_mix",
))
_BIAS_NAMES = {
    "gw_b0": "write",
    "W_R_gate_b0": "fetch",
    "W_lq_gate_b0": "local_q",
    "W_lk_gate_b0": "local_k",
}
_EPS = 1.0e-12


def _unwrap(value: Any) -> Any:
  while isinstance(value, (tuple, list)) and len(value) == 1:
    value = value[0]
  return value


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for component in path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _capture_gate_projection(module, method_name: str) -> bool:
  return method_name == "__call__" and module.name in _CAPTURE_NAMES


def _group_captures(collections) -> dict[int, dict[str, jax.Array]]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  for path, value in flatten_dict(collections.get("intermediates", {})).items():
    layer = _layer_from_path(path)
    if layer is None:
      continue
    names = _CAPTURE_NAMES.intersection(path)
    if len(names) == 1:
      grouped[layer][next(iter(names))] = _unwrap(value)
  return dict(sorted(grouped.items()))


def _extract_biases(params, num_layers: int) -> dict[int, dict[str, jax.Array]]:
  biases: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  for path, value in flatten_dict(params).items():
    layer = _layer_from_path(path)
    if layer is not None and path[-1] in _BIAS_NAMES:
      biases[layer][_BIAS_NAMES[path[-1]]] = value
  required = set(_BIAS_NAMES.values())
  for layer in range(num_layers):
    missing = required - biases[layer].keys()
    if missing:
      raise KeyError(f"layer {layer} is missing gate biases: {sorted(missing)}")
  return dict(biases)


def _sigmoid_with_compute_dtype(logits, bias):
  return jax.nn.sigmoid(logits + jnp.asarray(bias, logits.dtype))


def _forward(model, config, params, biases, batch, rng, stride):
  dropout_rng, params_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng},
      mutable=["intermediates"],
      capture_intermediates=_capture_gate_projection,
  )
  captures = _group_captures(collections)
  if len(captures) != config.num_decoder_layers:
    raise RuntimeError(
        f"expected {config.num_decoder_layers} captured layers, got {len(captures)}")

  sampled = {}
  for layer, values in captures.items():
    missing = _CAPTURE_NAMES - values.keys()
    if missing:
      raise KeyError(f"layer {layer} is missing captures: {sorted(missing)}")
    fetch_gate = _sigmoid_with_compute_dtype(
        values["W_R_gate"], biases[layer]["fetch"])
    if fetch_gate.shape[-2] != 1:
      raise ValueError(f"expected V1 n_f=1 gate, got {fetch_gate.shape}")
    fetch_gate = jnp.squeeze(fetch_gate, axis=-2)
    local_q_gate = _sigmoid_with_compute_dtype(
        values["W_lq_gate"], biases[layer]["local_q"])
    local_k_gate = _sigmoid_with_compute_dtype(
        values["W_lk_gate"], biases[layer]["local_k"])

    def normalize_head_mix(raw):
      raw = jnp.asarray(raw, jnp.float32)
      normalized = raw * jax.lax.rsqrt(
          jnp.mean(jnp.square(raw), axis=-2, keepdims=True)
          + config.bam_read_key_epsilon)
      return jnp.asarray(normalized, config.dtype)

    sampled[layer] = {
        "write_gate": _sigmoid_with_compute_dtype(
            values["W_gw"], biases[layer]["write"])[:, ::stride],
        "fetch_gate": fetch_gate[:, ::stride],
        "local_q_gate": local_q_gate[:, ::stride],
        "local_k_gate": local_k_gate[:, ::stride],
        "local_q_head_mix": normalize_head_mix(
            values["W_lq_head_mix"])[:, ::stride],
        "local_k_head_mix": normalize_head_mix(
            values["W_lk_head_mix"])[:, ::stride],
    }
  mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.sum(mask, axis=-1)
  return {
      "total_loss": jnp.sum(xent * mask),
      "total_weights": jnp.sum(sequence_weights),
      "sequence_loss": (
          jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1)),
      "sequence_weights": sequence_weights,
  }, sampled, mask[:, ::stride]


def _iter_microbatches(batch, size):
  batch_size = int(batch["inputs"].shape[0])
  if batch_size % size:
    raise ValueError(f"batch {batch_size} is not divisible by microbatch {size}")
  for start in range(0, batch_size, size):
    yield {name: value[start:start + size] for name, value in batch.items()}


def _describe(values, gate=False) -> dict[str, float]:
  values = np.asarray(values, np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  if not values.size:
    return {"count": 0}
  report = {
      "count": int(values.size),
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "min": float(np.min(values)),
      "p01": float(np.percentile(values, 1)),
      "p10": float(np.percentile(values, 10)),
      "p25": float(np.percentile(values, 25)),
      "p50": float(np.percentile(values, 50)),
      "p75": float(np.percentile(values, 75)),
      "p90": float(np.percentile(values, 90)),
      "p99": float(np.percentile(values, 99)),
      "max": float(np.max(values)),
  }
  if gate:
    report.update({
        "fraction_lt_0.01": float(np.mean(values < 0.01)),
        "fraction_lt_0.05": float(np.mean(values < 0.05)),
        "fraction_gt_0.10": float(np.mean(values > 0.10)),
        "fraction_gt_0.50": float(np.mean(values > 0.50)),
        "fraction_gt_0.90": float(np.mean(values > 0.90)),
        "fraction_gt_0.99": float(np.mean(values > 0.99)),
    })
  return report


class _Collector:
  def __init__(self):
    self.tokens: dict[str, dict[int, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))
    self.sequence_means: dict[str, dict[int, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list))

  def add(self, channel: str, layer: int, values, mask):
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, bool)
    self.tokens[channel][layer].append(values[mask])
    per_sequence = []
    for sequence, valid in zip(values, mask, strict=True):
      per_sequence.append(np.mean(sequence[valid], axis=0))
    self.sequence_means[channel][layer].append(np.stack(per_sequence))

  def report(self, channel: str, gate: bool) -> dict[str, Any]:
    by_layer = {}
    all_values = []
    all_sequence_means = []
    layer_head_mean = []
    for layer in sorted(self.tokens[channel]):
      values = np.concatenate(self.tokens[channel][layer], axis=0)
      sequence_means = np.concatenate(self.sequence_means[channel][layer], axis=0)
      all_values.append(values.reshape(-1))
      all_sequence_means.append(sequence_means.reshape(-1))
      layer_report = {
          "distribution": _describe(values, gate=gate),
          "per_sequence_mean": _describe(sequence_means),
      }
      if values.ndim == 2:
        head_mean = np.mean(values, axis=0)
        layer_head_mean.append(head_mean)
        layer_report["head_mean"] = head_mean.tolist()
        layer_report["head_mean_distribution"] = _describe(head_mean)
      by_layer[f"layer_{layer:02d}"] = layer_report

    output = {
        "overall": _describe(np.concatenate(all_values), gate=gate),
        "per_sequence_mean": _describe(np.concatenate(all_sequence_means)),
        "by_layer": by_layer,
    }
    if layer_head_mean:
      matrix = np.stack(layer_head_mean)
      output["layer_head_mean"] = matrix.tolist()
      output["head_mean_across_layers"] = np.mean(matrix, axis=0).tolist()
      output["head_mean_distribution"] = _describe(np.mean(matrix, axis=0))
      output["layer_mean"] = np.mean(matrix, axis=1).tolist()
      output["layer_head_cv"] = (
          np.std(matrix, axis=1) / np.maximum(np.mean(np.abs(matrix), axis=1), _EPS)
      ).tolist()
    return output


def _add_microbatch(collector, sampled, mask):
  for layer, values in sampled.items():
    write = values["write_gate"]
    fetch = values["fetch_gate"]
    local_q = values["local_q_gate"]
    local_k = values["local_k_gate"]
    q_mix = values["local_q_head_mix"]
    k_mix = values["local_k_head_mix"]

    collector.add("write", layer, write, mask)
    collector.add("fetch_row", layer, fetch[..., 0], mask)
    collector.add("fetch_col", layer, fetch[..., 1], mask)
    collector.add("local_q_row", layer, local_q[..., 0], mask)
    collector.add("local_q_col", layer, local_q[..., 1], mask)
    collector.add("local_k_row", layer, local_k[..., 0], mask)
    collector.add("local_k_col", layer, local_k[..., 1], mask)

    for prefix, gate, mix in (
        ("local_q", local_q, q_mix),
        ("local_k", local_k, k_mix),
    ):
      for side, index in (("row", 0), ("col", 1)):
        signed_mix = mix[..., index]
        effective = np.abs(signed_mix) * gate[..., None, index]
        collector.add(f"{prefix}_{side}_head_mix", layer, signed_mix, mask)
        collector.add(f"{prefix}_{side}_effective", layer, effective, mask)


def run(config) -> None:
  if not config.bam_enabled or not config.only_eval:
    raise ValueError("use a BAM config with only_eval=True")
  if config.bam_local_qk_key_mode != "factorized" or config.bam_n_f != 1:
    raise ValueError("this runner expects V1 factorized LocalQK and n_f=1")

  batches = int(os.environ.get("BAM_GATE_DIAG_BATCHES", "4"))
  microbatch_size = int(os.environ.get("BAM_GATE_DIAG_MICROBATCH", "16"))
  stride = int(os.environ.get("BAM_GATE_DIAG_TOKEN_STRIDE", "8"))
  output_path = Path(os.environ.get(
      "BAM_GATE_DIAG_OUTPUT", "/tmp/bam_gate_diagnostics.json"))
  if batches <= 0 or microbatch_size <= 0 or stride <= 0:
    raise ValueError("batch, microbatch, and stride controls must be positive")
  output_path.parent.mkdir(parents=True, exist_ok=True)

  start = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  biases = _extract_biases(state.params, config.num_decoder_layers)
  setup_seconds = time.perf_counter() - start

  compiled = jax.jit(
      lambda params, b, rng: _forward(
          model, config, params, biases, b, rng, stride))
  collector = _Collector()
  sequence_hashes = []
  sequence_losses = []
  sequence_weights = []
  total_loss = 0.0
  total_weights = 0.0
  timings = []
  microbatch_index = 0

  for batch_index in range(batches):
    data_start = time.perf_counter()
    batch = next(eval_data_iterator)
    data_seconds = time.perf_counter() - data_start
    full_inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in full_inputs)

    for microbatch in _iter_microbatches(batch, microbatch_size):
      rng = jax.random.fold_in(init_rng, microbatch_index)
      forward_start = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics_device, sampled_device, mask_device = compiled(
            state.params, microbatch, rng)
      jax.block_until_ready((metrics_device, sampled_device, mask_device))
      forward_seconds = time.perf_counter() - forward_start

      transfer_start = time.perf_counter()
      metrics = jax.device_get(metrics_device)
      sampled = jax.device_get(sampled_device)
      mask = jax.device_get(mask_device)
      transfer_seconds = time.perf_counter() - transfer_start
      stats_start = time.perf_counter()
      _add_microbatch(collector, sampled, mask)
      stats_seconds = time.perf_counter() - stats_start

      total_loss += float(metrics["total_loss"])
      total_weights += float(metrics["total_weights"])
      sequence_losses.extend(np.asarray(metrics["sequence_loss"]).tolist())
      sequence_weights.extend(
          np.asarray(metrics["sequence_weights"]).astype(int).tolist())
      timings.append({
          "batch": batch_index,
          "microbatch": microbatch_index,
          "data_seconds": (
              data_seconds
              if microbatch_index
              % (int(batch["inputs"].shape[0]) // microbatch_size) == 0
              else 0.0),
          "forward_compile_execute_seconds": forward_seconds,
          "device_to_host_seconds": transfer_seconds,
          "host_collect_seconds": stats_seconds,
      })
      print(
          f"BAM_GATE_DIAG batch={batch_index} microbatch={microbatch_index} "
          f"loss={float(metrics['total_loss']) / max(float(metrics['total_weights']), 1):.6f} "
          f"forward_s={forward_seconds:.1f}", flush=True)
      microbatch_index += 1

  channels = {}
  for channel in sorted(collector.tokens):
    is_gate = channel == "write" or channel.startswith("fetch_") or (
        channel.startswith("local_")
        and not channel.endswith("_head_mix")
        and not channel.endswith("_effective"))
    channels[channel] = collector.report(channel, gate=is_gate)

  unique_hashes = len(set(sequence_hashes))
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_GATE_DIAG_CODE_COMMIT", ""),
          "num_batches": batches,
          "microbatch_size": microbatch_size,
          "sequence_count": len(sequence_hashes),
          "unique_sequence_count": unique_hashes,
          "cohort_hash": hashlib.sha256(
              "".join(sequence_hashes).encode()).hexdigest()[:16],
          "sequence_hashes": sequence_hashes,
          "sequence_length": int(config.max_target_length),
          "token_stride": stride,
          "sampled_positions_per_sequence": (
              int(config.max_target_length) + stride - 1) // stride,
          "data_shuffle_seed": int(config.data_shuffle_seed),
          "eval_shuffle_buffer_size": int(config.eval_shuffle_buffer_size),
          "device_count": jax.device_count(),
          "devices": [str(device) for device in jax.devices()],
          "setup_seconds": setup_seconds,
          "elapsed_seconds": time.perf_counter() - start,
          "architecture": {
              "fetch": "W_R gate for V1 CombinedRead over fetched M + local M",
              "local_qk": (
                  "sigmoid row/col gate is shared across heads; per-head effective "
                  "strength is gate * abs(RMS-normalized signed head mix)"),
          },
      },
      "eval": {
          "loss": total_loss / max(total_weights, 1.0),
          "sequence_loss": _describe(sequence_losses),
          "sequence_weights": _describe(sequence_weights),
      },
      "channels": channels,
      "timings": timings,
  }
  output_path.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"BAM_GATE_DIAG_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Fixed-cohort checkpoint diagnostics for fetched-read amplitude experiments.

This runner stays outside ``BamAttention``.  It restores one training checkpoint
read-only, captures the existing fetched-read projection/gate/output, and reports
the learned amplitude, sigmoid gate, effective post-RMS key scale, and fetched
readout energy relative to standard attention.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


_LAYER_RE = re.compile(r"layers_(\d+)")
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


def _stats(values: np.ndarray) -> dict[str, float]:
  values = np.asarray(values, np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  if values.size == 0:
    return {name: float("nan") for name in (
        "mean", "std", "min", "p05", "p50", "p95", "max")}
  return {
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "min": float(np.min(values)),
      "p05": float(np.percentile(values, 5)),
      "p50": float(np.percentile(values, 50)),
      "p95": float(np.percentile(values, 95)),
      "max": float(np.max(values)),
  }


def _stack_parameter(params, name: str, num_layers: int) -> jax.Array:
  matches = []
  for path, value in flatten_dict(params).items():
    if path[-1] == name:
      matches.append((path, value))
  if len(matches) == num_layers:
    matches.sort(key=lambda item: _layer_from_path(item[0]))
    return jnp.stack([value for _, value in matches])
  if len(matches) != 1:
    raise ValueError(f"expected one scanned or {num_layers} {name} leaves, found {len(matches)}")
  value = matches[0][1]
  layer_axes = [axis for axis, width in enumerate(value.shape) if width == num_layers]
  if len(layer_axes) != 1:
    raise ValueError(f"cannot identify layer axis for {name}: {value.shape}")
  return jnp.moveaxis(value, layer_axes[0], 0)


def _stack_module_capture(collections, module_name: str, num_layers: int) -> jax.Array:
  grouped = {}
  scanned = []
  for path, raw in flatten_dict(collections.get("intermediates", {})).items():
    if module_name not in path:
      continue
    value = _unwrap(raw)
    layer = _layer_from_path(path)
    if layer is None:
      scanned.append(value)
    else:
      grouped[layer] = value
  if grouped:
    if set(grouped) != set(range(num_layers)):
      raise ValueError(f"incomplete {module_name} captures: {sorted(grouped)}")
    return jnp.stack([grouped[layer] for layer in range(num_layers)])
  if len(scanned) != 1 or scanned[0].shape[0] != num_layers:
    shapes = [getattr(value, "shape", None) for value in scanned]
    raise ValueError(f"unexpected scanned {module_name} captures: {shapes}")
  return scanned[0]


def _stack_method_captures(collections, num_layers: int) -> dict[str, jax.Array]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  scanned = {}
  for path, raw in flatten_dict(collections.get("intermediates", {})).items():
    is_read = "_read_fetched_m" in path
    is_attention = "_query_chunk_op" in path or "_attention_block" in path
    if not (is_read or is_attention):
      continue
    value = _unwrap(raw)
    layer = _layer_from_path(path)
    target = scanned if layer is None else grouped[layer]
    if is_read:
      target["read"] = value
    else:
      if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"unexpected attention capture at {path}: {type(value)}")
      target["y_std"], target["mbar"] = value
  expected = {"read", "y_std", "mbar"}
  if set(scanned) == expected:
    for name, value in scanned.items():
      if value.shape[0] != num_layers:
        raise ValueError(f"scanned {name} has shape {value.shape}")
    return scanned
  if set(grouped) != set(range(num_layers)):
    raise ValueError(f"captured layers differ: {sorted(grouped)}")
  return {
      name: jnp.stack([grouped[layer][name] for layer in range(num_layers)])
      for name in expected
  }


def _capture(model, config, params, batch, rng):
  attention_method = (
      "_query_chunk_op" if config.query_chunk_size is not None
      else "_attention_block")

  def capture_intermediates(module, method_name):
    return (
        method_name in ("_read_fetched_m", attention_method)
        or (method_name == "__call__" and module.name in ("W_R", "W_R_gate")))

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
      capture_intermediates=capture_intermediates,
  )
  captures = _stack_method_captures(collections, config.num_decoder_layers)
  captures["raw_key"] = _stack_module_capture(
      collections, "W_R", config.num_decoder_layers)
  captures["gate_projection"] = _stack_module_capture(
      collections, "W_R_gate", config.num_decoder_layers)
  mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.maximum(jnp.sum(mask, axis=-1), 1)
  captures["sequence_loss"] = jnp.sum(xent * mask, axis=-1) / sequence_weights
  return captures


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
  mask = np.asarray(mask, bool)
  while mask.ndim < values.ndim:
    mask = mask[..., None]
  return np.asarray(values)[np.broadcast_to(mask, values.shape)]


def _side_summary(
    amplitude: np.ndarray,
    gate: np.ndarray,
    coefficient: np.ndarray,
    key_rms: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
  return {
      "amplitude_parameter": _stats(amplitude),
      "amplitude_negative_fraction": float(np.mean(amplitude < 0)),
      "sigmoid_gate": _stats(_masked(gate, mask)),
      "effective_coefficient": _stats(_masked(coefficient, mask)),
      "effective_coefficient_abs": _stats(np.abs(_masked(coefficient, mask))),
      "post_gate_key_rms": _stats(_masked(key_rms, mask)),
  }


def _readout_summary(
    read: np.ndarray, y_std: np.ndarray, mask: np.ndarray, bam_k: int, read_v_dim: int
) -> dict[str, Any]:
  mask = np.asarray(mask, np.float32)[None, ..., None, None]

  def squared(value):
    return np.sum(np.square(value.astype(np.float32)) * mask, axis=(1, 2, 3, 4))

  std_square = squared(y_std)
  read_square = squared(read)
  col_square = squared(read[..., :bam_k])
  row_square = squared(read[..., bam_k:bam_k + read_v_dim])
  dot = np.sum(read.astype(np.float32) * y_std.astype(np.float32) * mask,
               axis=(1, 2, 3, 4))
  return {
      "bam_to_std_frobenius": np.sqrt(read_square / np.maximum(std_square, _EPS)),
      "column_to_std_frobenius": np.sqrt(col_square / np.maximum(std_square, _EPS)),
      "row_to_std_frobenius": np.sqrt(row_square / np.maximum(std_square, _EPS)),
      "bam_std_cosine": dot / np.sqrt(np.maximum(read_square * std_square, _EPS)),
  }


def run(config):
  if not config.only_eval:
    raise ValueError("fetch_amplitude_diagnostics.py requires only_eval=True")
  output_path = Path(os.environ.get(
      "BAM_FETCHAMP_DIAG_OUTPUT", "/tmp/fetch_amplitude_diagnostics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  started = time.perf_counter()

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  batch = next(eval_iterator)
  inputs = np.asarray(jax.device_get(batch["inputs"]))
  mask = np.asarray(jax.device_get(batch["targets_segmentation"] != 0))

  capture_fn = jax.jit(lambda params, batch, rng: _capture(
      model, config, params, batch, rng))
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    captured = capture_fn(state.params, batch, init_rng)
  captured = jax.device_get(captured)

  gate_bias = np.asarray(jax.device_get(_stack_parameter(
      state.params, "W_R_gate_b0", config.num_decoder_layers)), np.float32)
  gate_projection = np.asarray(captured["gate_projection"], np.float32)
  raw_key = np.asarray(captured["raw_key"], np.float32)
  if gate_bias.shape[-2] == 1:
    gate_bias = np.squeeze(gate_bias, axis=-2)
  if gate_projection.shape[-2] == 1:
    gate_projection = np.squeeze(gate_projection, axis=-2)
  if raw_key.shape[-2] == 1:
    raw_key = np.squeeze(raw_key, axis=-2)

  logits = gate_projection + gate_bias[:, None, None]
  gate = 1.0 / (1.0 + np.exp(-logits))
  width_scale = math.sqrt(config.bam_abs_v_compression_dim or config.bam_v)
  if config.bam_fetched_read_amplitude_init is None:
    # The legacy path uses coefficient = 2 * sigmoid(gate).  Express it in
    # the explicit-amplitude convention so both parameterizations share the
    # same diagnostic formulas.
    amplitude = np.full_like(gate_bias, 2.0 * width_scale)
  else:
    amplitude = np.asarray(jax.device_get(_stack_parameter(
        state.params, "W_R_amplitude_scale", config.num_decoder_layers)), np.float32)
    if amplitude.shape[-2] == 1:
      amplitude = np.squeeze(amplitude, axis=-2)
  coefficient = amplitude[:, None, None] / width_scale * gate
  raw_row, raw_col = np.split(raw_key, [config.bam_k], axis=-1)
  read_epsilon = float(getattr(
      config, "bam_fetched_read_key_epsilon", config.bam_read_key_epsilon))
  row_direction_rms = np.sqrt(
      np.mean(np.square(raw_row), axis=-1)
      / (np.mean(np.square(raw_row), axis=-1) + read_epsilon))
  col_direction_rms = np.sqrt(
      np.mean(np.square(raw_col), axis=-1)
      / (np.mean(np.square(raw_col), axis=-1) + read_epsilon))
  row_key_rms = np.abs(coefficient[..., 0]) * row_direction_rms
  col_key_rms = np.abs(coefficient[..., 1]) * col_direction_rms

  readout = _readout_summary(
      np.asarray(captured["read"]), np.asarray(captured["y_std"]), mask,
      int(config.bam_k), int(config.bam_abs_v_compression_dim or config.bam_v))
  layers = {}
  for layer in range(config.num_decoder_layers):
    layer_mask = mask
    layers[f"layer_{layer:02d}"] = {
        "row": _side_summary(
            amplitude[layer, ..., 0], gate[layer, ..., 0],
            coefficient[layer, ..., 0], row_key_rms[layer], layer_mask),
        "column": _side_summary(
            amplitude[layer, ..., 1], gate[layer, ..., 1],
            coefficient[layer, ..., 1], col_key_rms[layer], layer_mask),
        "readout": {
            name: float(values[layer]) for name, values in readout.items()
        },
    }

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "exp_class": config.exp_class,
          "num_sequences": int(inputs.shape[0]),
          "sequence_hashes": [
              hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
              for sequence in inputs
          ],
          "bam_k": int(config.bam_k),
          "bam_v": int(config.bam_v),
          "read_v_dim": int(config.bam_abs_v_compression_dim or config.bam_v),
          "amplitude_init": config.bam_fetched_read_amplitude_init,
          "width_scale": width_scale,
          "fetched_read_key_epsilon": read_epsilon,
          "elapsed_seconds": time.perf_counter() - started,
      },
      "sequence_loss": np.asarray(captured["sequence_loss"]).tolist(),
      "layers": layers,
  }
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_FETCHAMP_DIAG_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

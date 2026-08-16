"""Compare key-codebook and value-axis compression on a frozen BAM V1 checkpoint.

The runner uses four shuffled Pile eval batches.  It fits all layerwise subspaces on
the first two batches, then reports paired losses on the held-out two batches.  The
checkpoint is restored once and never mutated or saved.

``BAM_AXIS_DIAG_SCHEDULES_JSON`` may name a JSON file mapping variant names to one
compression width per layer.  This reuses the same fitted per-layer bases to test
heterogeneous, equal-cache schedules without changing the production BAM path.
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


_LAYER_RE = re.compile(r"layers_(\d+)")
_PROJECTOR_NAMES = (
    "diag_mc_left", "diag_mc_right", "diag_mr_left", "diag_mr_right",
    "diag_yu", "diag_yv",
)
_EPS = 1.0e-12


class _ConfigOverlay:
  def __init__(self, base, **overrides):
    self._base = base
    self._overrides = overrides

  def __getattr__(self, name):
    if name in self._overrides:
      return self._overrides[name]
    return getattr(self._base, name)

  def get_keys(self):
    return {**self._base.get_keys(), **self._overrides}


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


def _group_raw(collections) -> dict[int, dict[str, jax.Array]]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  for path, value in flatten_dict(collections.get("bam_raw", {})).items():
    layer = _layer_from_path(path)
    if layer is not None:
      grouped[layer][path[-1]] = _unwrap(value)
  return dict(sorted(grouped.items()))


def _loss_metrics(xent, correct, batch):
  mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.sum(mask, axis=-1)
  total_weights = jnp.sum(sequence_weights)
  return {
      "total_loss": jnp.sum(xent * mask),
      "total_weights": total_weights,
      "accuracy_numerator": correct,
      "sequence_loss": (
          jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1)),
      "sequence_weights": sequence_weights,
  }


def _capture_forward(model, params, batch, rng, bam_k):
  dropout_rng, params_rng = jax.random.split(rng)
  (xent, correct, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng},
      mutable=["bam_raw"],
  )
  grouped = _group_raw(collections)
  covariances = {}
  for layer, raw in grouped.items():
    matrix = jnp.asarray(raw["Mbar"], jnp.float32)  # [b,f,t,k,v]
    key = jnp.asarray(raw["read_key_W_R_post_gate"], jnp.float32)
    row_key, col_key = jnp.split(key, [bam_k], axis=-1)
    y_u = jnp.einsum("bftkv,btnfv->btnk", matrix, col_key)
    y_v = jnp.einsum("bftkv,btnfk->btnv", matrix, row_key)
    matrix_flat = matrix.reshape((-1, matrix.shape[-2], matrix.shape[-1]))
    row_flat = row_key.reshape((-1, row_key.shape[-1]))
    col_flat = col_key.reshape((-1, col_key.shape[-1]))
    row_by_head = jnp.moveaxis(row_key, -3, 0).reshape(
        (row_key.shape[-3], -1, row_key.shape[-1]))
    col_by_head = jnp.moveaxis(col_key, -3, 0).reshape(
        (col_key.shape[-3], -1, col_key.shape[-1]))
    covariances[layer] = {
        "weight": jnp.asarray(matrix_flat.shape[0], jnp.float32),
        "m_left": jnp.einsum("akv,alv->kl", matrix_flat, matrix_flat),
        "m_right": jnp.einsum("akv,akw->vw", matrix_flat, matrix_flat),
        "row_key": row_flat.T @ row_flat,
        "col_key": col_flat.T @ col_flat,
        "row_key_per_head": jnp.einsum(
            "nak,nal->nkl", row_by_head, row_by_head),
        "col_key_per_head": jnp.einsum(
            "nak,nal->nkl", col_by_head, col_by_head),
        "y_u": jnp.einsum("btnk,btnj->nkj", y_u, y_u),
        "y_v": jnp.einsum("btnv,btnw->nvw", y_v, y_v),
    }
  return _loss_metrics(xent, correct, batch), covariances


def _plain_forward(model, params, batch, rng):
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
  return _loss_metrics(xent, correct, batch)


def _iter_microbatches(batch, size):
  batch_size = int(batch["inputs"].shape[0])
  if batch_size % size:
    raise ValueError(f"batch {batch_size} is not divisible by microbatch {size}")
  for start in range(0, batch_size, size):
    yield {name: value[start:start + size] for name, value in batch.items()}


def _add_covariances(target, source):
  for layer, values in source.items():
    for name, value in values.items():
      array = np.asarray(value, np.float64)
      if name not in target[layer]:
        target[layer][name] = array
      else:
        target[layer][name] += array


def _eigendecomposition(covariance):
  covariance = np.asarray(covariance, np.float64)
  covariance = 0.5 * (covariance + covariance.T)
  values, vectors = np.linalg.eigh(covariance)
  order = np.argsort(values)[::-1]
  return np.maximum(values[order], 0.0), vectors[:, order]


def _spectrum_report(covariance, ranks):
  values, _ = _eigendecomposition(covariance)
  total = max(float(np.sum(values)), _EPS)
  probabilities = values / total
  positive = probabilities[probabilities > 0]
  return {
      "eigenvalues": values.tolist(),
      "energy": {
          str(rank): float(np.sum(probabilities[:rank])) for rank in ranks
      },
      "stable_rank": float(total / max(float(values[0]), _EPS)),
      "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
      "numerical_rank_rtol_1e-6": int(np.sum(values > values[0] * 1.0e-12)),
  }


def _per_head_spectrum_report(covariances, ranks):
  reports = [_spectrum_report(covariance, ranks) for covariance in covariances]
  optimal_energy = {}
  for rank in ranks:
    values = [report["energy"][str(rank)] for report in reports]
    optimal_energy[str(rank)] = {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }
  return {
      "heads": reports,
      "optimal_energy": optimal_energy,
  }


def _projector(basis, rank):
  width = basis.shape[0]
  if rank >= width:
    return np.eye(width, dtype=np.float32)
  selected = basis[:, :rank]
  return np.asarray(selected @ selected.T, np.float32)


def _decoder_map(basis, output_covariance, rank):
  """Shared encoder plus a least-squares decoder for one head."""
  width = basis.shape[0]
  if rank >= width:
    return np.eye(width, dtype=np.float32)
  encoder = basis[:, :rank]
  covariance = np.asarray(output_covariance, np.float64)
  gram = encoder.T @ covariance @ encoder
  decoder = np.linalg.pinv(gram, rcond=1.0e-8) @ encoder.T @ covariance
  return np.asarray(encoder @ decoder, np.float32)


def _wr_covariances(params, bam_k):
  output = {}
  paths = {}
  for path, value in flatten_dict(params).items():
    if len(path) < 2 or path[-2:] != ("W_R", "kernel"):
      continue
    layer = _layer_from_path(path)
    if layer is None:
      continue
    kernel = np.asarray(jax.device_get(value), np.float64)
    row = kernel[..., :bam_k].reshape((-1, bam_k))
    col = kernel[..., bam_k:].reshape((-1, kernel.shape[-1] - bam_k))
    row_structured = kernel[..., :bam_k]
    col_structured = kernel[..., bam_k:]
    row_by_head = np.moveaxis(row_structured, -3, 0).reshape(
        (row_structured.shape[-3], -1, bam_k))
    col_by_head = np.moveaxis(col_structured, -3, 0).reshape(
        (col_structured.shape[-3], -1, col_structured.shape[-1]))
    output[layer] = {
        "row": row.T @ row,
        "col": col.T @ col,
        "row_per_head": np.einsum("nak,nal->nkl", row_by_head, row_by_head),
        "col_per_head": np.einsum("nak,nal->nkl", col_by_head, col_by_head),
    }
    paths[layer] = path
  return output, paths


def _identity_controls(num_heads, bam_k, bam_v):
  return {
      "diag_mc_left": np.eye(bam_k, dtype=np.float32),
      "diag_mc_right": np.eye(bam_v, dtype=np.float32),
      "diag_mr_left": np.eye(bam_k, dtype=np.float32),
      "diag_mr_right": np.eye(bam_v, dtype=np.float32),
      "diag_yu": np.broadcast_to(
          np.eye(bam_k, dtype=np.float32), (num_heads, bam_k, bam_k)).copy(),
      "diag_yv": np.broadcast_to(
          np.eye(bam_v, dtype=np.float32), (num_heads, bam_v, bam_v)).copy(),
  }


def _variant_controls(mode, rank, covariance, num_heads, bam_k, bam_v):
  controls = _identity_controls(num_heads, bam_k, bam_v)
  _, row_key_basis = _eigendecomposition(covariance["row_key"])
  _, col_key_basis = _eigendecomposition(covariance["col_key"])
  _, right_basis = _eigendecomposition(covariance["m_right"])
  _, output_u_basis = _eigendecomposition(np.sum(covariance["y_u"], axis=0))
  _, output_v_basis = _eigendecomposition(np.sum(covariance["y_v"], axis=0))
  if mode == "key_codebook":
    controls["diag_mr_left"] = _projector(row_key_basis, rank)
    controls["diag_mc_right"] = _projector(col_key_basis, rank)
  elif mode == "value_v_tied":
    right_projector = _projector(right_basis, rank)
    controls["diag_mc_right"] = right_projector
    controls["diag_yv"] = np.broadcast_to(
        right_projector, (num_heads, bam_v, bam_v)).copy()
  elif mode == "value_v_head_decoder":
    controls["diag_mc_right"] = _projector(right_basis, rank)
    controls["diag_yv"] = np.stack([
        _decoder_map(right_basis, covariance["y_v"][head], rank)
        for head in range(num_heads)
    ])
  elif mode in ("value_v_output_head_decoder", "compress_fixed_v"):
    controls["diag_mc_right"] = _projector(output_v_basis, rank)
    controls["diag_yv"] = np.stack([
        _decoder_map(output_v_basis, covariance["y_v"][head], rank)
        for head in range(num_heads)
    ])
  elif mode == "compress_fixed_k":
    controls["diag_mr_left"] = _projector(output_u_basis, rank)
    controls["diag_yu"] = np.stack([
        _decoder_map(output_u_basis, covariance["y_u"][head], rank)
        for head in range(num_heads)
    ])
  elif mode == "compress_local_v_keep_local_k":
    controls["diag_yu"] = np.stack([
        _decoder_map(output_u_basis, covariance["y_u"][head], rank)
        for head in range(num_heads)
    ])
    controls["diag_yv"] = np.stack([
        _decoder_map(output_v_basis, covariance["y_v"][head], rank)
        for head in range(num_heads)
    ])
  elif mode != "identity":
    raise ValueError(f"Unknown mode {mode}")
  return controls


def _insert_controls(params, wr_paths, controls_by_layer):
  flat = dict(flatten_dict(params))
  for layer, wr_path in wr_paths.items():
    parent = wr_path[:-2]
    controls = controls_by_layer[layer]
    for name in _PROJECTOR_NAMES:
      flat[parent + (name,)] = jax.device_put(jnp.asarray(controls[name]))
  updated = unflatten_dict(flat)
  return freeze(updated) if isinstance(params, FrozenDict) else updated


def _merge_metrics(parts):
  total_loss = sum(float(part["total_loss"]) for part in parts)
  total_weights = sum(float(part["total_weights"]) for part in parts)
  sequence_loss = np.concatenate([
      np.asarray(part["sequence_loss"], np.float64) for part in parts])
  sequence_weights = np.concatenate([
      np.asarray(part["sequence_weights"], np.int64) for part in parts])
  return {
      "loss": total_loss / max(total_weights, 1.0),
      "total_loss": total_loss,
      "total_weights": int(total_weights),
      "sequence_loss": sequence_loss,
      "sequence_weights": sequence_weights,
  }


def _delta_report(candidate, baseline):
  delta = candidate["sequence_loss"] - baseline["sequence_loss"]
  return {
      "loss": candidate["loss"],
      "delta_loss": candidate["loss"] - baseline["loss"],
      "sequence_delta": {
          "mean": float(np.mean(delta)),
          "std": float(np.std(delta)),
          "min": float(np.min(delta)),
          "p25": float(np.percentile(delta, 25)),
          "median": float(np.median(delta)),
          "p75": float(np.percentile(delta, 75)),
          "max": float(np.max(delta)),
          "improved_fraction": float(np.mean(delta < 0)),
      },
      "sequence_loss": candidate["sequence_loss"].tolist(),
  }


def _subset(parts, logical_batch_indices, microbatches_per_batch):
  selected = []
  for batch_index in logical_batch_indices:
    start = batch_index * microbatches_per_batch
    selected.extend(parts[start:start + microbatches_per_batch])
  return _merge_metrics(selected)


def _rank_aggregate(per_layer, ranks):
  output = {}
  names = next(iter(per_layer.values())).keys()
  for name in names:
    output[name] = {}
    for rank in ranks:
      values = [layer[name]["energy"][str(rank)] for layer in per_layer.values()]
      output[name][str(rank)] = {
          "min": float(np.min(values)),
          "median": float(np.median(values)),
          "max": float(np.max(values)),
      }
  return output


def _load_schedules(path, num_layers, max_rank):
  if not path:
    return {}
  schedules = json.loads(Path(path).read_text())
  if not isinstance(schedules, dict):
    raise ValueError("BAM_AXIS_DIAG_SCHEDULES_JSON must contain a JSON object")
  output = {}
  for name, widths in schedules.items():
    if not isinstance(name, str) or not name:
      raise ValueError(f"invalid schedule name: {name!r}")
    if len(widths) != num_layers:
      raise ValueError(
          f"schedule {name!r} has {len(widths)} layers, expected {num_layers}")
    widths = tuple(int(width) for width in widths)
    if any(width < 0 or width > max_rank for width in widths):
      raise ValueError(
          f"schedule {name!r} widths must be in [0, {max_rank}]: {widths}")
    output[name] = widths
  return output


def _select_budget_schedule(layer_costs, budget):
  """Minimize summed single-layer loss deltas under a cache-width cap."""
  states = {0: (0.0, [])}
  for layer in sorted(layer_costs):
    next_states = {}
    for used, (cost, schedule) in states.items():
      for width, layer_cost in layer_costs[layer].items():
        candidate_used = used + width
        if candidate_used > budget:
          continue
        candidate = (cost + layer_cost, schedule + [width])
        if candidate_used not in next_states or candidate[0] < next_states[candidate_used][0]:
          next_states[candidate_used] = candidate
    states = next_states
  if not states:
    raise ValueError(f"no feasible schedule for cache-width budget {budget}")
  used, (cost, schedule) = min(
      states.items(), key=lambda item: (item[1][0], -item[0]))
  return tuple(schedule), used, cost


def run(config) -> None:
  started = time.perf_counter()
  if not config.only_eval or not config.bam_diagnostics:
    raise ValueError("Use the V1 diagnostics config with only_eval=True")
  num_batches = int(os.environ.get("BAM_AXIS_DIAG_BATCHES", "4"))
  calibration_batches = int(os.environ.get("BAM_AXIS_DIAG_CALIBRATION_BATCHES", "2"))
  microbatch_size = int(os.environ.get("BAM_AXIS_DIAG_MICROBATCH", "16"))
  ranks = tuple(int(x) for x in os.environ.get(
      "BAM_AXIS_DIAG_RANKS", "2,4,8,12,16,32").split(","))
  modes = tuple(x for x in os.environ.get(
      "BAM_AXIS_DIAG_MODES",
      "identity,key_codebook,value_v_tied,value_v_head_decoder,value_v_output_head_decoder",
  ).split(",") if x)
  layerwise = os.environ.get("BAM_AXIS_DIAG_LAYERWISE", "0").lower() in (
      "1", "true", "yes")
  schedules = _load_schedules(
      os.environ.get("BAM_AXIS_DIAG_SCHEDULES_JSON", ""),
      int(config.num_decoder_layers), int(config.bam_v))
  auto_budgets = tuple(int(value) for value in os.environ.get(
      "BAM_AXIS_DIAG_AUTO_BUDGETS", "").split(",") if value)
  auto_widths = tuple(sorted(set(int(value) for value in os.environ.get(
      "BAM_AXIS_DIAG_AUTO_WIDTHS", "0,4,8,12,16,24,32").split(","))))
  if auto_budgets:
    if any(width < 0 or width > int(config.bam_v) for width in auto_widths):
      raise ValueError(f"auto widths must lie in [0, {config.bam_v}]")
    if int(config.bam_v) not in auto_widths:
      raise ValueError("auto widths must include the uncompressed width")
    if num_batches - calibration_batches < 2:
      raise ValueError("auto schedule selection needs one selection and one validation batch")
  output_path = Path(os.environ.get(
      "BAM_AXIS_DIAG_OUTPUT", "/tmp/bam_codebook_axis_diagnostics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if not 0 < calibration_batches < num_batches:
    raise ValueError("calibration batches must be a strict subset")

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  setup_seconds = time.perf_counter() - started

  logical_batches = []
  sequence_hashes = []
  for _ in range(num_batches):
    batch = next(eval_data_iterator)
    jax.block_until_ready(batch)
    host_inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.append([
        hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in host_inputs])
    logical_batches.append(batch)
  microbatches = [
      microbatch
      for batch in logical_batches
      for microbatch in _iter_microbatches(batch, microbatch_size)
  ]
  microbatches_per_batch = int(logical_batches[0]["inputs"].shape[0]) // microbatch_size

  capture = jax.jit(lambda params, batch, rng: _capture_forward(
      model, params, batch, rng, int(config.bam_k)))
  calibration_covariances = defaultdict(dict)
  all_covariances = defaultdict(dict)
  baseline_parts = []
  capture_started = time.perf_counter()
  for index, batch in enumerate(microbatches):
    rng = jax.random.fold_in(init_rng, index)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      metrics_device, covariances_device = capture(state.params, batch, rng)
    metrics_device, covariances_device = jax.block_until_ready(
        (metrics_device, covariances_device))
    metrics = jax.device_get(metrics_device)
    covariances = jax.device_get(covariances_device)
    baseline_parts.append(metrics)
    _add_covariances(all_covariances, covariances)
    if index < calibration_batches * microbatches_per_batch:
      _add_covariances(calibration_covariances, covariances)
    print(f"BAM_AXIS_CAPTURE microbatch={index + 1}/{len(microbatches)}", flush=True)
  capture_seconds = time.perf_counter() - capture_started

  wr_covariances, wr_paths = _wr_covariances(state.params, int(config.bam_k))
  if len(wr_paths) != int(config.num_decoder_layers):
    raise RuntimeError(f"Expected {config.num_decoder_layers} W_R paths, got {len(wr_paths)}")
  per_layer_rank = {}
  per_head_rank = {}
  layer_bases = {}
  for layer in sorted(calibration_covariances):
    covariance = calibration_covariances[layer]
    per_layer_rank[f"layer_{layer:02d}"] = {
        "weight_row": _spectrum_report(wr_covariances[layer]["row"], ranks),
        "weight_col": _spectrum_report(wr_covariances[layer]["col"], ranks),
        "runtime_row": _spectrum_report(covariance["row_key"], ranks),
        "runtime_col": _spectrum_report(covariance["col_key"], ranks),
        "M_left": _spectrum_report(covariance["m_left"], ranks),
        "M_right": _spectrum_report(covariance["m_right"], ranks),
        "read_yu": _spectrum_report(np.sum(covariance["y_u"], axis=0), ranks),
        "read_yv": _spectrum_report(np.sum(covariance["y_v"], axis=0), ranks),
    }
    per_head_rank[f"layer_{layer:02d}"] = {
        "weight_row": _per_head_spectrum_report(
            wr_covariances[layer]["row_per_head"], ranks),
        "weight_col": _per_head_spectrum_report(
            wr_covariances[layer]["col_per_head"], ranks),
        "runtime_row": _per_head_spectrum_report(
            covariance["row_key_per_head"], ranks),
        "runtime_col": _per_head_spectrum_report(
            covariance["col_key_per_head"], ranks),
        "read_yu": _per_head_spectrum_report(covariance["y_u"], ranks),
        "read_yv": _per_head_spectrum_report(covariance["y_v"], ranks),
    }
    layer_bases[layer] = covariance

  variant_config = _ConfigOverlay(
      config, bam_diagnostics=False, bam_diagnostic_read_projection=True)
  variant_model = train.Transformer(variant_config, mesh, quant=model.quant)
  variant_forward = jax.jit(
      lambda params, batch, rng: _plain_forward(variant_model, params, batch, rng))
  baseline_all = _merge_metrics(baseline_parts)
  baseline_calibration = _subset(
      baseline_parts, range(calibration_batches), microbatches_per_batch)
  baseline_heldout = _subset(
      baseline_parts, range(calibration_batches, num_batches), microbatches_per_batch)
  baseline_selection = _subset(
      baseline_parts, (calibration_batches,), microbatches_per_batch)
  baseline_validation = _subset(
      baseline_parts, range(calibration_batches + 1, num_batches),
      microbatches_per_batch)

  variants = {}
  layerwise_variants = {}
  scheduled_variants = {}
  auto_schedule_selection = {}
  auto_rate_distortion = {}
  evaluation_started = time.perf_counter()
  if layerwise:
    identity_controls = {
        layer: _identity_controls(
            int(config.num_query_heads), int(config.bam_k), int(config.bam_v))
        for layer in layer_bases
    }
    heldout_start = calibration_batches * microbatches_per_batch
    for layer in sorted(layer_bases):
      if layer == 0:
        continue
      for mode in modes:
        if mode == "identity":
          continue
        for rank in ranks:
          controls = dict(identity_controls)
          controls[layer] = _variant_controls(
              mode, rank, layer_bases[layer], int(config.num_query_heads),
              int(config.bam_k), int(config.bam_v))
          variant_params = _insert_controls(state.params, wr_paths, controls)
          parts = []
          variant_started = time.perf_counter()
          for index in range(heldout_start, len(microbatches)):
            rng = jax.random.fold_in(init_rng, index)
            batch = microbatches[index]
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
              metrics_device = variant_forward(variant_params, batch, rng)
            parts.append(jax.device_get(jax.block_until_ready(metrics_device)))
          name = f"layer_{layer:02d}_{mode}_C{rank}"
          candidate_heldout = _merge_metrics(parts)
          layerwise_variants[name] = {
              "layer": layer,
              "mode": mode,
              "C": rank,
              "heldout": _delta_report(candidate_heldout, baseline_heldout),
              "seconds": time.perf_counter() - variant_started,
          }
          print(
              f"BAM_AXIS_LAYERWISE {name} heldout_loss={candidate_heldout['loss']:.8f} "
              f"delta={candidate_heldout['loss'] - baseline_heldout['loss']:+.8f}",
              flush=True)
  else:
    for mode in modes:
      mode_ranks = (32,) if mode == "identity" else ranks
      for rank in mode_ranks:
        controls = {
            layer: _variant_controls(
                mode, rank, covariance, int(config.num_query_heads),
                int(config.bam_k), int(config.bam_v))
            for layer, covariance in layer_bases.items()
        }
        variant_params = _insert_controls(state.params, wr_paths, controls)
        parts = []
        variant_started = time.perf_counter()
        for index, batch in enumerate(microbatches):
          rng = jax.random.fold_in(init_rng, index)
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            metrics_device = variant_forward(variant_params, batch, rng)
          parts.append(jax.device_get(jax.block_until_ready(metrics_device)))
        name = f"{mode}_C{rank}"
        candidate_all = _merge_metrics(parts)
        candidate_calibration = _subset(
            parts, range(calibration_batches), microbatches_per_batch)
        candidate_heldout = _subset(
            parts, range(calibration_batches, num_batches), microbatches_per_batch)
        variants[name] = {
            "mode": mode,
            "C": rank,
            "all": _delta_report(candidate_all, baseline_all),
            "calibration": _delta_report(candidate_calibration, baseline_calibration),
            "heldout": _delta_report(candidate_heldout, baseline_heldout),
            "seconds": time.perf_counter() - variant_started,
        }
        print(
            f"BAM_AXIS_VARIANT {name} heldout_loss={candidate_heldout['loss']:.8f} "
            f"delta={candidate_heldout['loss'] - baseline_heldout['loss']:+.8f}",
            flush=True)
    if auto_budgets:
      identity_controls = {
          layer: _identity_controls(
              int(config.num_query_heads), int(config.bam_k), int(config.bam_v))
          for layer in layer_bases
      }
      selection_start = calibration_batches * microbatches_per_batch
      selection_stop = selection_start + microbatches_per_batch
      layer_costs = {}
      for layer in sorted(layer_bases):
        layer_costs[layer] = {0: 0.0} if layer == 0 else {}
        if layer == 0:
          continue
        for width in auto_widths:
          if width == int(config.bam_v):
            layer_costs[layer][width] = 0.0
            continue
          controls = dict(identity_controls)
          controls[layer] = _variant_controls(
              "compress_fixed_v", width, layer_bases[layer],
              int(config.num_query_heads), int(config.bam_k), int(config.bam_v))
          variant_params = _insert_controls(state.params, wr_paths, controls)
          parts = []
          for index in range(selection_start, selection_stop):
            rng = jax.random.fold_in(init_rng, index)
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
              metrics_device = variant_forward(variant_params, microbatches[index], rng)
            parts.append(jax.device_get(jax.block_until_ready(metrics_device)))
          candidate = _merge_metrics(parts)
          layer_costs[layer][width] = candidate["loss"] - baseline_selection["loss"]
          print(
              f"BAM_AXIS_RATE_DISTORTION layer={layer:02d} C={width} "
              f"selection_delta={layer_costs[layer][width]:+.8f}", flush=True)
      for budget in auto_budgets:
        widths, used, predicted_delta = _select_budget_schedule(
            layer_costs, budget)
        name = f"auto_dp_c{budget}"
        schedules[name] = widths
        auto_schedule_selection[name] = {
            "budget": budget,
            "used_width": used,
            "predicted_additive_selection_delta": predicted_delta,
            "widths": list(widths),
        }
        print(
            f"BAM_AXIS_AUTO_SCHEDULE {name} used={used} "
            f"predicted_delta={predicted_delta:+.8f} widths={widths}", flush=True)
      auto_rate_distortion = {
          f"layer_{layer:02d}": {
              str(width): cost for width, cost in sorted(costs.items())}
          for layer, costs in sorted(layer_costs.items())
      }
    for name, widths in schedules.items():
      controls = {
          layer: _variant_controls(
              "compress_fixed_v", widths[layer], covariance,
              int(config.num_query_heads), int(config.bam_k), int(config.bam_v))
          for layer, covariance in layer_bases.items()
      }
      variant_params = _insert_controls(state.params, wr_paths, controls)
      parts = []
      variant_started = time.perf_counter()
      heldout_start = calibration_batches * microbatches_per_batch
      for index in range(heldout_start, len(microbatches)):
        rng = jax.random.fold_in(init_rng, index)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          metrics_device = variant_forward(variant_params, microbatches[index], rng)
        parts.append(jax.device_get(jax.block_until_ready(metrics_device)))
      candidate_heldout = _merge_metrics(parts)
      candidate_selection = _merge_metrics(parts[:microbatches_per_batch])
      candidate_validation = _merge_metrics(parts[microbatches_per_batch:])
      scheduled_variants[name] = {
          "mode": "compress_fixed_v",
          "widths": list(widths),
          "total_width": int(sum(widths)),
          "heldout": _delta_report(candidate_heldout, baseline_heldout),
          "selection": _delta_report(candidate_selection, baseline_selection),
          "validation": _delta_report(candidate_validation, baseline_validation),
          "seconds": time.perf_counter() - variant_started,
      }
      print(
          f"BAM_AXIS_SCHEDULE {name} total_width={sum(widths)} "
          f"selection_delta={candidate_selection['loss'] - baseline_selection['loss']:+.8f} "
          f"validation_delta={candidate_validation['loss'] - baseline_validation['loss']:+.8f}",
          flush=True)
  evaluation_seconds = time.perf_counter() - evaluation_started

  rank_for_aggregate = {
      layer: values for layer, values in per_layer_rank.items()
      if layer != "layer_00"
  }
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "num_batches": num_batches,
          "calibration_batches": calibration_batches,
          "selection_batches": 1 if auto_budgets else 0,
          "validation_batches": (
              num_batches - calibration_batches - 1 if auto_budgets else 0),
          "heldout_batches": num_batches - calibration_batches,
          "batch_size": int(logical_batches[0]["inputs"].shape[0]),
          "sequence_length": int(logical_batches[0]["inputs"].shape[1]),
          "microbatch_size": microbatch_size,
          "ranks": list(ranks),
          "modes": list(modes),
          "schedules": {name: list(widths) for name, widths in schedules.items()},
          "auto_budgets": list(auto_budgets),
          "auto_widths": list(auto_widths),
          "layerwise": layerwise,
          "data_shuffle_seed": int(config.data_shuffle_seed),
          "eval_shuffle_buffer_size": int(config.eval_shuffle_buffer_size),
          "sequence_hashes": sequence_hashes,
          "code_commit": os.environ.get("BAM_AXIS_DIAG_CODE_COMMIT", "unknown"),
          "overlay_sha256": os.environ.get("BAM_AXIS_DIAG_OVERLAY_SHA256", "unknown"),
          "setup_seconds": setup_seconds,
          "capture_seconds": capture_seconds,
          "evaluation_seconds": evaluation_seconds,
          "total_seconds": time.perf_counter() - started,
          "checkpoint_mutated": False,
          "device_count": jax.device_count(),
          "devices": [str(device) for device in jax.devices()],
      },
      "baseline": {
          "all": {"loss": baseline_all["loss"]},
          "calibration": {"loss": baseline_calibration["loss"]},
          "heldout": {"loss": baseline_heldout["loss"]},
      },
      "rank": {
          "per_layer": per_layer_rank,
          "per_head": per_head_rank,
          "aggregate_layers_01_23": _rank_aggregate(rank_for_aggregate, ranks),
      },
      "variants": variants,
      "layerwise_variants": layerwise_variants,
      "auto_schedule_selection": auto_schedule_selection,
      "auto_rate_distortion": auto_rate_distortion,
      "scheduled_variants": scheduled_variants,
  }
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_AXIS_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

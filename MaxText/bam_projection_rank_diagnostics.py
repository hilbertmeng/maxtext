"""Analyze Direct checkpoint P_loc and W_R rank, then run paired P_loc ablations.

The checkpoint is restored once and never mutated or saved.  Four shuffled Pile
eval batches are split 2/2: the first half fits activation subspaces and affine
biases; the second half is held out for reconstruction and loss evaluation.
No diagnostic computation is added to BamAttention.
"""

from __future__ import annotations

import hashlib
import json
import math
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
_KEY_STAGES = ("pre_rms", "post_rms_pre_gate", "post_gate")
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


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for component in path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _unwrap(value: Any) -> Any:
  while isinstance(value, (tuple, list)) and len(value) == 1:
    value = value[0]
  return value


def _capture_ploc(module, method_name: str) -> bool:
  return method_name == "__call__" and module.name == "P_loc"


def _group_capture(collections) -> dict[int, dict[str, jax.Array]]:
  grouped = defaultdict(dict)
  for path, value in flatten_dict(collections.get("bam_raw", {})).items():
    layer = _layer_from_path(path)
    if layer is not None and path[-1].startswith("read_key_W_R_"):
      grouped[layer][path[-1]] = _unwrap(value)
  for path, value in flatten_dict(collections.get("intermediates", {})).items():
    layer = _layer_from_path(path)
    if layer is not None and "P_loc" in path:
      grouped[layer]["P_loc"] = _unwrap(value)
  return dict(sorted(grouped.items()))


def _masked_moments(values: jax.Array, valid: jax.Array) -> dict[str, jax.Array]:
  values = jnp.asarray(values, jnp.float32)
  valid = jnp.asarray(valid, jnp.float32)
  masked = values * valid[..., None]
  return {
      "count": jnp.sum(valid),
      "sum": jnp.sum(masked, axis=(0, 1)),
      "gram": jnp.einsum("btf,btg->fg", masked, values),
  }


def _capture_forward(model, params, batch, rng, stride: int, bam_k: int):
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
      mutable=["bam_raw", "intermediates"],
      capture_intermediates=_capture_ploc,
  )
  target_mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.sum(target_mask, axis=-1)
  grouped = _group_capture(collections)
  sampled_valid = target_mask[:, ::stride]
  moments = {}
  ploc_samples = {}
  for layer, values in grouped.items():
    p_loc = values["P_loc"][:, ::stride].reshape(
        values["P_loc"].shape[0], -1, math.prod(values["P_loc"].shape[2:]))
    ploc_samples[layer] = p_loc
    moments[layer] = {}
    for stage in _KEY_STAGES:
      keys = values[f"read_key_W_R_{stage}"][:, ::stride]
      row = keys[..., :bam_k].reshape(keys.shape[0], keys.shape[1], -1)
      col = keys[..., bam_k:].reshape(keys.shape[0], keys.shape[1], -1)
      moments[layer][f"W_R_{stage}_row"] = _masked_moments(row, sampled_valid)
      moments[layer][f"W_R_{stage}_col"] = _masked_moments(col, sampled_valid)
  return {
      "total_loss": jnp.sum(xent * target_mask),
      "total_weights": jnp.sum(target_mask),
      "accuracy_numerator": correct,
      "sequence_loss": (
          jnp.sum(xent * target_mask, axis=-1) / jnp.maximum(sequence_weights, 1)),
      "sequence_weights": sequence_weights,
  }, moments, ploc_samples, sampled_valid


def _loss_forward(model, params, batch, rng):
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
  sequence_weights = jnp.sum(mask, axis=-1)
  return {
      "total_loss": jnp.sum(xent * mask),
      "total_weights": jnp.sum(mask),
      "accuracy_numerator": correct,
      "sequence_loss": jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1),
      "sequence_weights": sequence_weights,
  }


def _rank_for_fraction(energy: np.ndarray, fraction: float) -> int:
  if not np.any(energy > 0):
    return 0
  return int(np.searchsorted(np.cumsum(energy) / np.sum(energy), fraction) + 1)


def _spectrum_summary(energy: np.ndarray, ranks: tuple[int, ...]) -> dict[str, Any]:
  energy = np.maximum(np.asarray(energy, np.float64), 0)
  total = float(np.sum(energy))
  fractions = energy / max(total, _EPS)
  nonzero = fractions[fractions > 0]
  output = {
      "dimension": int(energy.size),
      "stable_rank": float(total / max(float(energy[0]), _EPS)) if energy.size else 0.0,
      "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if nonzero.size else 0.0,
      "rank_90": _rank_for_fraction(energy, 0.90),
      "rank_95": _rank_for_fraction(energy, 0.95),
      "rank_99": _rank_for_fraction(energy, 0.99),
      "rank_999": _rank_for_fraction(energy, 0.999),
      "energy_fraction": {},
  }
  cumulative = np.cumsum(energy)
  for rank in ranks:
    if rank <= energy.size:
      output["energy_fraction"][str(rank)] = float(cumulative[rank - 1] / max(total, _EPS))
  return output


def _matrix_svd(matrix: np.ndarray, ranks: tuple[int, ...]):
  matrix = np.asarray(matrix, np.float32)
  u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
  summary = _spectrum_summary(np.square(singular.astype(np.float64)), ranks)
  summary["shape"] = list(matrix.shape)
  summary["numerical_rank"] = int(np.linalg.matrix_rank(matrix))
  return summary, (u, singular, vh)


def _empty_moment_accumulator():
  return {"count": 0.0, "sum": None, "gram": None}


def _add_moments(accumulator, moments):
  for layer, layer_values in moments.items():
    for name, values in layer_values.items():
      target = accumulator[layer][name]
      count = float(np.asarray(values["count"]))
      total = np.asarray(values["sum"], np.float64)
      gram = np.asarray(values["gram"], np.float64)
      target["count"] += count
      target["sum"] = total if target["sum"] is None else target["sum"] + total
      target["gram"] = gram if target["gram"] is None else target["gram"] + gram


def _covariance(moment, centered: bool):
  count = max(moment["count"], 1.0)
  second = moment["gram"] / count
  mean = moment["sum"] / count
  covariance = second - np.outer(mean, mean) if centered else second
  return (covariance + covariance.T) * 0.5, mean, second


def _moment_rank_report(fit, heldout, ranks):
  report = {}
  for centered in (False, True):
    fit_cov, fit_mean, fit_second = _covariance(fit, centered)
    held_cov, held_mean, held_second = _covariance(heldout, centered)
    fit_values, fit_vectors = np.linalg.eigh(fit_cov)
    held_values = np.linalg.eigvalsh(held_cov)
    order = np.argsort(fit_values)[::-1]
    fit_values = np.maximum(fit_values[order], 0)
    fit_vectors = fit_vectors[:, order]
    held_values = np.maximum(held_values[::-1], 0)
    split = {
        "fit": _spectrum_summary(fit_values, ranks),
        "heldout": _spectrum_summary(held_values, ranks),
        "heldout_energy_in_fit_basis": {},
    }
    held_total = max(float(np.trace(held_cov)), _EPS)
    for rank in ranks:
      if rank <= fit_vectors.shape[1]:
        basis = fit_vectors[:, :rank]
        split["heldout_energy_in_fit_basis"][str(rank)] = float(
            np.trace(basis.T @ held_cov @ basis) / held_total)
    report["centered" if centered else "uncentered"] = split
  fit_mean_energy = float(np.sum(np.square(fit_mean)))
  held_mean_energy = float(np.sum(np.square(held_mean)))
  report["mean_energy_fraction"] = {
      "fit": fit_mean_energy / max(float(np.trace(fit_second)), _EPS),
      "heldout": held_mean_energy / max(float(np.trace(held_second)), _EPS),
  }
  return report


def _normalize_per_head(values: np.ndarray, heads: int, epsilon: float) -> np.ndarray:
  values = values.reshape(values.shape[0], heads, -1)
  return values / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + epsilon)


def _reconstruction_report(base, candidate, heads, epsilon):
  base = np.asarray(base, np.float32)
  candidate = np.asarray(candidate, np.float32)

  def metrics(left, right):
    diff = right - left
    left_sq = float(np.sum(np.square(left), dtype=np.float64))
    right_sq = float(np.sum(np.square(right), dtype=np.float64))
    dot = float(np.sum(left * right, dtype=np.float64))
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    cosine = np.sum(left * right, axis=-1) / np.maximum(left_norm * right_norm, _EPS)
    return {
        "relative_rms_error": float(np.sqrt(
            np.sum(np.square(diff), dtype=np.float64) / max(left_sq, _EPS))),
        "global_cosine": dot / max(math.sqrt(left_sq * right_sq), _EPS),
        "per_vector_cosine_p50": float(np.percentile(cosine, 50)),
        "per_vector_cosine_p10": float(np.percentile(cosine, 10)),
    }

  return {
      "pre_rms": metrics(base, candidate),
      "post_rms": metrics(
          _normalize_per_head(base, heads, epsilon),
          _normalize_per_head(candidate, heads, epsilon)),
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
  }


def _device_array_like(value, array):
  result = jnp.asarray(array, value.dtype)
  sharding = getattr(value, "sharding", None)
  return jax.device_put(result, sharding) if sharding is not None else result


def _build_ploc_variant(params, ploc_info, bases, biases):
  flat = dict(flatten_dict(params))
  for layer, info in ploc_info.items():
    path = info["path"]
    kernel = info["matrix"]
    basis = bases[layer]
    projected = (kernel @ basis) @ basis.T
    flat[path] = _device_array_like(flat[path], projected.reshape(info["shape"]))
    bias_path = path[:-1] + ("bias",)
    flat[bias_path] = jnp.asarray(
        biases[layer].reshape(info["shape"][1:]), flat[path].dtype)
  updated = unflatten_dict(flat)
  return freeze(updated) if isinstance(params, FrozenDict) else updated


def _add_zero_ploc_bias(params, ploc_info):
  flat = dict(flatten_dict(params))
  for info in ploc_info.values():
    path = info["path"]
    flat[path[:-1] + ("bias",)] = jnp.zeros(info["shape"][1:], dtype=flat[path].dtype)
  updated = unflatten_dict(flat)
  return freeze(updated) if isinstance(params, FrozenDict) else updated


def run(config) -> None:
  started = time.perf_counter()
  if not config.only_eval or not config.bam_diagnostics:
    raise ValueError("requires only_eval=True and bam_diagnostics=True")
  if config.bam_write_v_mode != "x":
    raise ValueError(f"expected Direct write_v_mode=x, got {config.bam_write_v_mode}")
  num_batches = int(os.environ.get("BAM_PROJ_RANK_BATCHES", "4"))
  stride = int(os.environ.get("BAM_PROJ_RANK_TOKEN_STRIDE", "32"))
  ranks = tuple(int(value) for value in os.environ.get(
      "BAM_PROJ_RANKS", "64,128,256").split(","))
  if num_batches < 2 or num_batches % 2 or stride <= 0:
    raise ValueError("use an even BAM_PROJ_RANK_BATCHES >=2 and positive stride")
  output_dir = Path(os.environ.get(
      "BAM_PROJ_RANK_OUTPUT_DIR", "/tmp/bam_projection_rank_diagnostics"))
  output_dir.mkdir(parents=True, exist_ok=True)
  report_path = output_dir / "projection_rank.json"

  init_rng, writer, checkpoint_manager, mesh, capture_model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      capture_model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  setup_seconds = time.perf_counter() - started

  ploc_info = {}
  wr_info = {}
  for path, value in flatten_dict(state.params).items():
    layer = _layer_from_path(path)
    if layer is None or path[-1] != "kernel":
      continue
    if path[-2] == "P_loc":
      kernel = np.asarray(jax.device_get(value), np.float32)
      ploc_info[layer] = {
          "path": path,
          "shape": kernel.shape,
          "matrix": kernel.reshape(kernel.shape[0], -1),
      }
    elif path[-2] == "W_R":
      kernel = np.asarray(jax.device_get(value), np.float32)
      wr_info[layer] = {
          "path": path,
          "shape": kernel.shape,
          "row": kernel[..., :config.bam_k].reshape(kernel.shape[0], -1),
          "col": kernel[..., config.bam_k:].reshape(kernel.shape[0], -1),
      }
  expected_layers = set(range(config.num_decoder_layers))
  if set(ploc_info) != expected_layers or set(wr_info) != expected_layers:
    raise RuntimeError(
        f"projection layers differ: P_loc={sorted(ploc_info)} W_R={sorted(wr_info)}")

  weight_report = {}
  ploc_svds = {}
  for layer in sorted(ploc_info):
    ploc_summary, ploc_svd = _matrix_svd(ploc_info[layer]["matrix"], ranks)
    row_ranks = tuple(sorted(set(rank for rank in ranks if rank <= wr_info[layer]["row"].shape[1])))
    col_ranks = (8, 16, 32, 64, 96, 128)
    row_summary, _ = _matrix_svd(wr_info[layer]["row"], row_ranks)
    col_summary, _ = _matrix_svd(wr_info[layer]["col"], col_ranks)
    ploc_svds[layer] = ploc_svd
    weight_report[f"layer_{layer:02d}"] = {
        "P_loc_joint": ploc_summary,
        "W_R_row_joint": row_summary,
        "W_R_col_joint": col_summary,
    }
  del wr_info

  capture_forward = jax.jit(
      lambda params, batch, rng: _capture_forward(
          capture_model, params, batch, rng, stride, int(config.bam_k)))
  batches = []
  cohort = []
  moment_groups = {
      "fit": defaultdict(lambda: defaultdict(_empty_moment_accumulator)),
      "heldout": defaultdict(lambda: defaultdict(_empty_moment_accumulator)),
  }
  ploc_groups = {
      "fit": defaultdict(list),
      "heldout": defaultdict(list),
  }
  capture_timings = []
  split_batch = num_batches // 2
  for batch_index in range(num_batches):
    batch = next(eval_iterator)
    batches.append(batch)
    tokens = np.asarray(jax.device_get(batch["inputs"]))
    cohort.append({
        "batch": batch_index,
        "split": "fit" if batch_index < split_batch else "heldout",
        "sequence_hashes": [
            hashlib.sha256(sequence.tobytes()).hexdigest()[:16] for sequence in tokens],
    })
    batch_started = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      metrics_device, moments_device, ploc_device, valid_device = capture_forward(
          state.params, batch, jax.random.fold_in(init_rng, batch_index))
    jax.block_until_ready((metrics_device, moments_device, ploc_device, valid_device))
    execute_seconds = time.perf_counter() - batch_started
    transfer_started = time.perf_counter()
    metrics = jax.device_get(metrics_device)
    moments = jax.device_get(moments_device)
    ploc = jax.device_get(ploc_device)
    valid = np.asarray(jax.device_get(valid_device), bool)
    transfer_seconds = time.perf_counter() - transfer_started
    group = "fit" if batch_index < split_batch else "heldout"
    _add_moments(moment_groups[group], moments)
    for layer, values in ploc.items():
      flattened = np.asarray(values, np.float32).reshape(-1, values.shape[-1])
      ploc_groups[group][layer].append(flattened[valid.reshape(-1)])
    capture_timings.append({
        "batch": batch_index,
        "compile_execute": execute_seconds,
        "device_to_host": transfer_seconds,
        "loss": float(metrics["total_loss"] / metrics["total_weights"]),
    })
    print(
        f"BAM_PROJ_RANK_CAPTURE batch={batch_index} "
        f"loss={capture_timings[-1]['loss']:.8f} seconds={execute_seconds:.2f}",
        flush=True)

  activation_report = {}
  activation_bases = {rank: {} for rank in ranks}
  activation_biases = {rank: {} for rank in ranks}
  weight_bases = {rank: {} for rank in ranks}
  weight_biases = {rank: {} for rank in ranks}
  for layer in sorted(ploc_info):
    fit = np.concatenate(ploc_groups["fit"][layer], axis=0)
    heldout = np.concatenate(ploc_groups["heldout"][layer], axis=0)
    fit_mean = np.mean(fit, axis=0)
    fit_centered = fit - fit_mean
    covariance = fit_centered.T @ fit_centered / max(fit_centered.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    layer_report = {
        "fit_samples": int(fit.shape[0]),
        "heldout_samples": int(heldout.shape[0]),
        "centered_activation_spectrum": _spectrum_summary(eigenvalues, ranks),
        "weight_svd_reconstruction": {},
        "activation_pca_affine_reconstruction": {},
    }
    _, _, weight_vh = ploc_svds[layer]
    for rank in ranks:
      if rank > weight_vh.shape[0]:
        continue
      weight_basis = weight_vh[:rank].T
      weight_fit_projected = (fit @ weight_basis) @ weight_basis.T
      weight_bias = np.mean(fit - weight_fit_projected, axis=0)
      weight_heldout = (heldout @ weight_basis) @ weight_basis.T + weight_bias
      layer_report["weight_svd_reconstruction"][str(rank)] = _reconstruction_report(
          heldout, weight_heldout, int(config.num_query_heads),
          float(config.normalization_layer_epsilon))
      weight_bases[rank][layer] = weight_basis
      weight_biases[rank][layer] = weight_bias

      activation_basis = eigenvectors[:, :rank]
      activation_bias = fit_mean - (fit_mean @ activation_basis) @ activation_basis.T
      activation_heldout = (
          (heldout @ activation_basis) @ activation_basis.T + activation_bias)
      layer_report["activation_pca_affine_reconstruction"][str(rank)] = (
          _reconstruction_report(
              heldout, activation_heldout, int(config.num_query_heads),
              float(config.normalization_layer_epsilon)))
      activation_bases[rank][layer] = activation_basis
      activation_biases[rank][layer] = activation_bias
    activation_report[f"layer_{layer:02d}"] = layer_report

  wr_activation_report = {}
  for layer in sorted(moment_groups["fit"]):
    wr_activation_report[f"layer_{layer:02d}"] = {}
    for name, fit_moment in sorted(moment_groups["fit"][layer].items()):
      heldout_moment = moment_groups["heldout"][layer][name]
      dimension = fit_moment["sum"].size
      candidate_ranks = tuple(rank for rank in (
          (8, 16, 32, 64, 96, 128) if name.endswith("_col") else ranks)
                              if rank <= dimension)
      wr_activation_report[f"layer_{layer:02d}"][name] = _moment_rank_report(
          fit_moment, heldout_moment, candidate_ranks)

  loss_config = _ConfigOverlay(config, bam_diagnostics=False, bam_write_v_mode="x_bias")
  loss_model = train.Transformer(loss_config, mesh, quant=capture_model.quant)
  loss_forward = jax.jit(
      lambda params, batch, rng: _loss_forward(loss_model, params, batch, rng))
  base_params = _add_zero_ploc_bias(state.params, ploc_info)
  variant_specs = [("baseline", base_params)]
  for rank in ranks:
    variant_specs.append((
        f"activation_pca_affine_rank{rank}",
        _build_ploc_variant(
            base_params, ploc_info, activation_bases[rank], activation_biases[rank])))
  if 128 in ranks:
    variant_specs.append((
        "weight_svd_affine_rank128",
        _build_ploc_variant(base_params, ploc_info, weight_bases[128], weight_biases[128])))

  loss_results = {}
  for variant_index, (variant_name, variant_params) in enumerate(variant_specs):
    total_loss = total_weights = accuracy = 0.0
    split_totals = {
        "fit": {"loss": 0.0, "weights": 0.0},
        "heldout": {"loss": 0.0, "weights": 0.0},
    }
    sequence_loss = []
    seconds = []
    for batch_index, batch in enumerate(batches):
      run_started = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics_device = loss_forward(
            variant_params, batch, jax.random.fold_in(init_rng, batch_index))
      metrics = jax.device_get(jax.block_until_ready(metrics_device))
      seconds.append(time.perf_counter() - run_started)
      total_loss += float(metrics["total_loss"])
      total_weights += float(metrics["total_weights"])
      accuracy += float(metrics["accuracy_numerator"])
      split_name = "fit" if batch_index < split_batch else "heldout"
      split_totals[split_name]["loss"] += float(metrics["total_loss"])
      split_totals[split_name]["weights"] += float(metrics["total_weights"])
      sequence_loss.extend(np.asarray(metrics["sequence_loss"]).tolist())
    sequence_loss = np.asarray(sequence_loss, np.float64)
    loss_results[variant_name] = {
        "loss": total_loss / max(total_weights, 1.0),
        "fit_loss": split_totals["fit"]["loss"] / max(split_totals["fit"]["weights"], 1.0),
        "heldout_loss": (
            split_totals["heldout"]["loss"]
            / max(split_totals["heldout"]["weights"], 1.0)),
        "accuracy": accuracy / max(total_weights, 1.0),
        "sequence_loss": sequence_loss.tolist(),
        "seconds": seconds,
        "execution_order": variant_index,
    }
    print(
        f"BAM_PROJ_RANK_LOSS variant={variant_name} "
        f"loss={loss_results[variant_name]['loss']:.8f} "
        f"heldout={loss_results[variant_name]['heldout_loss']:.8f}", flush=True)

  baseline_sequence = np.asarray(loss_results["baseline"]["sequence_loss"])
  baseline_loss = loss_results["baseline"]["loss"]
  heldout_start = split_batch * baseline_sequence.size // num_batches
  for name, result in loss_results.items():
    sequence = np.asarray(result["sequence_loss"])
    delta = sequence - baseline_sequence
    result["loss_delta"] = float(result["loss"] - baseline_loss)
    result["sequence_delta"] = _delta_summary(delta)
    result["heldout_sequence_delta"] = _delta_summary(delta[heldout_start:])
    del result["sequence_loss"]

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_PROJ_RANK_COMMIT", "unknown"),
          "num_batches": num_batches,
          "fit_batches": split_batch,
          "heldout_batches": num_batches - split_batch,
          "batch_size": int(batches[0]["inputs"].shape[0]),
          "sequence_length": int(batches[0]["inputs"].shape[1]),
          "token_stride": stride,
          "ranks": list(ranks),
          "setup_seconds": setup_seconds,
          "total_seconds": time.perf_counter() - started,
          "checkpoint_mutated": False,
          "devices": [str(device) for device in jax.devices()],
      },
      "cohort": cohort,
      "capture_timings": capture_timings,
      "weight_spectra": weight_report,
      "P_loc_activations": activation_report,
      "W_R_activations": wr_activation_report,
      "paired_loss": loss_results,
  }
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_PROJ_RANK_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = _ConfigOverlay(
      pyconfig.initialize(argv),
      bam_diagnostics=True,
      eval_shuffle_buffer_size=32768,
  )
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

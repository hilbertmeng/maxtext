"""Evaluate V1 BAM historical-cache structure and causal block approximations.

The runner restores one read-only checkpoint, freezes four shuffled Pile eval
batches, and evaluates every approximation on exactly that cohort.  Attention
only exposes the necessary raw tensors; all statistics and policy live here.
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
_EPS = 1.0e-12
_LAGS = (1, 2, 4, 8, 16, 32)
_TOP_K = (1, 2, 4, 8, 16, 32, 64, 128)
_WINDOWS = (64, 128, 256, 512, 1024)
_COMPARE_NAMES = ("Mbar_fetch", "Mbar", "y_full")


class _ConfigOverlay:
  """Read-only config view with a few static model overrides."""

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


def _group_raw(collections) -> dict[int, dict[str, jax.Array]]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  for path, value in flatten_dict(collections.get("bam_raw", {})).items():
    layer = None
    for component in path:
      match = _LAYER_RE.fullmatch(component)
      if match:
        layer = int(match.group(1))
        break
    if layer is not None:
      grouped[layer][path[-1]] = _unwrap(value)
  return dict(sorted(grouped.items()))


def _adjacent_m_accumulators(M, positions, segments):
  """Small sufficient statistics for matrix-stream correlation across time."""
  M = jnp.asarray(M, jnp.float32)
  output = {}
  for lag in _LAGS:
    current = M[:, lag:]
    previous = M[:, :-lag]
    valid = (
        (segments[:, lag:] != 0)
        & (segments[:, lag:] == segments[:, :-lag])
        & (positions[:, lag:] - positions[:, :-lag] == lag)
    )
    valid_f = valid.astype(jnp.float32)
    dot = jnp.sum(current * previous, axis=(-2, -1))
    current_sq = jnp.sum(jnp.square(current), axis=(-2, -1))
    previous_sq = jnp.sum(jnp.square(previous), axis=(-2, -1))
    diff_sq = jnp.sum(jnp.square(current - previous), axis=(-2, -1))
    cosine = dot / jnp.maximum(jnp.sqrt(current_sq * previous_sq), _EPS)
    relative_delta = jnp.sqrt(diff_sq) / jnp.maximum(jnp.sqrt(current_sq), _EPS)
    output[f"lag_{lag}"] = {
        "dot": jnp.sum(dot * valid_f),
        "current_sq": jnp.sum(current_sq * valid_f),
        "previous_sq": jnp.sum(previous_sq * valid_f),
        "diff_sq": jnp.sum(diff_sq * valid_f),
        "cosine_sum": jnp.sum(cosine * valid_f),
        "relative_delta_sum": jnp.sum(relative_delta * valid_f),
        "count": jnp.sum(valid_f),
    }
  return output


def _forward(model, params, batch, rng, stride, capture_structure):
  rng1, aqt_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=["bam_raw"],
  )
  mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.sum(mask, axis=-1)
  grouped = _group_raw(collections)
  sampled = {}
  adjacent = {}
  for layer, raw in grouped.items():
    sampled[layer] = {
        name: (
            value[:, :, ::stride]
            if name in ("Mbar", "Mbar_fetch", "fetch_alpha", "fetch_alpha_pre_diagonal")
            else value
            if name in ("fetch_mix_logits", "fetch_mix_weights")
            else value[:, ::stride]
        )
        for name, value in raw.items()
        if name in _COMPARE_NAMES
        or (capture_structure and name in (
            "fetch_alpha", "fetch_alpha_pre_diagonal",
            "fetch_mix_logits", "fetch_mix_weights"))
    }
    if capture_structure:
      adjacent[layer] = _adjacent_m_accumulators(
          raw["M_in"], batch["inputs_position"], batch["inputs_segmentation"])
  return {
      "total_loss": jnp.sum(xent * mask),
      "total_weights": jnp.sum(mask),
      "sequence_loss": jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1),
      "sequence_weights": sequence_weights,
  }, sampled, adjacent


def _describe(values) -> dict[str, float]:
  values = np.asarray(values, np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  if not values.size:
    return {key: float("nan") for key in (
        "mean", "std", "rms", "min", "p10", "p25", "p50", "p75", "p90", "p99", "max")}
  return {
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "rms": float(np.sqrt(np.mean(np.square(values)))),
      "min": float(np.min(values)),
      "p10": float(np.percentile(values, 10)),
      "p25": float(np.percentile(values, 25)),
      "p50": float(np.percentile(values, 50)),
      "p75": float(np.percentile(values, 75)),
      "p90": float(np.percentile(values, 90)),
      "p99": float(np.percentile(values, 99)),
      "max": float(np.max(values)),
  }


class _DistributionCollector:
  def __init__(self):
    self._values: dict[str, list[np.ndarray]] = defaultdict(list)

  def add(self, name, values):
    self._values[name].append(np.asarray(values).reshape(-1))

  def report(self):
    return {
        name: _describe(np.concatenate(values))
        for name, values in sorted(self._values.items())
    }


def _joint_mix_report(collector):
  """Cross-head structure of the token-wise signed coefficient vectors."""
  columns = []
  head = 0
  while f"head_{head:02d}/coefficient" in collector._values:
    columns.append(np.concatenate(collector._values[f"head_{head:02d}/coefficient"]))
    head += 1
  weights = np.stack(columns, axis=-1).astype(np.float64)
  mean = np.mean(weights, axis=0)
  second_moment = weights.T @ weights / weights.shape[0]
  centered = weights - mean
  covariance = centered.T @ centered / weights.shape[0]
  std = np.sqrt(np.maximum(np.diag(covariance), _EPS))
  correlation = covariance / np.maximum(std[:, None] * std[None, :], _EPS)

  def spectrum(matrix):
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix)[::-1], 0.0)
    total = max(float(np.sum(eigenvalues)), _EPS)
    probabilities = eigenvalues / total
    return {
        "eigenvalues": eigenvalues.tolist(),
        "top1_energy_fraction": float(probabilities[0]),
        "top4_energy_fraction": float(np.sum(probabilities[:4])),
        "effective_rank": float(np.exp(-np.sum(
            probabilities * np.log(np.maximum(probabilities, _EPS))))),
    }

  return {
      "sample_count": int(weights.shape[0]),
      "mean_vector": mean.tolist(),
      "mean_vector_l2": float(np.linalg.norm(mean)),
      "head_rms": np.sqrt(np.diag(second_moment)).tolist(),
      "head_correlation": correlation.tolist(),
      "second_moment_spectrum": spectrum(second_moment),
      "centered_covariance_spectrum": spectrum(covariance),
  }


def _collect_mix_stats(collector, logits, weights, positions, segments):
  logits = np.asarray(logits, np.float32)
  weights = np.asarray(weights, np.float32)
  energy = np.square(weights)
  abs_weights = np.abs(weights)
  coefficient_l1 = np.sum(abs_weights, axis=-1)
  coefficient_positive_mass = np.sum(np.maximum(weights, 0), axis=-1)
  coefficient_negative_mass = np.sum(np.maximum(-weights, 0), axis=-1)
  coefficient_sum = np.sum(weights, axis=-1)
  token_valid = segments != 0
  token_count = np.maximum(np.sum(token_valid, axis=1), 1)
  per_sequence_mean = lambda values: np.sum(
      np.asarray(values) * token_valid, axis=1) / token_count
  effective_heads = 1.0 / np.maximum(np.sum(np.square(energy), axis=-1), _EPS)
  collector.add("coefficient/l2_norm", np.linalg.norm(weights, axis=-1)[token_valid])
  collector.add("coefficient/l1", coefficient_l1[token_valid])
  collector.add("coefficient/sum", coefficient_sum[token_valid])
  collector.add("coefficient/positive_mass", coefficient_positive_mass[token_valid])
  collector.add("coefficient/negative_abs_mass", coefficient_negative_mass[token_valid])
  collector.add("coefficient/negative_abs_mass_fraction", (
      coefficient_negative_mass / np.maximum(coefficient_l1, _EPS))[token_valid])
  collector.add("coefficient/cancellation_fraction", (
      1 - np.abs(coefficient_sum) / np.maximum(coefficient_l1, _EPS))[token_valid])
  collector.add("coefficient/no_positive", (coefficient_positive_mass == 0)[token_valid])
  collector.add("coefficient/max_abs", np.max(np.abs(weights), axis=-1)[token_valid])
  collector.add("coefficient/effective_heads", effective_heads[token_valid])
  collector.add("coefficient/positive_fraction", np.mean(weights > 0, axis=-1)[token_valid])
  collector.add("raw_logits/rms", np.sqrt(np.mean(np.square(logits), axis=-1))[token_valid])
  sorted_energy = np.sort(energy, axis=-1)[..., ::-1]
  collector.add("coefficient/top1_energy", sorted_energy[..., 0][token_valid])
  collector.add("coefficient/top2_energy", np.sum(sorted_energy[..., :2], axis=-1)[token_valid])
  collector.add("per_sequence/coefficient_effective_heads_mean",
                per_sequence_mean(effective_heads))
  collector.add("per_sequence/coefficient_max_abs_mean",
                per_sequence_mean(np.max(np.abs(weights), axis=-1)))
  for head in range(weights.shape[-1]):
    collector.add(f"head_{head:02d}/coefficient", weights[..., head][token_valid])
    collector.add(f"head_{head:02d}/energy", energy[..., head][token_valid])

  valid = (
      (segments[:, 1:] != 0)
      & (segments[:, 1:] == segments[:, :-1])
      & (positions[:, 1:] - positions[:, :-1] == 1)
  )
  left, right = weights[:, 1:], weights[:, :-1]
  cosine = np.sum(left * right, axis=-1) / np.maximum(
      np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), _EPS)
  collector.add("temporal/adjacent_cosine", cosine[valid])
  collector.add("temporal/adjacent_l2_delta", np.linalg.norm(left - right, axis=-1)[valid])


def _collect_alpha_stats(collector, pre_alpha, final_alpha, positions, segments, stride):
  pre_alpha = np.asarray(pre_alpha[:, 0], np.float32)
  final_alpha = np.asarray(final_alpha[:, 0], np.float32)
  query_indices = np.arange(0, positions.shape[1], stride)[:pre_alpha.shape[1]]
  query_positions = positions[:, query_indices]
  query_segments = segments[:, query_indices]
  query_valid = query_segments != 0
  add_row = lambda name, values: collector.add(name, np.asarray(values)[query_valid])
  query_count = np.maximum(np.sum(query_valid, axis=1), 1)
  add_sequence_mean = lambda name, values: collector.add(
      name, np.sum(np.asarray(values) * query_valid, axis=1) / query_count)
  valid = (
      (query_segments[:, :, None] != 0)
      & (query_segments[:, :, None] == segments[:, None, :])
      & (positions[:, None, :] <= query_positions[:, :, None])
  )
  values = np.where(valid, pre_alpha, 0.0)
  abs_values = np.abs(values)
  l1 = np.sum(abs_values, axis=-1)
  l2_sq = np.sum(np.square(values), axis=-1)
  valid_count = np.maximum(np.sum(valid, axis=-1), 1)
  add_row("alpha/signed_sum", np.sum(values, axis=-1))
  add_row("alpha/l1", l1)
  add_row("alpha/l2", np.sqrt(l2_sq))
  add_row("alpha/max_abs", np.max(abs_values, axis=-1))
  add_row("alpha/negative_fraction", np.sum((values < 0) & valid, axis=-1) / valid_count)
  add_row("alpha/positive_mass", np.sum(np.maximum(values, 0), axis=-1))
  add_row("alpha/negative_abs_mass", np.sum(np.maximum(-values, 0), axis=-1))
  negative_abs_mass = np.sum(np.maximum(-values, 0), axis=-1)
  signed_sum = np.sum(values, axis=-1)
  negative_abs_mass_fraction = negative_abs_mass / np.maximum(l1, _EPS)
  add_row("alpha/negative_abs_mass_fraction", negative_abs_mass_fraction)
  add_row("alpha/cancellation_fraction", 1 - np.abs(signed_sum) / np.maximum(l1, _EPS))
  add_sequence_mean(
      "per_sequence/alpha_negative_abs_mass_fraction_mean", negative_abs_mass_fraction)
  effective_support = np.square(l1) / np.maximum(l2_sq, _EPS)
  add_row("alpha/effective_support", effective_support)
  add_sequence_mean("per_sequence/alpha_effective_support_mean", effective_support)

  max_k = max(_TOP_K)
  largest = np.partition(abs_values, -max_k, axis=-1)[..., -max_k:]
  largest = np.sort(largest, axis=-1)[..., ::-1]
  for k in _TOP_K:
    add_row(f"alpha/top{k}_abs_mass_fraction",
            np.sum(largest[..., :k], axis=-1) / np.maximum(l1, _EPS))

  lag = query_positions[:, :, None] - positions[:, None, :]
  for window in _WINDOWS:
    in_window = valid & (lag < window)
    window_abs_fraction = (
        np.sum(abs_values * in_window, axis=-1) / np.maximum(l1, _EPS))
    add_row(f"alpha/window{window}_abs_mass_fraction", window_abs_fraction)
    if window == 256:
      add_sequence_mean(
          "per_sequence/alpha_window256_abs_mass_fraction_mean", window_abs_fraction)
    add_row(f"alpha/window{window}_signed_sum", np.sum(values * in_window, axis=-1))

  batch_indices = np.arange(pre_alpha.shape[0])[:, None]
  query_sample_indices = np.arange(pre_alpha.shape[1])[None, :]
  source_indices = np.broadcast_to(query_indices[None, :], query_sample_indices.shape)
  pre_diagonal = pre_alpha[batch_indices, query_sample_indices, source_indices]
  final_diagonal = final_alpha[batch_indices, query_sample_indices, source_indices]
  add_row("alpha/pre_diagonal", pre_diagonal)
  add_row("alpha/final_diagonal", final_diagonal)


class _ErrorAccumulator:
  def __init__(self):
    self.diff_sq = 0.0
    self.base_sq = 0.0
    self.candidate_sq = 0.0
    self.dot = 0.0
    self.count = 0
    self.relative_norms = []

  def add(self, base, candidate):
    base = np.asarray(base, np.float32)
    candidate = np.asarray(candidate, np.float32)
    diff = candidate - base
    self.diff_sq += float(np.sum(np.square(diff), dtype=np.float64))
    self.base_sq += float(np.sum(np.square(base), dtype=np.float64))
    self.candidate_sq += float(np.sum(np.square(candidate), dtype=np.float64))
    self.dot += float(np.sum(base * candidate, dtype=np.float64))
    self.count += base.size
    vector_axes = (-2, -1) if base.ndim == 5 else (-1,)
    base_norm = np.linalg.norm(base, axis=vector_axes)
    diff_norm = np.linalg.norm(diff, axis=vector_axes)
    self.relative_norms.append((diff_norm / np.maximum(base_norm, _EPS)).reshape(-1))

  def report(self):
    relative = np.concatenate(self.relative_norms) if self.relative_norms else np.array([])
    return {
        "relative_rms_error": float(np.sqrt(self.diff_sq / max(self.base_sq, _EPS))),
        "cosine": float(self.dot / max(np.sqrt(self.base_sq * self.candidate_sq), _EPS)),
        "candidate_to_base_rms": float(np.sqrt(self.candidate_sq / max(self.base_sq, _EPS))),
        "per_query_relative_norm_error": _describe(relative),
        "sampled_value_count": self.count,
    }


def _merge_adjacent(accumulator, batch_adjacent):
  for layer, layer_values in batch_adjacent.items():
    for lag, values in layer_values.items():
      for name, value in values.items():
        accumulator[layer][lag][name] += float(np.asarray(value))


def _adjacent_report(accumulator):
  report = {}
  overall = defaultdict(lambda: defaultdict(float))
  for layer, layer_values in sorted(accumulator.items()):
    report[f"layer_{layer:02d}"] = {}
    for lag, values in sorted(layer_values.items()):
      count = max(values["count"], 1.0)
      report[f"layer_{layer:02d}"][lag] = {
          "global_cosine": values["dot"] / max(
              np.sqrt(values["current_sq"] * values["previous_sq"]), _EPS),
          "mean_token_cosine": values["cosine_sum"] / count,
          "global_relative_delta_rms": np.sqrt(values["diff_sq"] / max(values["current_sq"], _EPS)),
          "mean_token_relative_delta": values["relative_delta_sum"] / count,
          "pair_count": int(values["count"]),
      }
      for name, value in values.items():
        overall[lag][name] += value
  report["all_layers"] = {}
  for lag, values in sorted(overall.items()):
    count = max(values["count"], 1.0)
    report["all_layers"][lag] = {
        "global_cosine": values["dot"] / max(
            np.sqrt(values["current_sq"] * values["previous_sq"]), _EPS),
        "mean_token_cosine": values["cosine_sum"] / count,
        "global_relative_delta_rms": np.sqrt(values["diff_sq"] / max(values["current_sq"], _EPS)),
        "mean_token_relative_delta": values["relative_delta_sum"] / count,
        "pair_count": int(values["count"]),
    }
  return report


def _write_report(path, report):
  path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_microbatches(batch, microbatch_size):
  """Yield stable contiguous slices without changing the input cohort."""
  batch_size = int(batch["inputs"].shape[0])
  if batch_size % microbatch_size:
    raise ValueError(
        f"logical batch size {batch_size} is not divisible by microbatch size {microbatch_size}")
  for microbatch_index, start in enumerate(range(0, batch_size, microbatch_size)):
    end = start + microbatch_size
    yield microbatch_index, {name: value[start:end] for name, value in batch.items()}


def _variant_specs(group):
  if group == "sign":
    return (
        ("mix_abs_l2", {"bam_fetch_sign_ablation": "mix_abs"}),
        ("mix_positive_l2", {"bam_fetch_sign_ablation": "mix_positive_l2"}),
        ("alpha_abs", {"bam_fetch_sign_ablation": "alpha_abs"}),
        ("alpha_positive_raw", {"bam_fetch_sign_ablation": "alpha_positive_raw"}),
        ("alpha_positive_l2", {"bam_fetch_sign_ablation": "alpha_positive_l2"}),
        ("alpha_negative_l2", {"bam_fetch_sign_ablation": "alpha_negative_l2"}),
    )
  if group != "cache":
    raise ValueError(f"Unknown BAM_CACHE_DIAG_VARIANT_GROUP={group!r}")
  specs = []
  for block_size in (8, 16, 32):
    for mode in ("mean", "linear"):
      specs.append((f"block{block_size}_{mode}", {
          "bam_fetch_sliding_window_size": None,
          "bam_fetch_temporal_block_size": block_size,
          "bam_fetch_temporal_block_mode": mode,
          "bam_fetch_temporal_recent_window_size": None,
      }))
  specs.extend((
      ("window256_only", {
          "bam_fetch_sliding_window_size": 256,
          "bam_fetch_temporal_block_size": None,
          "bam_fetch_temporal_block_mode": "none",
          "bam_fetch_temporal_recent_window_size": None,
      }),
      ("window256_oldblock16_mean", {
          "bam_fetch_sliding_window_size": None,
          "bam_fetch_temporal_block_size": 16,
          "bam_fetch_temporal_block_mode": "mean",
          "bam_fetch_temporal_recent_window_size": 256,
      }),
      ("window256_oldblock16_linear", {
          "bam_fetch_sliding_window_size": None,
          "bam_fetch_temporal_block_size": 16,
          "bam_fetch_temporal_block_mode": "linear",
          "bam_fetch_temporal_recent_window_size": 256,
      }),
  ))
  return specs


def run(config):
  if not config.only_eval or not config.bam_diagnostics:
    raise ValueError("bam_cache_diagnostics requires only_eval=True and bam_diagnostics=True")
  num_batches = int(os.environ.get("BAM_CACHE_DIAG_BATCHES", "4"))
  requested_microbatch_size = int(os.environ.get("BAM_CACHE_DIAG_MICROBATCH_SIZE", "0"))
  stride = int(os.environ.get("BAM_CACHE_DIAG_TOKEN_STRIDE", "32"))
  variant_group = os.environ.get("BAM_CACHE_DIAG_VARIANT_GROUP", "cache")
  output_dir = Path(os.environ.get("BAM_CACHE_DIAG_OUTPUT_DIR", "/tmp/bam_cache_diagnostics"))
  output_dir.mkdir(parents=True, exist_ok=True)
  report_path = output_dir / "bam_cache_diagnostics.json"
  started = time.perf_counter()

  init_rng, writer, checkpoint_manager, mesh, base_model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      base_model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  setup_seconds = time.perf_counter() - started

  batches = []
  cohort = []
  for batch_index in range(num_batches):
    batch = next(eval_iterator)
    batches.append(batch)
    tokens = np.asarray(jax.device_get(batch["inputs"]))
    positions = np.asarray(jax.device_get(batch["inputs_position"]))
    segments = np.asarray(jax.device_get(batch["inputs_segmentation"]))
    cohort.append({
        "batch": batch_index,
        "sequence_hashes": [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in tokens],
        "nonzero_tokens": np.sum(segments != 0, axis=1).astype(int).tolist(),
        "segment_counts": [int(np.unique(row[row != 0]).size) for row in segments],
    })

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_CACHE_DIAG_COMMIT", "unknown"),
          "num_batches": num_batches,
          "batch_size": int(batches[0]["inputs"].shape[0]),
          "microbatch_size": (
              requested_microbatch_size or int(batches[0]["inputs"].shape[0])),
          "sequence_length": int(batches[0]["inputs"].shape[1]),
          "token_stride": stride,
          "variant_group": variant_group,
          "data_shuffle_seed": config.data_shuffle_seed,
          "eval_shuffle_buffer_size": config.eval_shuffle_buffer_size,
          "setup_seconds": setup_seconds,
          "devices": [str(device) for device in jax.devices()],
      },
      "cohort": cohort,
      "baseline": {},
      "mixing": {},
      "adjacent_M": {},
      "variants": {},
  }
  microbatch_size = report["metadata"]["microbatch_size"]
  if microbatch_size <= 0:
    raise ValueError(f"microbatch size must be positive, got {microbatch_size}")
  for batch in batches:
    if int(batch["inputs"].shape[0]) % microbatch_size:
      raise ValueError(
          f"logical batch size {batch['inputs'].shape[0]} is not divisible by "
          f"microbatch size {microbatch_size}")
  report["metadata"]["num_microbatches"] = sum(
      int(batch["inputs"].shape[0]) // microbatch_size for batch in batches)
  _write_report(report_path, report)

  baseline_forward = jax.jit(
      lambda params, batch, rng: _forward(
          base_model, params, batch, rng, stride, True))
  baseline_raw = []
  baseline_total_loss = baseline_total_weights = 0.0
  baseline_sequence_loss = []
  mix_collectors = defaultdict(_DistributionCollector)
  adjacent_accumulator = defaultdict(
      lambda: defaultdict(lambda: defaultdict(float)))
  baseline_timings = []

  execution_index = 0
  for batch_index, batch in enumerate(batches):
    for microbatch_index, microbatch in _iter_microbatches(batch, microbatch_size):
      batch_started = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics_device, sampled_device, adjacent_device = baseline_forward(
            state.params, microbatch, jax.random.fold_in(init_rng, execution_index))
      jax.block_until_ready((metrics_device, sampled_device, adjacent_device))
      compile_execute_seconds = time.perf_counter() - batch_started
      transfer_started = time.perf_counter()
      metrics = jax.device_get(metrics_device)
      sampled = jax.device_get(sampled_device)
      adjacent = jax.device_get(adjacent_device)
      positions = np.asarray(jax.device_get(microbatch["inputs_position"]))
      segments = np.asarray(jax.device_get(microbatch["inputs_segmentation"]))
      transfer_seconds = time.perf_counter() - transfer_started
      baseline_timings.append({
          "batch": batch_index,
          "microbatch": microbatch_index,
          "compile_execute": compile_execute_seconds,
          "device_to_host": transfer_seconds,
      })
      baseline_total_loss += float(metrics["total_loss"])
      baseline_total_weights += float(metrics["total_weights"])
      baseline_sequence_loss.extend(np.asarray(metrics["sequence_loss"]).tolist())
      _merge_adjacent(adjacent_accumulator, adjacent)
      baseline_raw.append({
          layer: {name: np.asarray(values[name]) for name in _COMPARE_NAMES}
          for layer, values in sampled.items()
      })
      for layer, values in sampled.items():
        collector = mix_collectors[layer]
        _collect_mix_stats(
            collector, values["fetch_mix_logits"], values["fetch_mix_weights"],
            positions, segments)
        _collect_alpha_stats(
            collector, values["fetch_alpha_pre_diagonal"], values["fetch_alpha"],
            positions, segments, stride)
      print(f"BAM_CACHE_DIAG baseline batch={batch_index + 1}/{num_batches} "
            f"microbatch={microbatch_index + 1} "
            f"loss={float(metrics['total_loss']) / float(metrics['total_weights']):.6f} "
            f"seconds={compile_execute_seconds:.1f}", flush=True)
      execution_index += 1

  baseline_loss = baseline_total_loss / baseline_total_weights
  report["baseline"] = {
      "loss": baseline_loss,
      "sequence_loss": baseline_sequence_loss,
      "timing_seconds": baseline_timings,
  }
  report["mixing"] = {
      f"layer_{layer:02d}": {
          "distributions": collector.report(),
          "cross_head": _joint_mix_report(collector),
      }
      for layer, collector in sorted(mix_collectors.items())
  }
  all_mix = _DistributionCollector()
  for collector in mix_collectors.values():
    for name, arrays in collector._values.items():
      for values in arrays:
        all_mix.add(name, values)
  report["mixing"]["all_layers"] = {
      "distributions": all_mix.report(),
      "cross_head": _joint_mix_report(all_mix),
  }
  report["adjacent_M"] = _adjacent_report(adjacent_accumulator)
  _write_report(report_path, report)

  for variant_name, overrides in _variant_specs(variant_group):
    variant_config = _ConfigOverlay(config, **overrides)
    variant_model = train.Transformer(variant_config, mesh, quant=base_model.quant)
    variant_forward = jax.jit(
        lambda params, batch, rng, model=variant_model: _forward(
            model, params, batch, rng, stride, False))
    total_loss = total_weights = 0.0
    sequence_loss = []
    errors = {
        "all_layers": {name: _ErrorAccumulator() for name in _COMPARE_NAMES},
        **{
            f"layer_{layer:02d}": {name: _ErrorAccumulator() for name in _COMPARE_NAMES}
            for layer in range(config.num_decoder_layers)
        },
    }
    timings = []
    execution_index = 0
    for batch_index, batch in enumerate(batches):
      for microbatch_index, microbatch in _iter_microbatches(batch, microbatch_size):
        batch_started = time.perf_counter()
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          metrics_device, sampled_device, _ = variant_forward(
              state.params, microbatch, jax.random.fold_in(init_rng, execution_index))
        jax.block_until_ready((metrics_device, sampled_device))
        compile_execute_seconds = time.perf_counter() - batch_started
        transfer_started = time.perf_counter()
        metrics = jax.device_get(metrics_device)
        sampled = jax.device_get(sampled_device)
        transfer_seconds = time.perf_counter() - transfer_started
        timings.append({
            "batch": batch_index,
            "microbatch": microbatch_index,
            "compile_execute": compile_execute_seconds,
            "device_to_host": transfer_seconds,
        })
        total_loss += float(metrics["total_loss"])
        total_weights += float(metrics["total_weights"])
        sequence_loss.extend(np.asarray(metrics["sequence_loss"]).tolist())
        for layer, values in sampled.items():
          for name in _COMPARE_NAMES:
            errors[f"layer_{layer:02d}"][name].add(
                baseline_raw[execution_index][layer][name], values[name])
            errors["all_layers"][name].add(
                baseline_raw[execution_index][layer][name], values[name])
        print(f"BAM_CACHE_DIAG variant={variant_name} batch={batch_index + 1}/{num_batches} "
              f"microbatch={microbatch_index + 1} "
              f"loss={float(metrics['total_loss']) / float(metrics['total_weights']):.6f} "
              f"seconds={compile_execute_seconds:.1f}", flush=True)
        execution_index += 1

    loss = total_loss / total_weights
    sequence_loss = np.asarray(sequence_loss)
    sequence_delta = sequence_loss - np.asarray(baseline_sequence_loss)
    report["variants"][variant_name] = {
        "overrides": overrides,
        "loss": loss,
        "delta_loss": loss - baseline_loss,
        "sequence_delta_loss": _describe(sequence_delta),
        "fraction_sequences_improved": float(np.mean(sequence_delta < 0)),
        "errors": {
            scope: {name: accumulator.report() for name, accumulator in values.items()}
            for scope, values in errors.items()
        },
        "timing_seconds": timings,
    }
    report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
    _write_report(report_path, report)
    print(f"BAM_CACHE_DIAG variant={variant_name} done loss={loss:.6f} "
          f"dloss={loss - baseline_loss:+.6f}", flush=True)
    del variant_forward, variant_model
    jax.clear_caches()

  report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
  _write_report(report_path, report)
  print(f"BAM_CACHE_DIAG_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Run randomized, reproducible Pile eval batches and inspect BAM runtime tensors.

The production attention module only exposes raw values through the ``bam_raw``
Flax collection.  Sampling, statistics, SVDs, reporting, and persistence live
here so diagnostics can evolve without accumulating policy in attention code.

Environment controls:
  BAM_DIAG_BATCHES       Number of eval batches (default: 1).
  BAM_DIAG_TOKEN_STRIDE  Keep every Nth token in saved raw samples (default: 32).
  BAM_DIAG_OUTPUT_DIR    Local output directory (default: /tmp/bam_diagnostics).
  BAM_DIAG_SAVE_RAW      Save sampled arrays (default: 1).
  BAM_DIAG_RAW_LAYERS    Optional comma-separated layers to save, e.g. 17,23.
  BAM_DIAG_KEYS_ONLY     Return only full/local_o key transform stages and rank statistics.
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
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


_READ_PROJECTION_NAMES = frozenset(("W_R", "W_Ro", "W_lq", "W_lk", "W_beta"))
_READ_KEY_STAGES = ("pre_rms", "post_rms_pre_gate", "post_gate")
_LAYER_RE = re.compile(r"layers_(\d+)")
_EPS = 1.0e-12


def _capture_read_projections(module, method_name: str) -> bool:
  """Capture only BAM read-key projections; all other raw values use bam_raw."""
  return method_name == "__call__" and module.name in _READ_PROJECTION_NAMES


def _unwrap_sow(value: Any) -> Any:
  while isinstance(value, (tuple, list)) and len(value) == 1:
    value = value[0]
  return value


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for component in path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _group_raw_by_layer(collections: dict[str, Any]) -> dict[int, dict[str, jax.Array]]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  raw = collections.get("bam_raw", {})
  for path, value in flatten_dict(raw).items():
    layer = _layer_from_path(path)
    if layer is not None:
      grouped[layer][path[-1]] = _unwrap_sow(value)

  captured = collections.get("intermediates", {})
  for path, value in flatten_dict(captured).items():
    layer = _layer_from_path(path)
    if layer is None:
      continue
    for projection_name in _READ_PROJECTION_NAMES:
      if projection_name in path:
        grouped[layer][f"read_key_{projection_name}"] = _unwrap_sow(value)
        break
  return dict(sorted(grouped.items()))


def _sample_layer_on_device(layer_raw: dict[str, jax.Array], stride: int) -> dict[str, jax.Array]:
  sampled = {}
  for name, value in layer_raw.items():
    if name == "Mbar":
      sampled[name] = value[:, :, ::stride]
    elif value.ndim >= 2:
      sampled[name] = value[:, ::stride]
    else:
      sampled[name] = value
  return sampled


def _finite(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32).reshape(-1)
  return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float]:
  values = _finite(values)
  if values.size == 0:
    return {key: float("nan") for key in ("mean", "std", "rms", "abs_max", "p50", "p90", "p99")}
  abs_values = np.abs(values)
  return {
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "rms": float(np.sqrt(np.mean(np.square(values)))),
      "abs_max": float(np.max(abs_values)),
      "p50": float(np.percentile(values, 50)),
      "p90": float(np.percentile(values, 90)),
      "p99": float(np.percentile(values, 99)),
  }


def _norms(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  return np.linalg.norm(values, axis=-1)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
  return np.asarray(numerator, np.float32) / np.maximum(np.asarray(denominator, np.float32), _EPS)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  left = np.asarray(left, np.float32)
  right = np.asarray(right, np.float32)
  dot = np.sum(left * right, axis=-1)
  return dot / np.maximum(_norms(left) * _norms(right), _EPS)


def _row_column_balance(row_value: np.ndarray, column_value: np.ndarray) -> dict[str, Any]:
  """Compare V-from-r_row against U-from-r_col at each batch/token/head position."""
  row_norm = _norms(row_value)
  column_norm = _norms(column_value)
  valid = column_norm > _EPS
  row_rms = float(np.sqrt(np.mean(np.square(row_norm))))
  column_rms = float(np.sqrt(np.mean(np.square(column_norm))))
  return {
      "row_norm": _stats(row_norm),
      "column_norm": _stats(column_norm),
      "row_to_column": _stats(np.where(valid, row_norm / np.maximum(column_norm, _EPS), np.nan)),
      "aggregate_rms_row_to_column": row_rms / column_rms if column_rms > _EPS else None,
      "fraction_row_gt_column": float(np.mean(row_norm[valid] > column_norm[valid])) if np.any(valid) else None,
      "valid_ratio_fraction": float(np.mean(valid)),
  }


def _masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
  broadcast_mask = np.broadcast_to(mask, values.shape)
  return _stats(values[broadcast_mask])


def _route_logit_stats(
    layer_raw: dict[str, np.ndarray], token_positions: np.ndarray, segment_ids: np.ndarray
) -> dict[str, Any]:
  q_std = np.asarray(layer_raw["query_std"], np.float32)
  k_std = np.asarray(layer_raw["key_std"], np.float32)
  q_route = np.asarray(layer_raw["query_route"], np.float32)
  k_route = np.asarray(layer_raw["key_route"], np.float32)
  head_dim = q_std.shape[-1]

  q0 = np.transpose(q_std, (0, 2, 1, 3)) / math.sqrt(head_dim)
  k0 = np.transpose(k_std, (0, 2, 1, 3))
  q_delta = np.transpose(q_route - q_std, (0, 2, 1, 3)) / math.sqrt(head_dim)
  k_delta = np.transpose(k_route - k_std, (0, 2, 1, 3))

  logits_std = np.einsum("bnqd,bnkd->bnqk", q0, k0)
  logits_q_cross = np.einsum("bnqd,bnkd->bnqk", q_delta, k0)
  logits_k_cross = np.einsum("bnqd,bnkd->bnqk", q0, k_delta)
  logits_bam_bam = np.einsum("bnqd,bnkd->bnqk", q_delta, k_delta)
  logits_delta = logits_q_cross + logits_k_cross + logits_bam_bam
  logits_route = logits_std + logits_delta

  positions = np.asarray(token_positions)
  segments = np.asarray(segment_ids)
  valid = (positions[:, None, :] <= positions[:, :, None]) & (
      segments[:, :, None] == segments[:, None, :]
  )
  valid &= (segments[:, :, None] != 0) & (segments[:, None, :] != 0)
  valid = valid[:, None, :, :]

  std_rms = _masked_stats(logits_std, valid)["rms"]
  delta_rms = _masked_stats(logits_delta, valid)["rms"]
  return {
      "standard": _masked_stats(logits_std, valid),
      "q_cross": _masked_stats(logits_q_cross, valid),
      "k_cross": _masked_stats(logits_k_cross, valid),
      "bam_bam": _masked_stats(logits_bam_bam, valid),
      "delta": _masked_stats(logits_delta, valid),
      "route": _masked_stats(logits_route, valid),
      "delta_to_standard_rms": float(delta_rms / max(std_rms, _EPS)),
  }


def _matrix_stats(m_in: np.ndarray, m_out: np.ndarray, decay: float) -> dict[str, Any]:
  m_in = np.asarray(m_in, np.float32)
  m_out = np.asarray(m_out, np.float32)
  delta = m_out - decay * m_in
  axes = (-2, -1)
  in_fro = np.linalg.norm(m_in, axis=axes)
  out_fro = np.linalg.norm(m_out, axis=axes)
  delta_fro = np.linalg.norm(delta, axis=axes)
  dot = np.sum(m_in * delta, axis=axes)
  nonzero_history = in_fro > _EPS
  write_cosine = np.where(
      nonzero_history & (delta_fro > _EPS), dot / np.maximum(in_fro * delta_fro, _EPS), np.nan
  )
  delta_to_history = np.where(nonzero_history, delta_fro / np.maximum(in_fro, _EPS), np.nan)

  matrices = m_out.reshape((-1,) + m_out.shape[-2:])
  singular_values = np.linalg.svd(matrices, compute_uv=False)
  energy = np.square(singular_values)
  energy_sum = np.maximum(np.sum(energy, axis=-1), _EPS)
  probabilities = energy / energy_sum[:, None]
  stable_rank = energy_sum / np.maximum(energy[:, 0], _EPS)
  effective_rank = np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, _EPS)), axis=-1))

  return {
      "M_in_fro": _stats(in_fro),
      "M_out_fro": _stats(out_fro),
      "delta_M_fro": _stats(delta_fro),
      # Layer 0 starts from an exactly-zero M; exclude those tokens rather than
      # reporting a meaningless ratio inflated by the numerical epsilon.
      "delta_to_M_in": _stats(delta_to_history),
      "write_cosine": _stats(write_cosine),
      "stable_rank": _stats(stable_rank),
      "effective_rank": _stats(effective_rank),
      "top1_energy_fraction": _stats(probabilities[:, 0]),
      "finite_fraction": float(np.mean(np.isfinite(m_out))),
  }


def _read_stats(layer_raw: dict[str, np.ndarray]) -> dict[str, Any]:
  y_std = np.asarray(layer_raw["y_std"], np.float32)
  components = {
      "codebook": np.asarray(layer_raw["y_codebook"], np.float32),
      "full": np.asarray(layer_raw["y_full"], np.float32),
      "local_o": np.asarray(layer_raw["y_local_o"], np.float32),
  }
  y_bam = sum(components.values())
  std_norm = _norms(y_std)
  bam_norm = _norms(y_bam)
  output = {
      "standard_norm": _stats(std_norm),
      "combined_norm": _stats(bam_norm),
      "combined_to_standard": _stats(_safe_ratio(bam_norm, std_norm)),
      "combined_cosine_standard": _stats(_cosine(y_bam, y_std)),
  }
  for name, value in components.items():
    component_norm = _norms(value)
    half = value.shape[-1] // 2
    column_value = value[..., :half]
    row_value = value[..., half:]
    output[name] = {
        "norm": _stats(component_norm),
        "to_standard": _stats(_safe_ratio(component_norm, std_norm)),
        "cosine_standard": _stats(_cosine(value, y_std)),
        "U_norm": _stats(_norms(column_value)),
        "V_norm": _stats(_norms(row_value)),
        "row_column": _row_column_balance(row_value, column_value),
    }
  return output


def _full_fetch_stats(layer_raw: dict[str, np.ndarray]) -> dict[str, Any]:
  """Decompose the full-read output by fetch mode from raw Mbar and runtime keys."""
  if "Mbar" not in layer_raw or "read_key_W_R" not in layer_raw:
    return {}
  mbar = np.asarray(layer_raw["Mbar"], np.float32)                 # [b,f,t,k,v]
  read_key = np.asarray(layer_raw["read_key_W_R"], np.float32)    # [b,t,n,f,k+v]
  k = mbar.shape[-2]
  r_row, r_col = np.split(read_key, [k], axis=-1)
  y_u = np.einsum("bftkv,btnfv->btnfk", mbar, r_col)
  y_v = np.einsum("bftkv,btnfk->btnfv", mbar, r_row)
  by_fetch = np.concatenate((y_u, y_v), axis=-1)                  # [b,t,n,f,d]
  combined = np.sum(by_fetch, axis=-2)
  captured = np.asarray(layer_raw["y_full"], np.float32)
  standard = np.asarray(layer_raw["y_std"], np.float32)
  standard_norm = _norms(standard)

  output = {}
  for fetch in range(by_fetch.shape[-2]):
    value = by_fetch[..., fetch, :]
    value_norm = _norms(value)
    ratio = _safe_ratio(value_norm, standard_norm)
    output[f"fetch_{fetch}"] = {
        "norm": _stats(value_norm),
        "to_standard": _stats(ratio),
        "cosine_standard": _stats(_cosine(value, standard)),
        "per_sequence_median_to_standard": _stats(
            np.median(ratio.reshape(ratio.shape[0], -1), axis=1)
        ),
        "row_column": _row_column_balance(y_v[..., fetch, :], y_u[..., fetch, :]),
    }
  if by_fetch.shape[-2] == 2:
    norm_0 = _norms(by_fetch[..., 0, :])
    norm_1 = _norms(by_fetch[..., 1, :])
    output["fetch_1_to_fetch_0"] = _stats(_safe_ratio(norm_1, norm_0))
    output["fetch_cosine"] = _stats(_cosine(by_fetch[..., 0, :], by_fetch[..., 1, :]))
    output["sum_to_sum_of_norms"] = _stats(
        _safe_ratio(_norms(combined), norm_0 + norm_1)
    )
  reconstruction_error = combined - captured
  output["reconstruction_relative_rms"] = float(
      np.sqrt(np.mean(np.square(reconstruction_error)))
      / max(float(np.sqrt(np.mean(np.square(captured)))), _EPS)
  )
  output["reconstruction_abs_max"] = float(np.max(np.abs(reconstruction_error)))
  return output


def _key_matrix_rank_stats(
    keys: np.ndarray, *, group_name: str | None = None
) -> dict[str, Any]:
  """Rank concentration for [..., matrix_rows, key_width] runtime-key matrices."""
  keys = np.asarray(keys, np.float32)
  gram = np.einsum("...fd,...gd->...fg", keys, keys)
  eigenvalues = np.maximum(np.linalg.eigvalsh(gram)[..., ::-1], 0.0)
  total_energy = np.sum(eigenvalues, axis=-1)
  valid = total_energy > _EPS
  top1 = np.where(valid, eigenvalues[..., 0] / np.maximum(total_energy, _EPS), np.nan)
  rank1_error = np.sqrt(np.maximum(1.0 - top1, 0.0))
  if eigenvalues.shape[-1] > 1:
    second_to_first = np.sqrt(
        eigenvalues[..., 1] / np.maximum(eigenvalues[..., 0], _EPS))
  else:
    second_to_first = np.zeros_like(top1)

  batch_size = keys.shape[0]
  per_sequence = top1.reshape(batch_size, -1)
  top1_stats = _stats(top1)
  top1_finite = _finite(top1)
  top1_stats["p10"] = (
      float(np.percentile(top1_finite, 10)) if top1_finite.size else float("nan"))
  output = {
      "matrix_shape": list(keys.shape[-2:]),
      "matrix_count": int(np.prod(keys.shape[:-2])),
      "nonzero_fraction": float(np.mean(valid)),
      "top1_energy_fraction": top1_stats,
      "rank1_relative_fro_error": _stats(rank1_error),
      "second_to_first_singular_ratio": _stats(second_to_first),
      "per_sequence_median_top1": _stats(np.nanmedian(per_sequence, axis=1)),
  }
  if group_name is not None:
    groups = np.moveaxis(top1, -1, 0).reshape(top1.shape[-1], -1)
    output[f"per_{group_name}_median_top1"] = np.nanmedian(groups, axis=1).tolist()
  return output


def _read_key_rank_stats(
    layer_raw: dict[str, np.ndarray], bam_k: int
) -> dict[str, Any]:
  """Join full fetch and local_o keys and measure each transform stage by side."""
  output = {}
  for stage in _READ_KEY_STAGES:
    full = np.asarray(layer_raw[f"read_key_W_R_{stage}"], np.float32)  # [b,t,n,f,k+v]
    local_name = f"read_key_W_Ro_{stage}"
    # CombinedRead intentionally evaluates the shared key once, so there is no
    # separate W_Ro capture.  Its n_f=1 local key is exactly the W_R fetch key.
    local_o = (
        np.asarray(layer_raw[local_name], np.float32)
        if local_name in layer_raw
        else full[..., 0, :]
    )
    if full.shape[:-2] != local_o.shape[:-1] or full.shape[-1] != local_o.shape[-1]:
      raise ValueError(
          f"Incompatible full/local_o key shapes at {stage}: {full.shape}, {local_o.shape}")
    keys = np.concatenate((full, local_o[..., None, :]), axis=-2)
    output[stage] = {
        "row": _key_matrix_rank_stats(keys[..., :bam_k], group_name="head"),
        "column": _key_matrix_rank_stats(keys[..., bam_k:], group_name="head"),
    }
  return output


def _head_axis_read_key_rank_stats(
    layer_raw: dict[str, np.ndarray], bam_k: int
) -> dict[str, Any]:
  """Fix each read source and measure rank across heads; also join sources per head."""
  output = {}
  for stage in _READ_KEY_STAGES:
    full = np.asarray(layer_raw[f"read_key_W_R_{stage}"], np.float32)  # [b,t,n,f,k+v]
    local_name = f"read_key_W_Ro_{stage}"
    local_o = (
        np.asarray(layer_raw[local_name], np.float32)
        if local_name in layer_raw
        else full[..., 0, :]
    )
    if full.shape[:-2] != local_o.shape[:-1] or full.shape[-1] != local_o.shape[-1]:
      raise ValueError(
          f"Incompatible full/local_o key shapes at {stage}: {full.shape}, {local_o.shape}")

    sources = {f"fetch_{fetch}": full[..., fetch, :] for fetch in range(full.shape[-2])}
    sources["local_o"] = local_o
    stage_output = {
        source: {
            "row": _key_matrix_rank_stats(keys[..., :bam_k]),
            "column": _key_matrix_rank_stats(keys[..., bam_k:]),
        }
        for source, keys in sources.items()
    }

    # Each head is one row.  Concatenating source features asks whether heads share a
    # low-dimensional joint full-fetch/local_o key pattern, without mixing row/column keys.
    joint_row = np.concatenate([keys[..., :bam_k] for keys in sources.values()], axis=-1)
    joint_column = np.concatenate([keys[..., bam_k:] for keys in sources.values()], axis=-1)
    stage_output["joint"] = {
        "row": _key_matrix_rank_stats(joint_row),
        "column": _key_matrix_rank_stats(joint_column),
    }
    output[stage] = stage_output
  return output


def _parameter_summary(params: dict[str, Any]) -> dict[str, Any]:
  """Summarize W_R kernels only; do not retain checkpoint arrays on host."""
  output = {}
  for path, value in flatten_dict(params).items():
    layer = _layer_from_path(path)
    if layer is None or "W_R" not in path or path[-1] != "kernel":
      continue
    kernel = np.asarray(jax.device_get(value), np.float32)
    if kernel.ndim < 3:
      continue
    layer_output = {"shape": list(kernel.shape)}
    for fetch in range(kernel.shape[-2]):
      fetch_kernel = kernel[..., fetch, :]
      half = fetch_kernel.shape[-1] // 2
      reduce_axes = tuple(axis for axis in range(fetch_kernel.ndim) if axis != fetch_kernel.ndim - 2)
      layer_output[f"fetch_{fetch}"] = {
          "value": _stats(fetch_kernel),
          "fro": float(np.linalg.norm(fetch_kernel)),
          "r_row_rms": _stats(fetch_kernel[..., :half])["rms"],
          "r_col_rms": _stats(fetch_kernel[..., half:])["rms"],
          "per_head_rms": np.sqrt(np.mean(np.square(fetch_kernel), axis=reduce_axes)).tolist(),
      }
    output[f"layer_{layer:02d}"] = layer_output
  return dict(sorted(output.items()))


def _layer_summary(
    layer_raw: dict[str, np.ndarray], token_positions: np.ndarray, segment_ids: np.ndarray, decay: float
) -> dict[str, Any]:
  query_delta = np.asarray(layer_raw["query_route"], np.float32) - np.asarray(layer_raw["query_std"], np.float32)
  key_delta = np.asarray(layer_raw["key_route"], np.float32) - np.asarray(layer_raw["key_std"], np.float32)
  gate = np.asarray(layer_raw["write_gate"], np.float32)
  summary = {
      "write_gate": {
          **_stats(gate),
          "fraction_gt_0.01": float(np.mean(gate > 0.01)),
          "fraction_gt_0.1": float(np.mean(gate > 0.1)),
          "fraction_gt_0.5": float(np.mean(gate > 0.5)),
          "fraction_lt_0.01": float(np.mean(gate < 0.01)),
          "fraction_gt_0.99": float(np.mean(gate > 0.99)),
      },
      "route_vectors": {
          "query_delta_norm": _stats(_norms(query_delta)),
          "query_delta_to_standard": _stats(_safe_ratio(_norms(query_delta), _norms(layer_raw["query_std"]))),
          "key_delta_norm": _stats(_norms(key_delta)),
          "key_delta_to_standard": _stats(_safe_ratio(_norms(key_delta), _norms(layer_raw["key_std"]))),
      },
      "route_logits": _route_logit_stats(layer_raw, token_positions, segment_ids),
      "matrix": _matrix_stats(layer_raw["M_in"], layer_raw["M_out"], decay),
      "read": _read_stats(layer_raw),
      "full_fetch": _full_fetch_stats(layer_raw),
  }
  read_keys = {}
  for name, value in layer_raw.items():
    if name.startswith("read_key_"):
      read_keys[name.removeprefix("read_key_")] = {
          "value": _stats(value),
          "vector_norm": _stats(_norms(value)),
          "finite_fraction": float(np.mean(np.isfinite(value))),
      }
  summary["read_keys"] = read_keys
  return summary


def _flatten_scalars(tree: Any, prefix: str = "") -> dict[str, float]:
  output = {}
  if isinstance(tree, dict):
    for key, value in tree.items():
      child_prefix = f"{prefix}/{key}" if prefix else str(key)
      output.update(_flatten_scalars(value, child_prefix))
  elif isinstance(tree, (float, int, np.floating, np.integer)):
    output[prefix] = float(tree)
  return output


def _aggregate_batches(batch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
  by_path: dict[str, list[float]] = defaultdict(list)
  for summary in batch_summaries:
    for path, value in _flatten_scalars(summary).items():
      if np.isfinite(value):
        by_path[path].append(value)
  return {
      path: {
          "mean": float(np.mean(values)),
          "std": float(np.std(values)),
          "min": float(np.min(values)),
          "max": float(np.max(values)),
      }
      for path, values in sorted(by_path.items())
  }


def _safe_npz_name(name: str) -> str:
  return name.replace("/", "__")


def _save_sample(output_dir: Path, batch_index: int, layers: dict[int, dict[str, np.ndarray]], batch_sample: dict[str, np.ndarray]):
  arrays = {f"batch__{_safe_npz_name(name)}": value for name, value in batch_sample.items()}
  for layer, layer_raw in layers.items():
    for name, value in layer_raw.items():
      arrays[f"layer_{layer:02d}__{_safe_npz_name(name)}"] = value
  np.savez_compressed(output_dir / f"bam_raw_batch_{batch_index:02d}.npz", **arrays)


def _forward(model, config, params, batch, rng, stride, keys_only):
  rng1, aqt_rng = jax.random.split(rng)
  (xent, correct, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=["bam_raw"] if keys_only else ["bam_raw", "intermediates"],
      capture_intermediates=False if keys_only else _capture_read_projections,
  )
  mask = batch["targets_segmentation"] != 0
  total_loss = jnp.sum(xent * mask)
  total_weights = jnp.sum(mask)
  sequence_weights = jnp.sum(mask, axis=-1)
  grouped = _group_raw_by_layer(collections)
  if keys_only:
    read_key_names = {
        f"read_key_{projection}_{stage}"
        for projection in ("W_R", "W_Ro")
        for stage in _READ_KEY_STAGES
    }
    grouped = {
        layer: {
            name: value for name, value in layer_raw.items()
            if name in read_key_names
        }
        for layer, layer_raw in grouped.items()
    }
  sampled = {
      layer: _sample_layer_on_device(layer_raw, stride) for layer, layer_raw in grouped.items()
  }
  return {
      "loss": total_loss / jnp.maximum(total_weights, 1),
      "total_loss": total_loss,
      "total_weights": total_weights,
      "accuracy": correct / jnp.maximum(total_weights, 1),
      "sequence_loss": jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1),
      "sequence_weights": sequence_weights,
  }, sampled


def run(config) -> None:
  run_start = time.perf_counter()
  if not config.bam_enabled or not config.bam_diagnostics:
    raise ValueError("bam_diagnostics.py requires bam_enabled=True and bam_diagnostics=True")
  if not config.only_eval:
    raise ValueError("bam_diagnostics.py is inference-only; pass only_eval=True")

  num_batches = int(os.environ.get("BAM_DIAG_BATCHES", "1"))
  stride = int(os.environ.get("BAM_DIAG_TOKEN_STRIDE", "32"))
  output_dir = Path(os.environ.get("BAM_DIAG_OUTPUT_DIR", "/tmp/bam_diagnostics"))
  save_raw = os.environ.get("BAM_DIAG_SAVE_RAW", "1").lower() not in ("0", "false", "no")
  keys_only = os.environ.get("BAM_DIAG_KEYS_ONLY", "0").lower() in ("1", "true", "yes")
  raw_layers_text = os.environ.get("BAM_DIAG_RAW_LAYERS", "").strip()
  raw_layers = {int(layer) for layer in raw_layers_text.split(",") if layer.strip()}
  if num_batches <= 0 or stride <= 0:
    raise ValueError("BAM_DIAG_BATCHES and BAM_DIAG_TOKEN_STRIDE must be positive")
  output_dir.mkdir(parents=True, exist_ok=True)

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled; eval_interval must be positive")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )
  parameter_summary = {} if keys_only else _parameter_summary(state.params)
  setup_seconds = time.perf_counter() - run_start

  # Slice diagnostic values before they leave the compiled computation.  Returning the full
  # mutable collection at batch=32 would itself consume tens of GiB.
  compiled_forward = jax.jit(
      lambda params, batch, rng: _forward(model, config, params, batch, rng, stride, keys_only)
  )
  summaries = []
  metadata = {
      "checkpoint": config.load_parameters_path,
      "num_batches": num_batches,
      "token_stride": stride,
      "eval_batch_size": int(config.eval_per_device_batch_size * jax.local_device_count()),
      "eval_shuffle_buffer_size": config.eval_shuffle_buffer_size,
      "data_shuffle_seed": config.data_shuffle_seed,
      "save_raw": save_raw,
      "keys_only": keys_only,
      "raw_layers": sorted(raw_layers),
      "setup_seconds": setup_seconds,
      "device_count": jax.device_count(),
      "devices": [str(device) for device in jax.devices()],
  }

  for batch_index in range(num_batches):
    data_start = time.perf_counter()
    batch = next(eval_data_iterator)
    data_seconds = time.perf_counter() - data_start
    batch_rng = jax.random.fold_in(init_rng, batch_index)
    forward_start = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      batch_metrics, sampled_device = compiled_forward(state.params, batch, batch_rng)
    jax.block_until_ready((batch_metrics, sampled_device))
    forward_seconds = time.perf_counter() - forward_start

    if len(sampled_device) != config.num_decoder_layers:
      raise RuntimeError(
          f"Expected {config.num_decoder_layers} BAM layers, found {len(sampled_device)}"
      )

    transfer_start = time.perf_counter()
    sampled_host = jax.device_get(sampled_device)
    batch_sample = jax.device_get({
        "inputs": batch["inputs"][:, ::stride],
        "positions": batch["inputs_position"][:, ::stride],
        "segments": batch["inputs_segmentation"][:, ::stride],
    })
    full_inputs = np.asarray(jax.device_get(batch["inputs"]))
    transfer_seconds = time.perf_counter() - transfer_start
    token_positions = batch_sample["positions"]
    segment_ids = batch_sample["segments"]

    stats_start = time.perf_counter()
    if keys_only:
      layer_summaries = {
          f"layer_{layer:02d}": {
              "read_key_rank": _read_key_rank_stats(layer_raw, int(config.bam_k)),
              "head_axis_read_key_rank": _head_axis_read_key_rank_stats(
                  layer_raw, int(config.bam_k)
              ),
          }
          for layer, layer_raw in sampled_host.items()
      }
    else:
      layer_summaries = {
          f"layer_{layer:02d}": _layer_summary(
              layer_raw, token_positions, segment_ids, float(config.bam_lambda_decay)
          )
          for layer, layer_raw in sampled_host.items()
      }
    metric_host = jax.device_get(batch_metrics)
    metric_values = {
        key: float(value)
        for key, value in metric_host.items()
        if np.asarray(value).ndim == 0
    }
    batch_summary = {
        "batch": batch_index,
        "eval": metric_values,
        "sequence_loss": np.asarray(metric_host["sequence_loss"]).tolist(),
        "sequence_weights": np.asarray(metric_host["sequence_weights"]).astype(int).tolist(),
        "sequence_hashes": [
            hashlib.sha256(sequence.tobytes()).hexdigest()[:16] for sequence in full_inputs
        ],
        "layers": layer_summaries,
    }
    stats_seconds = time.perf_counter() - stats_start
    save_start = time.perf_counter()
    if save_raw:
      saved_layers = (
          {layer: values for layer, values in sampled_host.items() if layer in raw_layers}
          if raw_layers
          else sampled_host
      )
      _save_sample(output_dir, batch_index, saved_layers, batch_sample)
    save_seconds = time.perf_counter() - save_start
    batch_summary["timing_seconds"] = {
        "data": data_seconds,
        "forward_compile_execute": forward_seconds,
        "device_to_host": transfer_seconds,
        "host_statistics": stats_seconds,
        "raw_save": save_seconds,
    }
    summaries.append(batch_summary)

    if keys_only:
      layer23_rank = layer_summaries["layer_23"]["read_key_rank"]
      layer23_head_rank = layer_summaries["layer_23"]["head_axis_read_key_rank"]
      print(
          f"BAM_DIAG batch={batch_index} loss={metric_values['loss']:.6f} "
          + " ".join(
              f"layer23_{stage}="
              f"{layer23_rank[stage]['row']['top1_energy_fraction']['p50']:.4f}/"
              f"{layer23_rank[stage]['column']['top1_energy_fraction']['p50']:.4f}"
              for stage in _READ_KEY_STAGES),
          f"layer23_head_post_gate_joint="
          f"{layer23_head_rank['post_gate']['joint']['row']['top1_energy_fraction']['p50']:.4f}/"
          f"{layer23_head_rank['post_gate']['joint']['column']['top1_energy_fraction']['p50']:.4f}",
          flush=True,
      )
    else:
      print(
          f"BAM_DIAG batch={batch_index} loss={metric_values['loss']:.6f} "
          f"layer23_gate={layer_summaries['layer_23']['write_gate']['mean']:.4f} "
          f"layer23_read_ratio={layer_summaries['layer_23']['read']['combined_to_standard']['p50']:.4f}",
          flush=True,
      )

  report = {
      "metadata": metadata,
      "parameters": parameter_summary,
      "batches": summaries,
      "aggregate": _aggregate_batches(summaries),
  }
  report["metadata"]["total_seconds_before_report_write"] = time.perf_counter() - run_start
  report_path = output_dir / "bam_diagnostics.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"BAM_DIAG_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Measure whether V2 repeatedly writes the same BAM address.

The matrix stream is per token and accumulates across decoder layers.  For each
sampled token this probe compares a layer/head write address against addresses
from earlier layers, measures same-layer cross-head overlap separately, and
tests the delta-rule residual implied by the pre-write matrix::

  u_hat = M_in @ v / (v.T @ v)
  delta_data = u - u_hat

Only the raw write factors and gates cross the device/host boundary; all policy
statistics live in this standalone diagnostic.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

from absl import app
import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import bam_diagnostics
import exp
import pyconfig
import train


_EPS = 1.0e-12
_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
_REQUIRED = ("M_in", "y_std", "y_full", "write_gate", "read_key_P_loc_up")


class BamLlama2MediumV2DeltaRuleDiagnostics(exp.BamLlama2MediumV2):
  """Read-only V2 write-address reuse probe."""

  bam_diagnostics = True
  scan_layers = False
  eval_per_device_batch_size = 16.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_delta_rule_tb/"


exp.BamLlama2MediumV2DeltaRuleDiagnostics = BamLlama2MediumV2DeltaRuleDiagnostics


def _sample_write_tensors(layer_raw, stride):
  missing = set(_REQUIRED) - set(layer_raw)
  if missing:
    raise KeyError(f"missing write diagnostic tensors: {sorted(missing)}")
  return {name: layer_raw[name][:, ::stride] for name in _REQUIRED}


def _minimal_layer_summary(layer_raw, unused_positions, unused_segments, unused_decay):
  gate = np.asarray(layer_raw["write_gate"], np.float32)
  return {
      "write_gate": bam_diagnostics._stats(gate),  # pylint: disable=protected-access
      "read": {"combined_to_standard": {"p50": float("nan")}},
  }


# Reuse the historical diagnostic collection without adding policy to BamAttention.
bam_diagnostics._READ_PROJECTION_NAMES = (  # pylint: disable=protected-access
    bam_diagnostics._READ_PROJECTION_NAMES | frozenset(("P_loc_up",))
)
bam_diagnostics._sample_layer_on_device = _sample_write_tensors  # pylint: disable=protected-access
bam_diagnostics._layer_summary = _minimal_layer_summary  # pylint: disable=protected-access


def _rms_norm(value: np.ndarray, epsilon: float) -> np.ndarray:
  value = _to_float32(value)
  return value / np.sqrt(np.mean(np.square(value), axis=-1, keepdims=True) + epsilon)


def _to_float32(value: np.ndarray) -> np.ndarray:
  """Decode JAX bf16 saved by NumPy as an opaque two-byte scalar."""
  value = np.asarray(value)
  if value.dtype.kind == "V" and value.dtype.itemsize == 2:
    value = value.view(ml_dtypes.bfloat16)
  return value.astype(np.float32)


def _l2_normalize(value: np.ndarray) -> np.ndarray:
  return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), _EPS)


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
  weight_sum = float(np.sum(weight))
  return float(np.sum(value * weight) / weight_sum) if weight_sum > _EPS else float("nan")


def _distribution(value: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
  value = np.asarray(value, np.float32).reshape(-1)
  finite = np.isfinite(value)
  value = value[finite]
  output = {
      "count": int(value.size),
      "mean": float(np.mean(value)),
      "std": float(np.std(value)),
      "p10": float(np.percentile(value, 10)),
      "p50": float(np.percentile(value, 50)),
      "p90": float(np.percentile(value, 90)),
      "p99": float(np.percentile(value, 99)),
  }
  if weight is not None:
    output["gate_weighted_mean"] = _weighted_mean(
        value, np.asarray(weight, np.float32).reshape(-1)[finite])
  return output


def _fraction(mask: np.ndarray, weight: np.ndarray | None = None) -> float:
  mask = np.asarray(mask, np.float32).reshape(-1)
  if weight is None:
    return float(np.mean(mask))
  return _weighted_mean(mask, np.asarray(weight, np.float32).reshape(-1))


class _Store:
  def __init__(self):
    self.values: dict[str, list[np.ndarray]] = defaultdict(list)

  def add(self, **values: np.ndarray) -> None:
    for name, value in values.items():
      self.values[name].append(np.asarray(value).reshape(-1))

  def get(self, name: str) -> np.ndarray:
    return np.concatenate(self.values[name])


def _summarize(store: _Store) -> dict[str, Any]:
  gate = store.get("gate")
  output: dict[str, Any] = {
      "write_gate": _distribution(gate),
      "same_layer_max_abs_cosine": _distribution(
          store.get("same_layer_max_abs_cos"), gate),
      "same_layer_cross_token_null_max_abs_cosine": _distribution(
          store.get("same_layer_cross_token_null_max_abs_cos"), gate),
  }
  if "cross_layer_max_abs_cos" in store.values:
    cross_gate = store.get("cross_gate")
    max_abs = store.get("cross_layer_max_abs_cos")
    max_positive = store.get("cross_layer_max_positive_cos")
    pair_gate = store.get("matched_pair_gate")
    data_cos = store.get("matched_data_cos_sign_aligned")
    output["cross_layer"] = {
        "max_abs_cosine": _distribution(max_abs, cross_gate),
        "max_positive_cosine": _distribution(max_positive, cross_gate),
        "cross_token_null_max_abs_cosine": _distribution(
            store.get("cross_token_null_max_abs_cos"), cross_gate),
        "matched_data_cosine_sign_aligned": _distribution(data_cos, pair_gate),
        "thresholds": {},
    }
    for threshold in _THRESHOLDS:
      selected = max_abs >= threshold
      output["cross_layer"]["thresholds"][str(threshold)] = {
          "write_fraction": _fraction(selected),
          "cross_token_null_fraction": _fraction(
              store.get("cross_token_null_max_abs_cos") >= threshold),
          "current_gate_mass_fraction": _fraction(selected, cross_gate),
          "matched_pair_gate_mass_fraction": _fraction(selected, pair_gate),
          "matched_data_cosine_mean": (
              float(np.mean(data_cos[selected])) if np.any(selected) else float("nan")),
      }

    pred_ratio = store.get("prediction_norm_to_data")
    pred_cos = store.get("prediction_data_cosine")
    residual = store.get("delta_residual_norm_to_data")
    output["memory_prediction"] = {
        "prediction_norm_to_data": _distribution(pred_ratio, cross_gate),
        "prediction_data_cosine": _distribution(pred_cos, cross_gate),
        "delta_residual_norm_to_data": _distribution(residual, cross_gate),
        "fraction_prediction_norm_gt_0.1": _fraction(pred_ratio > 0.1, cross_gate),
        "fraction_prediction_norm_gt_0.5": _fraction(pred_ratio > 0.5, cross_gate),
        "fraction_prediction_norm_gt_1.0": _fraction(pred_ratio > 1.0, cross_gate),
        "fraction_delta_residual_lt_vanilla": _fraction(residual < 1.0, cross_gate),
        "fraction_delta_residual_lt_half": _fraction(residual < 0.5, cross_gate),
        "reuse_prediction_norm_correlation": float(np.corrcoef(max_abs, pred_ratio)[0, 1]),
        "reuse_residual_correlation": float(np.corrcoef(max_abs, residual)[0, 1]),
    }
  return output


def _load_batch(path: Path, rms_epsilon: float) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
  archive = np.load(path)
  valid = np.asarray(archive["batch__segments"]).reshape(-1) != 0
  layers = []
  layer = 0
  while f"layer_{layer:02d}__M_in" in archive:
    prefix = f"layer_{layer:02d}__"
    y_std = _to_float32(archive[prefix + "y_std"])
    y_full = _to_float32(archive[prefix + "y_full"])
    data = _rms_norm((y_std + y_full)[..., :32], rms_epsilon)
    address = _rms_norm(
        _to_float32(archive[prefix + "read_key_P_loc_up"]), rms_epsilon)
    layers.append({
        "M_in": _to_float32(archive[prefix + "M_in"]).reshape((-1, 32, 32))[valid],
        "data": data.reshape((-1,) + data.shape[-2:])[valid],
        "address": address.reshape((-1,) + address.shape[-2:])[valid],
        "gate": _to_float32(archive[prefix + "write_gate"]).reshape(
            (-1, address.shape[-2]))[valid],
    })
    layer += 1
  if layer != 24:
    raise ValueError(f"expected 24 layers in {path}, found {layer}")
  return valid, layers


def _analyze(output_dir: Path, rms_epsilon: float) -> dict[str, Any]:
  global_store = _Store()
  layer_stores = [_Store() for _ in range(24)]
  raw_files = sorted(output_dir.glob("bam_raw_batch_*.npz"))
  if not raw_files:
    raise FileNotFoundError(f"no raw diagnostic batches in {output_dir}")

  for raw_file in raw_files:
    _, layers = _load_batch(raw_file, rms_epsilon)
    previous_address: list[np.ndarray] = []
    previous_data: list[np.ndarray] = []
    previous_gate: list[np.ndarray] = []
    for layer_index, current in enumerate(layers):
      address = current["address"]
      data = current["data"]
      gate = current["gate"]
      sample_count, heads, _ = address.shape

      address_unit = _l2_normalize(address)
      same_sim = np.einsum("snv,smv->snm", address_unit, address_unit)
      diagonal = np.arange(heads)
      same_sim[:, diagonal, diagonal] = 0.0
      same_max_abs = np.max(np.abs(same_sim), axis=-1)
      null_same_sim = np.einsum(
          "snv,smv->snm", address_unit,
          np.roll(address_unit, sample_count // 2, axis=0))
      null_same_sim[:, diagonal, diagonal] = 0.0
      current_metrics = {
          "gate": gate,
          "same_layer_max_abs_cos": same_max_abs,
          "same_layer_cross_token_null_max_abs_cos": np.max(
              np.abs(null_same_sim), axis=-1),
      }

      if previous_address:
        old_address = np.concatenate(previous_address, axis=1)
        old_data = np.concatenate(previous_data, axis=1)
        old_gate = np.concatenate(previous_gate, axis=1)
        old_address_unit = _l2_normalize(old_address)
        similarity = np.einsum("snv,smv->snm", address_unit, old_address_unit)
        null_old_address = np.roll(old_address_unit, sample_count // 2, axis=0)
        null_similarity = np.einsum("snv,smv->snm", address_unit, null_old_address)
        nearest = np.argmax(np.abs(similarity), axis=-1)
        nearest_similarity = np.take_along_axis(similarity, nearest[..., None], axis=-1)[..., 0]
        sample_index = np.arange(sample_count)[:, None]
        matched_data = old_data[sample_index, nearest]
        matched_gate = old_gate[sample_index, nearest]
        matched_data *= np.where(nearest_similarity >= 0, 1.0, -1.0)[..., None]
        data_cos = np.sum(_l2_normalize(data) * _l2_normalize(matched_data), axis=-1)

        address_sq = np.sum(np.square(address), axis=-1)
        prediction = np.einsum("skv,snv->snk", current["M_in"], address)
        prediction /= np.maximum(address_sq[..., None], _EPS)
        data_norm = np.linalg.norm(data, axis=-1)
        prediction_norm = np.linalg.norm(prediction, axis=-1)
        prediction_ratio = prediction_norm / np.maximum(data_norm, _EPS)
        prediction_cos = np.sum(prediction * data, axis=-1) / np.maximum(
            prediction_norm * data_norm, _EPS)
        residual_ratio = np.linalg.norm(data - prediction, axis=-1) / np.maximum(
            data_norm, _EPS)

        current_metrics.update({
            "cross_gate": gate,
            "cross_layer_max_abs_cos": np.abs(nearest_similarity),
            "cross_layer_max_positive_cos": np.max(similarity, axis=-1),
            "cross_token_null_max_abs_cos": np.max(np.abs(null_similarity), axis=-1),
            "matched_pair_gate": gate * matched_gate,
            "matched_data_cos_sign_aligned": data_cos,
            "prediction_norm_to_data": prediction_ratio,
            "prediction_data_cosine": prediction_cos,
            "delta_residual_norm_to_data": residual_ratio,
        })

      layer_stores[layer_index].add(**current_metrics)
      global_store.add(**current_metrics)
      previous_address.append(address)
      previous_data.append(data)
      previous_gate.append(gate)

  diagnostic_report = json.loads((output_dir / "bam_diagnostics.json").read_text())
  return {
      "metadata": {
          **diagnostic_report["metadata"],
          "raw_batches": len(raw_files),
          "write_semantics": "U=data, V=address; same token accumulates M across layers",
          "prediction": "u_hat = M_in @ rms(v) / ||rms(v)||^2",
          "cross_layer_match": "max absolute cosine over all heads in earlier layers",
      },
      "aggregate": _summarize(global_store),
      "layers": {
          f"layer_{layer:02d}": _summarize(store)
          for layer, store in enumerate(layer_stores)
      },
  }


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  output_dir = Path(os.environ.get("BAM_DIAG_OUTPUT_DIR", "/tmp/bam_diagnostics"))
  analyze_only = os.environ.get("BAM_DIAG_ANALYZE_ONLY", "0").lower() in (
      "1", "true", "yes")
  if not analyze_only:
    bam_diagnostics.run(config)
  report = _analyze(output_dir, float(config.normalization_layer_epsilon))
  report_path = output_dir / "delta_rule_write_reuse.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"DELTA_RULE_WRITE_REUSE_DONE report={report_path}", flush=True)


if __name__ == "__main__":
  app.run(main)

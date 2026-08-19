"""Offline analysis for head_rank_diagnostics.py artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


_HEADS = 16
_LAYERS = 24
_RANKS = (1, 2, 4, 8, 12)
_EPS = 1.0e-30


def _load_capture(directory: Path, model_kind: str) -> dict[str, Any]:
  paths = sorted(directory.glob(f"{model_kind}_head_rank_batch_*.npz"))
  if not paths:
    raise ValueError(f"no {model_kind} capture batches in {directory}")
  values: dict[str, list[np.ndarray]] = {}
  hashes = []
  valid = []
  losses = []
  for path in paths:
    with np.load(path) as data:
      hashes.extend(data["sequence_hashes"].tolist())
      valid.append(data["valid"].astype(bool))
      losses.append(float(data["loss"]))
      for name in data.files:
        if name in ("sequence_hashes", "valid", "loss"):
          continue
        values.setdefault(name, []).append(np.asarray(data[name], np.float64))
  merged = {}
  for name, arrays in values.items():
    if name.endswith("__sequence_gram"):
      merged[name] = np.concatenate(arrays, axis=1)
    elif "__local_" in name:
      merged[name] = np.concatenate(arrays, axis=1)
    else:
      raise ValueError(f"unknown artifact {name}")
  return {
      "values": merged,
      "hashes": hashes,
      "valid": np.concatenate(valid, axis=0),
      "losses": losses,
      "paths": [str(path) for path in paths],
  }


def _center_gram(gram: np.ndarray) -> np.ndarray:
  center = np.eye(_HEADS) - np.ones((_HEADS, _HEADS)) / _HEADS
  return center @ gram @ center


def _spectrum(gram: np.ndarray) -> dict[str, Any]:
  gram = (gram + gram.T) * 0.5
  eigenvalues = np.linalg.eigvalsh(gram)[::-1]
  eigenvalues = np.maximum(eigenvalues, 0)
  total = max(float(eigenvalues.sum()), _EPS)
  p = eigenvalues / total
  cumulative = np.cumsum(p)
  output = {
      f"energy_top_{rank}": float(cumulative[min(rank, _HEADS) - 1])
      for rank in _RANKS
  }
  for threshold in (0.90, 0.95, 0.99):
    output[f"r{int(threshold * 100)}"] = int(
        np.searchsorted(cumulative, threshold) + 1)
  nonzero = p[p > 0]
  output["effective_rank"] = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
  output["stable_rank"] = float(total / max(float(eigenvalues[0]), _EPS))
  output["eigenvalue_fraction"] = p.tolist()
  return output


def _local_metrics(eigenvalues: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
  # eigenvalues: [sequence, query, head]
  selected = eigenvalues[valid]
  total = np.maximum(selected.sum(axis=-1, keepdims=True), _EPS)
  cumulative = np.cumsum(selected / total, axis=-1)
  output = {}
  for rank in _RANKS:
    values = cumulative[:, min(rank, _HEADS) - 1]
    output[f"energy_top_{rank}"] = {
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }
  for threshold in (0.90, 0.95, 0.99):
    ranks = np.argmax(cumulative >= threshold, axis=-1) + 1
    output[f"r{int(threshold * 100)}"] = {
        "p10": float(np.quantile(ranks, 0.10)),
        "median": float(np.median(ranks)),
        "p90": float(np.quantile(ranks, 0.90)),
    }
  return output


def _analyze_dataset(
    values: dict[str, np.ndarray], valid: np.ndarray, name: str
) -> dict[str, Any]:
  sequence_gram = values[f"{name}__sequence_gram"]
  local_eigenvalues = values[f"{name}__local_eigenvalues"]
  local_centered = values[f"{name}__local_centered_eigenvalues"]
  cosine = values[f"{name}__local_cosine_mean"]
  cosine_abs = values[f"{name}__local_cosine_abs_mean"]
  output = {"layers": []}
  for layer in range(_LAYERS):
    gram = sequence_gram[layer].sum(axis=0)
    layer_valid = valid
    output["layers"].append({
        "layer": layer,
        "global": _spectrum(gram),
        "global_centered": _spectrum(_center_gram(gram)),
        "local": _local_metrics(local_eigenvalues[layer], layer_valid),
        "local_centered": _local_metrics(local_centered[layer], layer_valid),
        "local_pairwise_cosine_mean": float(np.mean(cosine[layer][layer_valid])),
        "local_pairwise_abs_cosine_mean": float(
            np.mean(cosine_abs[layer][layer_valid])),
    })
  return output


def _heldout_retention(
    basis_sequence_gram: np.ndarray,
    target_sequence_gram: np.ndarray,
    train_count: int,
) -> list[dict[str, Any]]:
  output = []
  for layer in range(_LAYERS):
    fit = basis_sequence_gram[layer, :train_count].sum(axis=0)
    test = target_sequence_gram[layer, train_count:].sum(axis=0)
    fit = (fit + fit.T) * 0.5
    test = (test + test.T) * 0.5
    _, vectors = np.linalg.eigh(fit)
    vectors = vectors[:, ::-1]
    total = max(float(np.trace(test)), _EPS)
    layer_output = {"layer": layer}
    for rank in _RANKS:
      basis = vectors[:, :rank]
      layer_output[f"rank_{rank}"] = float(
          np.trace(basis.T @ test @ basis) / total)
    output.append(layer_output)
  return output


def analyze(bam_dir: Path, mha_dir: Path) -> dict[str, Any]:
  bam = _load_capture(bam_dir, "bam")
  mha = _load_capture(mha_dir, "mha")
  if bam["hashes"] != mha["hashes"]:
    raise ValueError("BAM and MHA cohorts differ")
  if not np.array_equal(bam["valid"], mha["valid"]):
    raise ValueError("BAM and MHA valid-position masks differ")
  values = {**bam["values"], **mha["values"]}
  dataset_names = sorted({name.split("__", 1)[0] for name in values})
  datasets = {
      name: _analyze_dataset(values, bam["valid"], name)
      for name in dataset_names
  }

  train_count = len(bam["hashes"]) // 2
  cross_validation = {}
  for side in ("row", "col"):
    key_name = f"bam_{side}_key_post_gate"
    key_gram = values[f"{key_name}__sequence_gram"]
    for target_suffix in ("key_post_gate", "native_read", "residual"):
      target_name = f"bam_{side}_{target_suffix}"
      target_gram = values[f"{target_name}__sequence_gram"]
      cross_validation[f"{key_name}_basis_to_{target_name}"] = _heldout_retention(
          key_gram, target_gram, train_count)
  for name in ("bam_mha_residual", "mha_mha_residual", "bam_fetch_residual"):
    gram = values[f"{name}__sequence_gram"]
    cross_validation[f"{name}_self_basis"] = _heldout_retention(
        gram, gram, train_count)

  return {
      "metadata": {
          "bam_dir": str(bam_dir),
          "mha_dir": str(mha_dir),
          "sequences": len(bam["hashes"]),
          "train_sequences": train_count,
          "heldout_sequences": len(bam["hashes"]) - train_count,
          "sequence_hashes": bam["hashes"],
          "bam_mean_batch_loss": float(np.mean(bam["losses"])),
          "mha_mean_batch_loss": float(np.mean(mha["losses"])),
          "space_policy": {
              "bam_native": "shared U/V coordinates",
              "cross_model": "per-head W_O contribution in residual R^D",
              "mha_native": "intentionally omitted",
          },
      },
      "datasets": datasets,
      "heldout_fixed_basis_retention": cross_validation,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bam-dir", type=Path, required=True)
  parser.add_argument("--mha-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = analyze(args.bam_dir, args.mha_dir)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(args.output)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Compare two residual-attribution runs on the same saved cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENTS = (
    "mlp",
    "mha",
    "bam_col_self",
    "bam_col_cross",
    "bam_row_self",
    "bam_row_cross",
)
GROUPS = {
    "mlp": (0,),
    "mha": (1,),
    "bam": (2, 3, 4, 5),
    "bam_col": (2, 3),
    "bam_row": (4, 5),
    "bam_self": (2, 4),
    "bam_cross": (3, 5),
}


def _load(directory: Path) -> dict[str, np.ndarray]:
  paths = sorted(directory.glob("residual_attribution_batch_*.npz"))
  if not paths:
    raise FileNotFoundError(f"no residual attribution batches in {directory}")
  keys = (
      "energy", "contribution_normalized", "endpoint_loss", "sequence_hashes",
      "inputs", "targets", "inputs_position", "inputs_segmentation",
      "targets_segmentation",
  )
  output = {}
  for key in keys:
    values = []
    for path in paths:
      with np.load(path) as data:
        values.append(np.asarray(data[key]))
    output[key] = np.concatenate(values, axis=0)
  return output


def _assert_paired(base: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> None:
  for key in (
      "sequence_hashes", "inputs", "targets", "inputs_position",
      "inputs_segmentation", "targets_segmentation",
  ):
    if not np.array_equal(base[key], candidate[key]):
      raise ValueError(f"cohort mismatch for {key}")


def _bootstrap_mean_delta(
    base: np.ndarray, candidate: np.ndarray, seed: int, draws: int
) -> list[float]:
  delta = np.asarray(candidate - base, dtype=np.float64)
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, delta.size, size=(draws, delta.size))
  means = np.mean(delta[indices], axis=1)
  return np.quantile(means, (0.025, 0.975)).tolist()


def _summary(
    base: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, object]:
  base = np.asarray(base, dtype=np.float64)
  candidate = np.asarray(candidate, dtype=np.float64)
  return {
      "base_mean": float(np.mean(base)),
      "candidate_mean": float(np.mean(candidate)),
      "delta_mean": float(np.mean(candidate - base)),
      "delta_bootstrap_95ci": _bootstrap_mean_delta(
          base, candidate, seed, draws),
  }


def compare(
    base_dir: Path,
    candidate_dir: Path,
    seed: int,
    draws: int,
) -> dict[str, object]:
  base = _load(base_dir)
  candidate = _load(candidate_dir)
  _assert_paired(base, candidate)
  if base["energy"].shape != candidate["energy"].shape:
    raise ValueError(
        f"attribution shape mismatch: {base['energy'].shape} vs "
        f"{candidate['energy'].shape}")

  report = {
      "base_dir": str(base_dir),
      "candidate_dir": str(candidate_dir),
      "samples": int(base["energy"].shape[0]),
      "layers": int(base["energy"].shape[1]),
      "component_order": list(COMPONENTS),
      "endpoint_loss": _summary(
          base["endpoint_loss"], candidate["endpoint_loss"], seed, draws),
      "groups": {},
  }
  for group_index, (name, component_indices) in enumerate(GROUPS.items()):
    indices = np.asarray(component_indices)
    base_energy_layer = np.sum(base["energy"][:, :, indices], axis=2)
    candidate_energy_layer = np.sum(candidate["energy"][:, :, indices], axis=2)
    base_value_layer = np.sum(
        base["contribution_normalized"][:, :, indices], axis=2)
    candidate_value_layer = np.sum(
        candidate["contribution_normalized"][:, :, indices], axis=2)
    base_energy = np.sum(base_energy_layer, axis=1)
    candidate_energy = np.sum(candidate_energy_layer, axis=1)
    base_value = np.sum(base_value_layer, axis=1)
    candidate_value = np.sum(candidate_value_layer, axis=1)
    report["groups"][name] = {
        "energy": _summary(
            base_energy, candidate_energy, seed + 2 * group_index + 1, draws),
        "normalized_contribution": _summary(
            base_value, candidate_value, seed + 2 * group_index + 2, draws),
        "ratio_of_mean_contribution_to_mean_energy": {
            "base": float(np.mean(base_value) / np.mean(base_energy)),
            "candidate": float(
                np.mean(candidate_value) / np.mean(candidate_energy)),
        },
        "mean_energy_by_layer": {
            "base": np.mean(base_energy_layer, axis=0).tolist(),
            "candidate": np.mean(candidate_energy_layer, axis=0).tolist(),
            "delta": np.mean(
                candidate_energy_layer - base_energy_layer, axis=0).tolist(),
        },
        "mean_normalized_contribution_by_layer": {
            "base": np.mean(base_value_layer, axis=0).tolist(),
            "candidate": np.mean(candidate_value_layer, axis=0).tolist(),
            "delta": np.mean(
                candidate_value_layer - base_value_layer, axis=0).tolist(),
        },
    }
  return report


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("base_dir", type=Path)
  parser.add_argument("candidate_dir", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--bootstrap-seed", type=int, default=20260904)
  parser.add_argument("--bootstrap-draws", type=int, default=20_000)
  args = parser.parse_args()
  report = compare(
      args.base_dir, args.candidate_dir, args.bootstrap_seed,
      args.bootstrap_draws)
  text = json.dumps(report, indent=2, sort_keys=True)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()

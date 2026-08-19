"""Merge sharded BAM readout-attribution artifacts and analyze side asymmetry."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


def _load_driver():
  path = Path(__file__).with_name("readout_attribution.py")
  spec = importlib.util.spec_from_file_location("readout_attribution_driver", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _side_relationship(files: list[Path], driver) -> dict:
  sums = {
      name: 0.0
      for name in (
          "col2", "row2", "col_row", "col_abs", "row_abs", "mixed_abs",
          "both_abs", "closure_abs", "reinforced_abs", "opposed_abs")
  }
  samples = {name: [] for name in ("col", "row", "mixed")}
  for path in files:
    with np.load(path) as data:
      valid = data["valid"].astype(bool)
      components = {
          name: np.asarray(data[f"attr_sumloss_{name}"], np.float64)
          for name in ("both", "col", "row", "mixed")
      }
    closure = (
        components["both"] - components["col"]
        - components["row"] - components["mixed"])
    for layer in range(driver._LAYERS):
      layer_valid = valid
      both = components["both"][layer][layer_valid].reshape(-1)
      col = components["col"][layer][layer_valid].reshape(-1)
      row = components["row"][layer][layer_valid].reshape(-1)
      mixed = components["mixed"][layer][layer_valid].reshape(-1)
      sums["col2"] += float(np.dot(col, col))
      sums["row2"] += float(np.dot(row, row))
      sums["col_row"] += float(np.dot(col, row))
      sums["col_abs"] += float(np.sum(np.abs(col)))
      sums["row_abs"] += float(np.sum(np.abs(row)))
      sums["mixed_abs"] += float(np.sum(np.abs(mixed)))
      sums["both_abs"] += float(np.sum(np.abs(both)))
      sums["closure_abs"] += float(np.sum(np.abs(closure[layer][layer_valid])))
      agreement_weight = np.minimum(np.abs(col), np.abs(row))
      sums["reinforced_abs"] += float(np.sum(
          agreement_weight[(col * row) >= 0]))
      sums["opposed_abs"] += float(np.sum(
          agreement_weight[(col * row) < 0]))
      select = np.linspace(0, col.size - 1, min(1000, col.size), dtype=np.int64)
      samples["col"].append(col[select])
      samples["row"].append(row[select])
      samples["mixed"].append(mixed[select])

  col_sample = np.concatenate(samples["col"])
  row_sample = np.concatenate(samples["row"])
  mixed_sample = np.concatenate(samples["mixed"])
  overlap = sums["reinforced_abs"] + sums["opposed_abs"]
  return {
      "col_to_row_absolute_mass_ratio": sums["col_abs"] / max(sums["row_abs"], 1e-12),
      "col_row_cosine": sums["col_row"] / max(
          np.sqrt(sums["col2"] * sums["row2"]), 1e-12),
      "col_row_sampled_pearson": float(np.corrcoef(col_sample, row_sample)[0, 1]),
      "col_row_sampled_spearman": driver._correlations(
          col_sample, row_sample, sample_cap=col_sample.size)["spearman"],
      "opposed_fraction_of_overlap_mass": sums["opposed_abs"] / max(overlap, 1e-12),
      "mixed_to_both_absolute_mass_ratio": sums["mixed_abs"] / max(sums["both_abs"], 1e-12),
      "closure_absolute_error_over_both": sums["closure_abs"] / max(sums["both_abs"], 1e-12),
      "sampled_mixed": driver._distribution(mixed_sample),
  }


def _p2_side_balance(files: list[Path], driver) -> dict:
  report = {}
  for family in ("fetch", "fetch_permuted_alpha", "local_q", "local_k"):
    row_width = driver._C if family.startswith("fetch") else driver._V
    energies = {"col": [], "row": []}
    controls = {
        side: {metric: [] for metric in (
            "read_key_rms", "head_mix_abs", "effective_read_strength")}
        for side in ("col", "row")
    }
    signed_mix = {side: [] for side in ("col", "row")}
    for path in files:
      with np.load(path) as data:
        for side in ("col", "row"):
          energies[side].append(driver._to_float32(
              data[f"{family}_{side}__actual_output_norm2"]))
          for metric in controls[side]:
            key = f"{family}_{side}__{metric}"
            if key in data:
              controls[side][metric].append(driver._to_float32(data[key]))
          mix_key = f"{family}_{side}__head_mix_signed"
          if mix_key in data:
            signed_mix[side].append(driver._to_float32(data[mix_key]))
    col = np.concatenate(energies["col"], axis=1)[1:]
    row = np.concatenate(energies["row"], axis=1)[1:]
    family_report = {
        "col_output_energy_fraction": float(
            np.sum(col) / max(np.sum(col) + np.sum(row), 1e-12)),
        "col_to_row_output_energy_ratio": float(
            np.sum(col) / max(np.sum(row), 1e-12)),
        "col_to_row_output_energy_per_coordinate_ratio": float(
            (np.sum(col) / driver._K)
            / max(np.sum(row) / row_width, 1e-12)),
        "col_row_output_energy_correlation": driver._correlations(col, row),
        "col_output_energy_fraction_by_layer": [
            float(np.sum(col[layer]) / max(
                np.sum(col[layer]) + np.sum(row[layer]), 1e-12))
            for layer in range(col.shape[0])
        ],
        "col_output_energy_fraction_by_head": [
            float(np.sum(col[..., head]) / max(
                np.sum(col[..., head]) + np.sum(row[..., head]), 1e-12))
            for head in range(col.shape[-1])
        ],
        "col_output_energy_fraction_by_depth_band": {
            label: float(np.sum(col[start:stop]) / max(
                np.sum(col[start:stop]) + np.sum(row[start:stop]), 1e-12))
            for label, start, stop in (
                ("layers_01_08", 0, 8),
                ("layers_09_16", 8, 16),
                ("layers_17_23", 16, 23),
            )
        },
    }
    for metric in controls["col"]:
      if controls["col"][metric] and controls["row"][metric]:
        col_metric = np.concatenate(controls["col"][metric], axis=1)[1:]
        row_metric = np.concatenate(controls["row"][metric], axis=1)[1:]
        family_report[f"col_to_row_{metric}_mean_ratio"] = float(
            np.mean(col_metric) / max(np.mean(row_metric), 1e-12))
        family_report[f"col_row_{metric}_correlation"] = driver._correlations(
            col_metric, row_metric)
    if signed_mix["col"] and signed_mix["row"]:
      col_mix = np.concatenate(signed_mix["col"], axis=1)[1:]
      row_mix = np.concatenate(signed_mix["row"], axis=1)[1:]
      family_report["col_head_mix_signed"] = driver._distribution(col_mix)
      family_report["row_head_mix_signed"] = driver._distribution(row_mix)
      family_report["col_row_head_mix_signed_correlation"] = driver._correlations(
          col_mix, row_mix)
    report[family] = family_report
  return report


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("artifact_root", type=Path)
  parser.add_argument("output_json", type=Path)
  args = parser.parse_args()
  driver = _load_driver()

  attribution_files = sorted(args.artifact_root.rglob("attribution_batch_*.npz"))
  p2_files = sorted(args.artifact_root.rglob("p2_batch_*.npz"))
  if len(attribution_files) != 64 or len(p2_files) != 64:
    raise ValueError(
        f"expected 64 P1 and 64 P2 batches, got "
        f"{len(attribution_files)} and {len(p2_files)}")
  if len({path.name for path in attribution_files}) != 64:
    raise ValueError("duplicate or missing P1 batch indices")
  if len({path.name for path in p2_files}) != 64:
    raise ValueError("duplicate or missing P2 batch indices")

  with tempfile.TemporaryDirectory(prefix="bam-readside-index-") as temp:
    index = Path(temp)
    for path in attribution_files + p2_files:
      (index / path.name).symlink_to(path.resolve())
    report = {
        "artifact_root": str(args.artifact_root.resolve()),
        "batch_count": 64,
        "sequence_count": 128,
        "p1": driver._analyze_p1(index),
        "p1_side_relationship": _side_relationship(attribution_files, driver),
        "p2": driver._analyze_p2(index),
        "p2_side_balance": _p2_side_balance(p2_files, driver),
    }
  args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(args.output_json)


if __name__ == "__main__":
  main()

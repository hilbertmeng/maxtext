#!/usr/bin/env python3
"""Summarize fixed-batch fetched-read gate-bin diagnostics across layer bands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LAYER_BANDS = {
    "early_1_7": range(1, 8),
    "middle_8_15": range(8, 16),
    "late_16_23": range(16, 24),
}
SIDES = ("row", "column")
THRESHOLDS = (
    0.02, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99)


def _layer(report, layer):
  return report["layers"][f"layer_{layer:02d}"]


def _mean(values):
  return sum(values) / len(values)


def _finite(value):
  return value is not None and math.isfinite(value)


def _aggregate_side(report, layers, side):
  layer_data = [_layer(report, layer) for layer in layers]
  gate_stats = [entry[side]["sigmoid_gate"] for entry in layer_data]
  gate_mean = _mean([stats["mean"] for stats in gate_stats])
  gate_second_moment = _mean([
      stats["std"] ** 2 + stats["mean"] ** 2 for stats in gate_stats
  ])

  bins = []
  total_read_energy = 0.0
  for entry in layer_data:
    for item in entry["gate_binned_readout"][side]:
      if _finite(item["read_rms"]):
        total_read_energy += item["fraction"] * item["read_rms"] ** 2

  for bin_idx in range(len(report["metadata"]["gate_bin_edges"]) - 1):
    items = [entry["gate_binned_readout"][side][bin_idx] for entry in layer_data]
    population = sum(item["fraction"] for item in items)
    read_energy = sum(
        item["fraction"] * item["read_rms"] ** 2
        for item in items if _finite(item["read_rms"]))
    std_energy = sum(
        item["fraction"] * item["std_slice_rms"] ** 2
        for item in items if _finite(item["std_slice_rms"]))
    full_std_energy = sum(
        item["fraction"] * item["read_rms"] ** 2
        / item["read_to_full_std_frobenius"] ** 2
        for item in items
        if (_finite(item["read_rms"])
            and _finite(item["read_to_full_std_frobenius"])
            and item["read_to_full_std_frobenius"] > 0))
    population_fraction = population / len(items)
    energy_fraction = read_energy / total_read_energy if total_read_energy else 0.0
    gate_population = sum(
        item["fraction"] * item["gate_mean"]
        for item in items if _finite(item["gate_mean"]))
    sensitivity_population = sum(
        item["fraction"] * item.get("sigmoid_derivative_mean", 0.0)
        for item in items if _finite(item.get("sigmoid_derivative_mean")))
    bins.append({
        "lo": items[0]["lo"],
        "hi": items[0]["hi"],
        "count": sum(item.get("count", 0) for item in items),
        "population_fraction": population_fraction,
        "gate_mean": gate_population / population if population else None,
        "read_energy_fraction": energy_fraction,
        "energy_enrichment": (
            energy_fraction / population_fraction if population_fraction else None),
        "sigmoid_derivative_mean": (
            sensitivity_population / population if population else None),
        "read_to_std_slice_rms": (
            math.sqrt(read_energy / std_energy) if std_energy else None),
        "side_read_to_full_y_std_frobenius": (
            math.sqrt(read_energy / full_std_energy)
            if full_std_energy else None),
    })

  thresholds = {}
  bin_edges = {item["lo"] for item in bins} | {bins[-1]["hi"]}
  for threshold in THRESHOLDS:
    if threshold not in bin_edges:
      continue
    selected = [item for item in bins if item["lo"] >= threshold]
    population = sum(item["population_fraction"] for item in selected)
    energy = sum(item["read_energy_fraction"] for item in selected)
    thresholds[f"gt_{threshold:g}"] = {
        "population_fraction": population,
        "read_energy_fraction": energy,
        "energy_enrichment": energy / population if population else None,
    }

  return {
      "gate_mean": gate_mean,
      "gate_std": math.sqrt(max(gate_second_moment - gate_mean ** 2, 0.0)),
      "gate_rms": math.sqrt(gate_second_moment),
      "gate_p05_mean_across_layers": _mean([stats["p05"] for stats in gate_stats]),
      "gate_p50_mean_across_layers": _mean([stats["p50"] for stats in gate_stats]),
      "gate_p95_mean_across_layers": _mean([stats["p95"] for stats in gate_stats]),
      "thresholds": thresholds,
      "bins": bins,
  }


def summarize(reports):
  hashes = [report["metadata"]["sequence_hashes"] for report in reports.values()]
  if any(run_hashes != hashes[0] for run_hashes in hashes[1:]):
    raise ValueError("diagnostic runs did not use the same sequence cohort")

  result = {
      "metadata": {
          "num_sequences": len(hashes[0]),
          "sequence_hashes": hashes[0],
          "gate_bin_edges": next(iter(reports.values()))["metadata"]["gate_bin_edges"],
      },
      "runs": {},
  }
  for run_name, report in reports.items():
    run_result = {
        "exp_class": report["metadata"]["exp_class"],
        "checkpoint": report["metadata"]["checkpoint"],
        "bands": {},
    }
    for band_name, layer_range in LAYER_BANDS.items():
      layers = list(layer_range)
      layer_data = [_layer(report, layer) for layer in layers]
      run_result["bands"][band_name] = {
          "mbar_rms": _mean([entry["mbar"]["rms"] for entry in layer_data]),
          "bam_to_std_frobenius": _mean([
              entry["readout"]["bam_to_std_frobenius"] for entry in layer_data
          ]),
          **{
              side: _aggregate_side(report, layers, side)
              for side in SIDES
          },
      }
    result["runs"][run_name] = run_result
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--control", type=Path, required=True)
  parser.add_argument("--p05", type=Path, required=True)
  parser.add_argument("--p50", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  reports = {}
  for run_name in ("control", "p05", "p50"):
    with getattr(args, run_name).open() as source:
      reports[run_name] = json.load(source)
  result = summarize(reports)
  rendered = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    args.output.write_text(rendered + "\n")
  else:
    print(rendered)


if __name__ == "__main__":
  main()

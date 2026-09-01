#!/usr/bin/env python3
"""Analyze depth-amplitude fetched-read compensation from local TB scalars.

The recorded ``pre_gate`` scalar is ``RMS(M) * mean(key_scale)`` rather than
an actual pre-gate readout RMS.  This tool combines it with the exact gate
second moment reconstructed from the recorded mean/std:

  gate_rms = sqrt(E[g^2]) = sqrt(mean(g)^2 + std(g)^2)
  input_strength_proxy = RMS(M) * mean(key_scale) * gate_rms

It then compares that proxy with the contracted readout RMS and y_BAM/y_std.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_ROOT = Path("/data0/xd/tensorboard_logs")


class Scalars:

  def __init__(self, path: Path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    self._accumulator = accumulator
    self.tags = set(accumulator.Tags()["scalars"])
    self._cache = {}

  def at(self, tag: str, step: int) -> float:
    if tag not in self._cache:
      self._cache[tag] = self._accumulator.Scalars(tag)
    events = self._cache[tag]
    return float(min(events, key=lambda event: abs(event.step - step)).value)


def _tag(group: str, side: str, layer: int, stat: str) -> str:
  return f"bam/{group}/{side}/layer_{layer:03d}/{stat}"


def _layer_record(scalars: Scalars, step: int, layer: int) -> dict:
  record = {"step": step, "layer": layer}
  m_rms = scalars.at(
      f"bam/fetched_read_health/layer_{layer:03d}/m_rms", step)
  record["m_rms"] = m_rms
  record["y_bam_over_y_std"] = scalars.at(
      f"bam/fetched_read_health/layer_{layer:03d}/y_bam_over_y_std", step)
  for side in ("row", "col"):
    mean = scalars.at(_tag("fetched_read_gate", side, layer, "mean"), step)
    std = scalars.at(_tag("fetched_read_gate", side, layer, "std"), step)
    gate_rms = math.hypot(mean, std)
    scale_m_proxy = scalars.at(
        _tag("fetched_read_pre_gate", side, layer, "rms"), step)
    input_strength = scale_m_proxy * gate_rms
    output_rms = scalars.at(
        _tag("fetched_read_output", side, layer, "rms"), step)
    record[side] = {
        "gate_mean": mean,
        "gate_std": std,
        "gate_rms": gate_rms,
        "gate_frac_lt_005": scalars.at(
            _tag("fetched_read_gate", side, layer, "frac_lt_005"), step),
        "gate_frac_gt_095": scalars.at(
            _tag("fetched_read_gate", side, layer, "frac_gt_095"), step),
        "m_times_scale_proxy": scale_m_proxy,
        "input_strength_proxy": input_strength,
        "output_rms": output_rms,
        "contraction_gain": output_rms / max(input_strength, 1e-12),
    }
  return record


def _correlation(records: list[dict], x, y) -> float:
  xs = np.asarray([x(record) for record in records], np.float64)
  ys = np.asarray([y(record) for record in records], np.float64)
  if np.std(xs) == 0 or np.std(ys) == 0:
    return float("nan")
  return float(np.corrcoef(xs, ys)[0, 1])


def _summarize(records: list[dict]) -> dict:
  summary = {}
  for side in ("row", "col"):
    summary[side] = {
        "corr_m_gate_mean": _correlation(
            records, lambda r: r["m_rms"], lambda r: r[side]["gate_mean"]),
        "corr_m_gate_rms": _correlation(
            records, lambda r: r["m_rms"], lambda r: r[side]["gate_rms"]),
        "corr_input_proxy_output": _correlation(
            records, lambda r: r[side]["input_strength_proxy"],
            lambda r: r[side]["output_rms"]),
        "corr_output_y_ratio": _correlation(
            records, lambda r: r[side]["output_rms"],
            lambda r: r["y_bam_over_y_std"]),
    }
  return summary


def analyze(run: str, root: Path, steps: list[int], num_layers: int) -> dict:
  scalars = Scalars(root / run)
  by_step = {}
  summaries = {}
  for step in steps:
    records = [_layer_record(scalars, step, layer) for layer in range(num_layers)]
    by_step[str(step)] = records
    summaries[str(step)] = _summarize(records[1:])
  return {"run": run, "steps": by_step, "layer_correlations": summaries}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("runs", nargs="+")
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--steps", default="200,1000,2800,6000,10000,13400")
  parser.add_argument("--num-layers", type=int, default=24)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  steps = [int(step) for step in args.steps.split(",")]
  report = json.dumps([
      analyze(run, args.root, steps, args.num_layers) for run in args.runs
  ], indent=2, sort_keys=True)
  if args.output is None:
    print(report)
  else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")


if __name__ == "__main__":
  main()

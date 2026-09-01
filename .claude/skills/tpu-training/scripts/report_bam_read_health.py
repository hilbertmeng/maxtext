#!/usr/bin/env python3
"""Summarize BAM fetched-read TensorBoard health metrics as time series."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LOCAL_TB_ROOT = Path("/data0/xd/tensorboard_logs")


def _parse_steps(value: str) -> list[int]:
  return sorted({int(item) for item in value.split(",") if item.strip()})


def _parse_bands(value: str) -> list[tuple[str, range]]:
  bands = []
  for item in value.split(","):
    lo, hi = (int(part) for part in item.split("-", 1))
    bands.append((f"L{lo}-{hi}", range(lo, hi + 1)))
  return bands


class Scalars:

  def __init__(self, event_dir):
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    self._accumulator = accumulator
    self.tags = set(accumulator.Tags()["scalars"])
    self._cache = {}

  def values(self, tag: str):
    if tag not in self._cache:
      self._cache[tag] = self._accumulator.Scalars(tag)
    return self._cache[tag]

  def at(self, tag: str, step: int) -> float:
    events = self.values(tag)
    event = min(events, key=lambda item: abs(item.step - step))
    return float(event.value)

  def band_mean(self, template: str, step: int, layers: range) -> float | None:
    values = []
    for layer in layers:
      tag = template.format(layer=layer)
      if tag in self.tags:
        values.append(self.at(tag, step))
    return float(np.mean(values)) if values else None


def _rounded(value, digits=5):
  return None if value is None or not math.isfinite(value) else round(value, digits)


def _collect(scalars: Scalars, steps: list[int], bands, num_layers: int):
  result = {"global": [], "bands": []}
  raw_grad_tag = "learning/raw_grad_norm"
  raw_grad_events = scalars.values(raw_grad_tag)
  for step in steps:
    wr_norms = []
    for layer in range(num_layers):
      tag = (
          f"raw_grads/decoder/layers_{layer}/block/self_attention/W_R/kernel"
      )
      if tag in scalars.tags:
        wr_norms.append(scalars.at(tag, step))
    result["global"].append({
        "step": step,
        "raw_grad": _rounded(scalars.at(raw_grad_tag, step)),
        "wr_grad_l2": _rounded(math.sqrt(sum(value * value for value in wr_norms))),
        "clip_fraction_to_step": _rounded(np.mean([
            event.value > 1.0 for event in raw_grad_events if event.step <= step
        ]), 4),
    })

  for band_name, layers in bands:
    for step in steps:
      row = {"step": step, "band": band_name}
      for side in ("row", "col"):
        amp_ratios = []
        for layer in layers:
          tag = (
              f"bam/fetched_read_amplitude/{side}/layer_{layer:03d}/mean"
          )
          if tag in scalars.tags:
            initial = scalars.at(tag, 0)
            amp_ratios.append(scalars.at(tag, step) / initial)
        row[f"amplitude_ratio_{side}"] = _rounded(
            np.mean(amp_ratios) if amp_ratios else None)
        gate_prefix = f"bam/fetched_read_gate/{side}/layer_{{layer:03d}}"
        row[f"gate_mean_{side}"] = _rounded(
            scalars.band_mean(gate_prefix + "/mean", step, layers))
        row[f"gate_std_{side}"] = _rounded(
            scalars.band_mean(gate_prefix + "/std", step, layers))
        row[f"gate_frac_lt_005_{side}"] = _rounded(
            scalars.band_mean(gate_prefix + "/frac_lt_005", step, layers))
        row[f"gate_frac_gt_095_{side}"] = _rounded(
            scalars.band_mean(gate_prefix + "/frac_gt_095", step, layers))
        row[f"pre_gate_rms_{side}"] = _rounded(scalars.band_mean(
            f"bam/fetched_read_pre_gate/{side}/layer_{{layer:03d}}/rms",
            step, layers))
        row[f"output_rms_{side}"] = _rounded(scalars.band_mean(
            f"bam/fetched_read_output/{side}/layer_{{layer:03d}}/rms",
            step, layers))
      health_prefix = "bam/fetched_read_health/layer_{layer:03d}"
      row["m_rms"] = _rounded(
          scalars.band_mean(health_prefix + "/m_rms", step, layers))
      row["y_bam_over_y_std"] = _rounded(
          scalars.band_mean(health_prefix + "/y_bam_over_y_std", step, layers))
      result["bands"].append(row)
  return result


def _print_text(run: str, result) -> None:
  print(f"RUN={run}")
  print("step raw_grad W_R_grad clip_frac")
  for row in result["global"]:
    print(row["step"], row["raw_grad"], row["wr_grad_l2"],
          row["clip_fraction_to_step"])
  print("\nstep band aR/a0 aC/a0 gateR(mean/std/<.05/>.95) "
        "gateC(mean/std/<.05/>.95) preR/preC outR/outC M_rms yBAM/ySTD")
  for row in result["bands"]:
    def gate(side):
      return "/".join(str(row[f"gate_{field}_{side}"]) for field in (
          "mean", "std", "frac_lt_005", "frac_gt_095"))
    print(
        row["step"], row["band"], row["amplitude_ratio_row"],
        row["amplitude_ratio_col"], gate("row"), gate("col"),
        f'{row["pre_gate_rms_row"]}/{row["pre_gate_rms_col"]}',
        f'{row["output_rms_row"]}/{row["output_rms_col"]}',
        row["m_rms"], row["y_bam_over_y_std"])


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("run")
  parser.add_argument(
      "--event-dir",
      help="Local or gs:// event directory; defaults to the local synced RUN")
  parser.add_argument("--steps", default="0,200,1000,2000,2800")
  parser.add_argument("--bands", default="0-7,8-15,16-23")
  parser.add_argument("--num-layers", type=int, default=24)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  event_dir = args.event_dir or DEFAULT_LOCAL_TB_ROOT / args.run
  result = _collect(
      Scalars(event_dir), _parse_steps(args.steps), _parse_bands(args.bands),
      args.num_layers)
  if args.json:
    print(json.dumps({"run": args.run, **result}, indent=2, sort_keys=True))
  else:
    _print_text(args.run, result)


if __name__ == "__main__":
  main()

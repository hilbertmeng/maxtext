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

  def band_ratio_mean(
      self, numerator: str, denominator: str, step: int, layers: range
  ) -> float | None:
    values = []
    for layer in layers:
      numerator_tag = numerator.format(layer=layer)
      denominator_tag = denominator.format(layer=layer)
      if numerator_tag in self.tags and denominator_tag in self.tags:
        values.append(
            self.at(numerator_tag, step)
            / max(self.at(denominator_tag, step), 1e-12))
    return float(np.mean(values)) if values else None


def _rounded(value, digits=5):
  return None if value is None or not math.isfinite(value) else round(value, digits)


def _collect(scalars: Scalars, steps: list[int], bands, num_layers: int):
  result = {"global": [], "bands": [], "local_qk": []}
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
        amp_cvs = []
        amp_min_over_means = []
        amp_max_over_means = []
        for layer in layers:
          mean_tag = (
              f"bam/fetched_read_amplitude/{side}/layer_{layer:03d}/mean"
          )
          if mean_tag in scalars.tags:
            mean = scalars.at(mean_tag, step)
            initial = scalars.at(mean_tag, 0)
            amp_ratios.append(mean / initial)
            prefix = mean_tag.removesuffix("/mean")
            std_tag, min_tag, max_tag = (
                prefix + "/std", prefix + "/min", prefix + "/max")
            if all(tag in scalars.tags for tag in (std_tag, min_tag, max_tag)):
              denominator = max(abs(mean), 1e-12)
              amp_cvs.append(scalars.at(std_tag, step) / denominator)
              amp_min_over_means.append(
                  scalars.at(min_tag, step) / denominator)
              amp_max_over_means.append(
                  scalars.at(max_tag, step) / denominator)
        row[f"amplitude_ratio_{side}"] = _rounded(
            np.mean(amp_ratios) if amp_ratios else None)
        row[f"amplitude_cv_{side}"] = _rounded(
            np.mean(amp_cvs) if amp_cvs else None)
        row[f"amplitude_min_over_mean_{side}"] = _rounded(
            np.mean(amp_min_over_means) if amp_min_over_means else None)
        row[f"amplitude_max_over_mean_{side}"] = _rounded(
            np.mean(amp_max_over_means) if amp_max_over_means else None)
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
      row["removed_std_over_std"] = _rounded(scalars.band_mean(
          health_prefix + "/removed_std_over_std", step, layers))
      for name in ("kept_std", "merged"):
        row[f"{name}_over_std"] = _rounded(scalars.band_ratio_mean(
            health_prefix + f"/{name}_rms",
            health_prefix + "/y_std_rms", step, layers))
      result["bands"].append(row)

  local_qk_probe = "bam/local_qk/local_q/row/layer_000/rank_sum_mean"
  if local_qk_probe in scalars.tags:
    for band_name, layers in bands:
      for step in steps:
        row = {"step": step, "band": band_name}
        for use_point in ("local_q", "local_k"):
          for side in ("row", "col"):
            prefix = (
                f"bam/local_qk/{use_point}/{side}/layer_{{layer:03d}}")
            key = f'{use_point.removeprefix("local_")}_{side}'
            for stat in (
                "rank_sum_mean", "rank_sum_std", "rank_sum_min",
                "rank_sum_max", "rank_sum_head_std", "dominant_rank_share"):
              row[f"{key}_{stat}"] = _rounded(
                  scalars.band_mean(prefix + f"/{stat}", step, layers))
            row[f"{key}_rank_mean_bins"] = [
                _rounded(scalars.band_mean(
                    prefix + f"/rank_mean_bin_{lo:02d}_{lo + 10:02d}",
                    step, layers))
                for lo in range(0, 100, 10)
            ]
        health = "bam/local_qk/health/layer_{layer:03d}"
        row["q_bam_over_std"] = _rounded(
            scalars.band_mean(health + "/q_bam_over_std", step, layers))
        row["k_bam_over_std"] = _rounded(
            scalars.band_mean(health + "/k_bam_over_std", step, layers))
        result["local_qk"].append(row)
  return result


def _paired_metric(run_value, base_value):
  if run_value is None or base_value is None:
    return None
  return {
      "run": run_value,
      "base": base_value,
      "delta": _rounded(run_value - base_value),
      "ratio": _rounded(run_value / base_value) if base_value != 0 else None,
  }


def _compare(run_result, base_result):
  comparison = {"global": [], "bands": []}
  base_global = {row["step"]: row for row in base_result["global"]}
  for run_row in run_result["global"]:
    base_row = base_global.get(run_row["step"])
    if base_row is None:
      continue
    row = {"step": run_row["step"]}
    for name in ("raw_grad", "wr_grad_l2", "clip_fraction_to_step"):
      row[name] = _paired_metric(run_row.get(name), base_row.get(name))
    comparison["global"].append(row)

  base_bands = {
      (row["step"], row["band"]): row for row in base_result["bands"]
  }
  for run_row in run_result["bands"]:
    base_row = base_bands.get((run_row["step"], run_row["band"]))
    if base_row is None:
      continue
    row = {"step": run_row["step"], "band": run_row["band"]}
    for name in run_row:
      if name not in ("step", "band"):
        row[name] = _paired_metric(run_row.get(name), base_row.get(name))
    comparison["bands"].append(row)
  return comparison


def _print_text(run: str, result) -> None:
  print(f"RUN={run}")
  print("step raw_grad W_R_grad clip_frac")
  for row in result["global"]:
    print(row["step"], row["raw_grad"], row["wr_grad_l2"],
          row["clip_fraction_to_step"])
  print("\nstep band aR/a0(cv/min/max) aC/a0(cv/min/max) "
        "gateR(mean/std/<.05/>.95) "
        "gateC(mean/std/<.05/>.95) preR/preC outR/outC M_rms "
        "yBAM/ySTD removedSTD/STD keptSTD/STD merged/STD")
  for row in result["bands"]:
    def gate(side):
      return "/".join(str(row[f"gate_{field}_{side}"]) for field in (
          "mean", "std", "frac_lt_005", "frac_gt_095"))
    print(
        row["step"], row["band"],
        "/".join(str(row[f"amplitude_{field}_row"]) for field in (
            "ratio", "cv", "min_over_mean", "max_over_mean")),
        "/".join(str(row[f"amplitude_{field}_col"]) for field in (
            "ratio", "cv", "min_over_mean", "max_over_mean")),
        gate("row"), gate("col"),
        f'{row["pre_gate_rms_row"]}/{row["pre_gate_rms_col"]}',
        f'{row["output_rms_row"]}/{row["output_rms_col"]}',
        row["m_rms"], row["y_bam_over_y_std"],
        row["removed_std_over_std"], row["kept_std_over_std"],
        row["merged_over_std"])
  if result["local_qk"]:
    print("\nstep band q/std k/std point-side="
          "sum_mean/sum_std/head_std/dominance bins(rank_mean:0-.1..9-1)")
    for row in result["local_qk"]:
      print(row["step"], row["band"], row["q_bam_over_std"],
            row["k_bam_over_std"])
      for key in ("q_row", "q_col", "k_row", "k_col"):
        summary = "/".join(str(row[f"{key}_{stat}"]) for stat in (
            "rank_sum_mean", "rank_sum_std", "rank_sum_head_std",
            "dominant_rank_share"))
        bins = "/".join(str(value) for value in row[
            f"{key}_rank_mean_bins"])
        print(f"  {key} {summary} {bins}")


def _format_pair(pair):
  if pair is None:
    return "--"
  return f'{pair["run"]}/{pair["base"]}({pair["delta"]:+g})'


def _print_comparison(run: str, base_run: str, comparison) -> None:
  print(f"RUN={run} BASE={base_run} values=RUN/BASE(delta)")
  print("step raw_grad W_R_grad clip_frac")
  for row in comparison["global"]:
    print(row["step"], _format_pair(row["raw_grad"]),
          _format_pair(row["wr_grad_l2"]),
          _format_pair(row["clip_fraction_to_step"]))
  print("\nstep band gateR_mean gateC_mean gateR_<.05 gateC_<.05 "
        "gateR_>.95 gateC_>.95 M_rms yBAM/ySTD")
  for row in comparison["bands"]:
    print(
        row["step"], row["band"], _format_pair(row["gate_mean_row"]),
        _format_pair(row["gate_mean_col"]),
        _format_pair(row["gate_frac_lt_005_row"]),
        _format_pair(row["gate_frac_lt_005_col"]),
        _format_pair(row["gate_frac_gt_095_row"]),
        _format_pair(row["gate_frac_gt_095_col"]),
        _format_pair(row["m_rms"]),
        _format_pair(row["y_bam_over_y_std"]))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("run")
  parser.add_argument(
      "--event-dir",
      help="Local or gs:// event directory; defaults to the local synced RUN")
  parser.add_argument("--base-run")
  parser.add_argument(
      "--base-event-dir",
      help="Baseline event directory; defaults to the local synced BASE RUN")
  parser.add_argument("--steps", default="0,200,1000,2000,2800")
  parser.add_argument("--bands", default="0-7,8-15,16-23")
  parser.add_argument("--num-layers", type=int, default=24)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  event_dir = args.event_dir or DEFAULT_LOCAL_TB_ROOT / args.run
  result = _collect(
      Scalars(event_dir), _parse_steps(args.steps), _parse_bands(args.bands),
      args.num_layers)
  base_result = comparison = None
  if args.base_run:
    base_event_dir = (
        args.base_event_dir or DEFAULT_LOCAL_TB_ROOT / args.base_run)
    base_result = _collect(
        Scalars(base_event_dir), _parse_steps(args.steps),
        _parse_bands(args.bands), args.num_layers)
    comparison = _compare(result, base_result)
  if args.json:
    payload = {"run": args.run, **result}
    if args.base_run:
      payload.update({
          "base_run": args.base_run,
          "base": base_result,
          "comparison": comparison,
      })
    print(json.dumps(payload, indent=2, sort_keys=True))
  elif args.base_run:
    _print_comparison(args.run, args.base_run, comparison)
  else:
    _print_text(args.run, result)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Summarize BAM XPlane scopes from one or more TPU trace.json.gz files."""

import argparse
import collections
import glob
import gzip
import json


OUTER = {
    "write_m": "bam/write_m",
    "mix_alpha": "bam/mix_alpha",
    "fetch_m": "bam/fetch_m",
    "local_qk": "bam/read_local_m_for_qk",
    "fetched": "bam/read_fetched_m",
}


def metric(event):
  args = event.get("args", {})
  return (
      float(event.get("dur", 0.0)) / 1000.0,
      float(args.get("model_flops", 0.0)) / 1e12,
      float(args.get("bytes_accessed", 0.0)) / 1e9,
  )


def add(dst, value):
  for index, item in enumerate(value):
    dst[index] += item


def classify_local(op):
  if "read_gate_projection" in op or "W_lq_gate" in op or "W_lk_gate" in op:
    return "gate_projection"
  if ("read_head_mix_projection" in op or "W_lq_head_mix" in op
      or "W_lk_head_mix" in op):
    return "head_mix_projection"
  if ("read_head_mix_expand" in op or "btk,btn->bntk" in op
      or "btv,btn->bntv" in op):
    return "head_mix_expand"
  if "read_head_mix_transform" in op:
    return "head_mix_transform"
  if ("read_m_contract" in op or "btkv,btv->btk" in op
      or "btkv,btk->btv" in op):
    return "read_m"
  if ("read_key_projection" in op or "/W_lq/" in op or "/W_lk/" in op):
    return "key_projection"
  if "read_key_transform" in op:
    return "key_transform"
  return "other"


def classify_fetched(op):
  if "read_gate_projection" in op or "W_R_gate" in op:
    return "gate_projection"
  if ("read_m_contract" in op or "contract_1a" in op or "contract_1b" in op
      or "bftkv,btnfv->btnk" in op or "bftkv,btnfk->btnv" in op):
    return "read_m"
  if "read_key_projection" in op or "/W_R/" in op:
    return "key_projection"
  if "read_key_transform" in op:
    return "key_transform"
  return "other"


def classify_write(op):
  if "bam/write_outer" in op:
    return "outer"
  return "other"


def summarize(path):
  with gzip.open(path, "rt") as stream:
    events = json.load(stream)["traceEvents"]
  device_pids = {
      event["pid"] for event in events
      if event.get("ph") == "M" and event.get("name") == "process_name"
      and str(event.get("args", {}).get("name", "")).startswith("/device:TPU:")
  }
  steps = collections.Counter(
      event["pid"] for event in events
      if event.get("pid") in device_pids and event.get("ph") == "X"
      and str(event.get("name", "")).startswith("jit_train_step(")
  )
  per_device = {}
  for pid, step_count in steps.items():
    buckets = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    step_ms = 0.0
    for event in events:
      if event.get("pid") != pid or event.get("ph") != "X":
        continue
      if str(event.get("name", "")).startswith("jit_train_step("):
        step_ms += float(event.get("dur", 0.0)) / 1000.0
        continue
      op = str(event.get("args", {}).get("tf_op", ""))
      value = metric(event)
      add(buckets["all_xla_ops"], value)
      for name, scope in OUTER.items():
        if scope not in op:
          continue
        add(buckets[name], value)
        if name == "write_m":
          add(buckets[f"write.{classify_write(op)}"], value)
        elif name == "local_qk":
          add(buckets[f"local.{classify_local(op)}"], value)
        elif name == "fetched":
          add(buckets[f"fetched.{classify_fetched(op)}"], value)
        break
      if "/attention/qk_logits/" in op:
        add(buckets["mha_qk_logits"], value)
    bam_total = [0.0, 0.0, 0.0]
    for name in OUTER:
      add(bam_total, buckets[name])
    buckets["bam_total"] = bam_total
    buckets["non_bam_xla_ops"] = [
        buckets["all_xla_ops"][index] - bam_total[index] for index in range(3)
    ]
    per_device[pid] = (step_ms / step_count, {
        name: [item / step_count for item in value]
        for name, value in buckets.items()
    })
  return per_device


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("traces", nargs="+")
  args = parser.parse_args()
  paths = []
  for pattern in args.traces:
    paths.extend(glob.glob(pattern))
  devices = []
  for path in paths:
    devices.extend(summarize(path).values())
  names = sorted({name for _, buckets in devices for name in buckets})
  print(f"traces={len(paths)} devices={len(devices)}")
  step_values = [step for step, _ in devices]
  print(f"step_ms={sum(step_values) / len(step_values):.3f} "
        f"range={min(step_values):.3f}..{max(step_values):.3f}")
  print("bucket\tms\tTF\tGB")
  for name in names:
    values = []
    for _, buckets in devices:
      values.append(buckets.get(name, [0.0, 0.0, 0.0]))
    means = [sum(value[i] for value in values) / len(values) for i in range(3)]
    print(f"{name}\t{means[0]:.3f}\t{means[1]:.5f}\t{means[2]:.3f}")


if __name__ == "__main__":
  main()

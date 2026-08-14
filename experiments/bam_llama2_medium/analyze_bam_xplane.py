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
    "mix_fetch": "bam/mix_fetch_m",
    "compress_abs_v": "bam/compress_abs_v_cache",
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
  if "local_qk_packed_projection" in op or "W_local_qk_packed" in op:
    return "packed_projection"
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
  if "read_key_transform" in op:
    return "key_transform"
  if ("read_m_contract" in op or "contract_1a" in op or "contract_1b" in op
      or "bftkv,btnfv->btnk" in op or "bftkv,btnfk->btnv" in op):
    return "read_m"
  if "/bam/read_fetched_m/reduce_sum" in op:
    return "read_m"
  if "read_key_projection" in op or "/W_R/" in op:
    return "key_projection"
  return "other"


def classify_write(op):
  if "/P_loc_down/" in op:
    return "ploc_down"
  if "/P_loc_up/" in op:
    return "ploc_up"
  if "/W_gw/" in op:
    return "gate_projection"
  if "bam/write_outer" in op:
    return "outer"
  return "other"


def classify_mix(op):
  if "/fetch_head_mix/" in op or "/bam/mix_alpha_projection/" in op:
    return "weight_projection"
  if ("/bam/mix_alpha_diagonal/" in op or "scatter" in op or "gather" in op
      or "select_n" in op):
    return "diagonal_update"
  # Every remaining op in this deliberately narrow scope belongs to the selected
  # alpha-head contraction, including layout transforms and reduction epilogues.
  return "alpha_contraction"


def classify_fetch(op):
  if "compress_abs_v_cache" in op:
    return "source_compression"
  if "bfts,bskv->bftkv" in op or "bcs,bskv->bckv" in op:
    return "temporal_contraction"
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
      hlo_name = str(event.get("name", "")).lower()
      value = metric(event)
      # A scanned layer appears as a device-side while parent whose duration,
      # FLOPs, and bytes already include the nested body kernels. Keep the
      # wrapper visible, but exclude it from additive leaf-work totals.
      if hlo_name.startswith("while."):
        add(buckets["kernel.control_wrapper"], value)
        continue
      add(buckets["all_xla_ops"], value)
      if "tpu_flash_attention/" in op:
        add(buckets["mha_flash"], value)
        if "pallas_call" in op:
          add(buckets["mha_flash.pallas"], value)
        elif "transpose" in op or "copy" in hlo_name:
          add(buckets["mha_flash.layout"], value)
        else:
          add(buckets["mha_flash.other"], value)
      if "/QChunk_0/" in op:
        add(buckets["mha_qchunk"], value)
      if "self_attention._query_chunk_shared_full_read/" in op:
        add(buckets["bam_qchunk"], value)
      if "self_attention.query_projection/query/" in op:
        add(buckets["mha_projection.q"], value)
      if "self_attention.kv_projection/key/" in op:
        add(buckets["mha_projection.k"], value)
      if "self_attention.kv_projection/value/" in op:
        add(buckets["mha_projection.v"], value)
      if "self_attention.out_projection/out/" in op:
        add(buckets["mha_projection.o"], value)
      if "/mlp/" in op:
        add(buckets["transformer.mlp"], value)
      if "layer_norm/" in op:
        add(buckets["transformer.norm"], value)
      if "/lm_head/" in op:
        add(buckets["transformer.lm_head"], value)
      if "pallas_call" in hlo_name:
        add(buckets["kernel.pallas"], value)
      elif "copy" in hlo_name:
        add(buckets["kernel.copy"], value)
      elif "fusion" in hlo_name:
        add(buckets["kernel.fusion"], value)
      elif "all-reduce" in hlo_name or "collective" in hlo_name:
        add(buckets["kernel.collective"], value)
      else:
        add(buckets["kernel.other"], value)
      for name, scope in OUTER.items():
        if scope not in op:
          continue
        add(buckets[name], value)
        if name == "write_m":
          add(buckets[f"write.{classify_write(op)}"], value)
        elif name == "mix_alpha":
          add(buckets[f"mix.{classify_mix(op)}"], value)
        elif name == "fetch_m":
          add(buckets[f"fetch.{classify_fetch(op)}"], value)
        elif name == "local_qk":
          add(buckets[f"local.{classify_local(op)}"], value)
        elif name == "fetched":
          add(buckets[f"fetched.{classify_fetched(op)}"], value)
        break
      if "/attention/qk_logits/" in op:
        add(buckets["mha_qk_logits"], value)
      if "/attention/softmax/" in op:
        add(buckets["mha_softmax"], value)
      if "/attention/av/" in op:
        add(buckets["mha_av"], value)
    bam_total = [0.0, 0.0, 0.0]
    for name in OUTER:
      add(bam_total, buckets[name])
    buckets["bam_total"] = bam_total
    if "mha_qchunk" in buckets:
      buckets["mha_qchunk.other"] = [
          buckets["mha_qchunk"][index]
          - buckets["mha_qk_logits"][index]
          - buckets["mha_softmax"][index]
          - buckets["mha_av"][index]
          for index in range(3)
      ]
    if "bam_qchunk" in buckets:
      buckets["bam_qchunk.other"] = [
          buckets["bam_qchunk"][index]
          - buckets["mha_qk_logits"][index]
          - buckets["mha_softmax"][index]
          - buckets["mha_av"][index]
          - buckets["mix_alpha"][index]
          - buckets["mix_fetch"][index]
          - buckets["compress_abs_v"][index]
          - buckets["fetch_m"][index]
          - buckets["fetched"][index]
          for index in range(3)
      ]
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

"""Per-layer structural diagnostics for block-RoPE on V2 C256.

The runner captures standard Q/K/V projections and the LocalQK readouts through
Flax intermediates.  It keeps diagnostic logic outside BamAttention and returns
only reduced per-layer/head statistics to the host.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator
from v2_c256_rope_gate_diagnostics import (  # pylint: disable=unused-import
    BamLlama2MediumV2C256RopeGateDiagnostics,
)


_LAYER_RE = re.compile(r"layers_(\d+)")
_PROJECTIONS = frozenset(("query", "key", "value"))
_BUCKET_NAMES = (
    "d0", "d1_4", "d5_16", "d17_64", "d65_256", "d257_1024", "d1025_plus"
)
_BUCKET_BOUNDS = ((0, 0), (1, 4), (5, 16), (17, 64), (65, 256),
                  (257, 1024), (1025, None))
_EPS = 1.0e-12


def _unwrap(value: Any) -> Any:
  while isinstance(value, (tuple, list)) and len(value) == 1:
    value = value[0]
  return value


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for component in path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _capture_rope_inputs(module, method_name: str) -> bool:
  return (
      method_name == "_read_local_qk"
      or (method_name == "__call__" and module.name in _PROJECTIONS)
  )


def _stack_captures(collections, num_layers: int) -> dict[str, jax.Array]:
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  for path, raw in flatten_dict(collections.get("intermediates", {})).items():
    layer = _layer_from_path(path)
    if layer is None:
      continue
    value = _unwrap(raw)
    if "_read_local_qk" in path:
      if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"unexpected LocalQK capture at {path}: {type(value)}")
      grouped[layer]["q_bam"], grouped[layer]["k_bam"] = value
      continue
    names = _PROJECTIONS.intersection(path)
    if len(names) == 1:
      grouped[layer][next(iter(names))] = value

  expected = {"query", "key", "value", "q_bam", "k_bam"}
  if set(grouped) != set(range(num_layers)):
    raise ValueError(f"captured layers differ: {sorted(grouped)}")
  for layer, values in grouped.items():
    if set(values) != expected:
      raise ValueError(
          f"layer {layer} captures differ: {sorted(expected ^ set(values))}")
  return {
      name: jnp.stack([grouped[layer][name] for layer in range(num_layers)])
      for name in sorted(expected)
  }


def _rope(x, positions, min_timescale, max_timescale):
  width = x.shape[-1]
  if width % 2:
    raise ValueError(f"RoPE width must be even, got {width}")
  half = width // 2
  fraction = 2 * jnp.arange(half, dtype=jnp.float32) / width
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  phase = positions[:, :, None, None].astype(jnp.float32) / timescale
  sin = jnp.sin(phase).astype(x.dtype)
  cos = jnp.cos(phase).astype(x.dtype)
  first, second = jnp.split(x, 2, axis=-1)
  return jnp.concatenate((first * cos - second * sin,
                          second * cos + first * sin), axis=-1)


def _block_rope(x, positions, split, min_timescale, max_timescale):
  return jnp.concatenate((
      _rope(x[..., :split], positions, min_timescale, max_timescale),
      _rope(x[..., split:], positions, min_timescale, max_timescale),
  ), axis=-1)


def _slice_norm_ratio(bam, standard, start, end):
  bam2 = jnp.sum(jnp.square(bam[..., start:end].astype(jnp.float32)),
                 axis=(0, 1, 3))
  std2 = jnp.sum(jnp.square(standard[..., start:end].astype(jnp.float32)),
                 axis=(0, 1, 3))
  return jnp.sqrt(bam2 / jnp.maximum(std2, _EPS))


def _bucket_reduce(value, valid, distance):
  outputs = {"signed_mean": [], "abs_mean": [], "rms": []}
  for lower, upper in _BUCKET_BOUNDS:
    bucket = distance >= lower
    if upper is not None:
      bucket &= distance <= upper
    mask = valid & bucket[None, :, :]
    mask = mask[:, None]
    count = jnp.maximum(jnp.sum(mask), 1)
    masked = jnp.where(mask, value, 0.0)
    outputs["signed_mean"].append(jnp.sum(masked, axis=(0, 2, 3)) / count)
    outputs["abs_mean"].append(
        jnp.sum(jnp.abs(masked), axis=(0, 2, 3)) / count)
    outputs["rms"].append(
        jnp.sqrt(jnp.sum(jnp.square(masked), axis=(0, 2, 3)) / count))
  return {name: jnp.stack(values, axis=-1) for name, values in outputs.items()}


def _attention_delta(q_current, k_current, q_target, k_target, value,
                     valid, lambd):
  q = q_current + lambd * (q_target - q_current)
  k = k_current + lambd * (k_target - k_current)
  scale = jnp.asarray(q.shape[-1] ** -0.5, jnp.float32)
  logits = jnp.einsum(
      "bqhd,bshd->bhqs", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
  current_logits = jnp.einsum(
      "bqhd,bshd->bhqs", q_current.astype(jnp.float32),
      k_current.astype(jnp.float32)) * scale
  mask = valid[:, None]
  floor = jnp.finfo(jnp.float32).min
  log_p = jax.nn.log_softmax(jnp.where(mask, current_logits, floor), axis=-1)
  log_q = jax.nn.log_softmax(jnp.where(mask, logits, floor), axis=-1)
  p = jnp.exp(log_p)
  p_target = jnp.exp(log_q)
  query_valid = jnp.any(valid, axis=-1)[:, None, :]
  count = jnp.maximum(jnp.sum(query_valid, axis=(0, 2)), 1)
  kl = jnp.sum(jnp.where(query_valid, jnp.sum(p * (log_p - log_q), axis=-1), 0),
               axis=(0, 2)) / count
  y = jnp.einsum("bhqs,bshd->bqhd", p, value.astype(jnp.float32))
  y_target = jnp.einsum("bhqs,bshd->bqhd", p_target, value.astype(jnp.float32))
  rel = jnp.sqrt(jnp.sum(jnp.square(y_target - y), axis=-1)) / jnp.maximum(
      jnp.sqrt(jnp.sum(jnp.square(y), axis=-1)), _EPS)
  rel = jnp.transpose(rel, (0, 2, 1))
  rel = jnp.sum(jnp.where(query_valid, rel, 0), axis=(0, 2)) / count
  return {"attention_kl": kl, "output_relative_norm": rel}


def _layer_metrics(q_std, k_std, value, q_bam, k_bam, positions, segments,
                   bam_k, bam_width, min_timescale, max_timescale,
                   query_indices):
  q_std_rope = _rope(q_std, positions, min_timescale, max_timescale)
  k_std_rope = _rope(k_std, positions, min_timescale, max_timescale)
  q_std_block = _block_rope(
      q_std, positions, bam_width, min_timescale, max_timescale)
  k_std_block = _block_rope(
      k_std, positions, bam_width, min_timescale, max_timescale)
  q_bam_block = _block_rope(
      q_bam, positions, bam_width, min_timescale, max_timescale)
  k_bam_block = _block_rope(
      k_bam, positions, bam_width, min_timescale, max_timescale)

  norm_ratio = {}
  for name, bam, standard in (("q", q_bam, q_std), ("k", k_bam, k_std)):
    norm_ratio[f"{name}_column_read"] = _slice_norm_ratio(
        bam, standard, 0, bam_k)
    norm_ratio[f"{name}_row_read"] = _slice_norm_ratio(
        bam, standard, bam_k, bam_width)
    norm_ratio[f"{name}_total"] = _slice_norm_ratio(
        bam, standard, 0, bam_width)
    norm_ratio[f"{name}_tail_leakage_rms"] = jnp.sqrt(jnp.mean(
        jnp.square(bam[..., bam_width:].astype(jnp.float32)), axis=(0, 1, 3)))

  q_index = query_indices
  q_seg = segments[:, q_index]
  key_index = jnp.arange(k_std.shape[1])
  valid = (
      (q_seg[:, :, None] != 0)
      & (q_seg[:, :, None] == segments[:, None, :])
      & (key_index[None, None, :] <= q_index[None, :, None])
  )
  distance = q_index[:, None] - key_index[None, :]

  def gather_query(x):
    return x[:, q_index]

  scale = jnp.asarray(q_std.shape[-1] ** -0.5, jnp.float32)
  current_parts = {
      "std_std": (gather_query(q_std_rope), k_std_rope),
      "std_bam": (gather_query(q_std_rope), k_bam),
      "bam_std": (gather_query(q_bam), k_std_rope),
      "bam_bam": (gather_query(q_bam), k_bam),
  }
  block_parts = {
      "std_std": (gather_query(q_std_block), k_std_block),
      "std_bam": (gather_query(q_std_block), k_bam_block),
      "bam_std": (gather_query(q_bam_block), k_std_block),
      "bam_bam": (gather_query(q_bam_block), k_bam_block),
  }
  contribution = {"current": {}, "block": {}}
  for family, parts in (("current", current_parts), ("block", block_parts)):
    for name, (q_part, k_part) in parts.items():
      logits = jnp.einsum(
          "bqhd,bshd->bhqs", q_part.astype(jnp.float32),
          k_part.astype(jnp.float32)) * scale
      contribution[family][name] = _bucket_reduce(logits, valid, distance)

  q_current = gather_query(q_std_rope + q_bam)
  k_current = k_std_rope + k_bam
  q_block = gather_query(q_std_block + q_bam_block)
  k_block = k_std_block + k_bam_block
  counterfactual = {
      "lambda_0.1": _attention_delta(
          q_current, k_current, q_block, k_block, value, valid, 0.1),
      "lambda_1.0": _attention_delta(
          q_current, k_current, q_block, k_block, value, valid, 1.0),
  }
  return {
      "norm_ratio": norm_ratio,
      "logit_contribution": contribution,
      "counterfactual": counterfactual,
  }


def _forward(model, config, params, batch, rng, query_count):
  dropout_rng, params_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng},
      mutable=["intermediates"],
      capture_intermediates=_capture_rope_inputs,
  )
  captured = _stack_captures(collections, config.num_decoder_layers)
  length = captured["query"].shape[2]
  query_indices = (jnp.arange(query_count) + 1) * (length // query_count) - 1
  one = lambda q, k, v, qb, kb: _layer_metrics(
      q, k, v, qb, kb, batch["inputs_position"],
      batch["inputs_segmentation"], config.bam_k,
      config.bam_k + config.bam_abs_v_compression_dim,
      config.rope_min_timescale, config.rope_max_timescale, query_indices)
  metrics = jax.vmap(one)(
      captured["query"], captured["key"], captured["value"],
      captured["q_bam"], captured["k_bam"])
  mask = batch["targets_segmentation"] != 0
  return {
      "loss_sum": jnp.sum(xent * mask),
      "weight_sum": jnp.sum(mask),
      "metrics": metrics,
  }


def _iter_microbatches(batch, size):
  for start in range(0, int(batch["inputs"].shape[0]), size):
    yield {name: value[start:start + size] for name, value in batch.items()}


def _mean_tree(trees):
  return jax.tree.map(
      lambda *values: np.mean(np.stack(values), axis=0), *trees)


def _json_tree(value):
  if isinstance(value, dict):
    return {key: _json_tree(child) for key, child in value.items()}
  return np.asarray(value).tolist()


def run(config):
  if config.qk_norm or config.fused_qkv or config.rope_half:
    raise ValueError("diagnostic expects V2 QKNorm-off, unfused QKV, full-width RoPE")
  if config.rope_type.lower().startswith(("llama3.1", "yarn")):
    raise ValueError("diagnostic currently supports standard RoPE only")
  batches = int(os.environ.get("BAM_ROPE_DIAG_BATCHES", "4"))
  microbatch = int(os.environ.get("BAM_ROPE_DIAG_MICROBATCH", "2"))
  query_count = int(os.environ.get("BAM_ROPE_DIAG_QUERY_COUNT", "16"))
  output_path = Path(os.environ.get(
      "BAM_ROPE_DIAG_OUTPUT", "/tmp/bam_v2_c256_rope_structure.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  setup_seconds = time.perf_counter() - started
  compiled = jax.jit(lambda params, batch, rng: _forward(
      model, config, params, batch, rng, query_count))

  metric_trees = []
  sequence_hashes = []
  total_loss = total_weight = 0.0
  timings = []
  index = 0
  for batch_index in range(batches):
    batch = next(eval_iterator)
    inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in inputs)
    for small_batch in _iter_microbatches(batch, microbatch):
      begin = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        result = compiled(state.params, small_batch,
                          jax.random.fold_in(init_rng, index))
      jax.block_until_ready(result)
      elapsed = time.perf_counter() - begin
      host = jax.device_get(result)
      total_loss += float(host["loss_sum"])
      total_weight += float(host["weight_sum"])
      metric_trees.append(host["metrics"])
      timings.append(elapsed)
      print(
          f"BAM_ROPE_DIAG batch={batch_index} microbatch={index} "
          f"loss={float(host['loss_sum']) / max(float(host['weight_sum']), 1):.6f} "
          f"forward_s={elapsed:.1f}", flush=True)
      index += 1

  metrics = _mean_tree(metric_trees)
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_ROPE_DIAG_CODE_COMMIT", ""),
          "sequence_count": len(sequence_hashes),
          "unique_sequence_count": len(set(sequence_hashes)),
          "cohort_hash": hashlib.sha256(
              "".join(sequence_hashes).encode()).hexdigest()[:16],
          "sequence_hashes": sequence_hashes,
          "batches": batches,
          "microbatch": microbatch,
          "query_count": query_count,
          "distance_buckets": list(_BUCKET_NAMES),
          "setup_seconds": setup_seconds,
          "elapsed_seconds": time.perf_counter() - started,
          "forward_seconds": timings,
      },
      "eval_loss": total_loss / max(total_weight, 1.0),
      "metrics": _json_tree(metrics),
  }
  output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"BAM_ROPE_DIAG_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

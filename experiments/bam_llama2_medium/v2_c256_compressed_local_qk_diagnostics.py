"""Counterfactual LocalQK reads from the V-compressed fetched-M view.

This is a read-only checkpoint diagnostic.  It maps each learned 32-D column
key to the least-squares coordinates of the layer's existing absolute-V
encoder, re-normalizes it in 8-D, and reads ``M @ E_v``.  The row answer is
either injected directly into head dims 32:40 or decoded back to 32-D with
``pinv(E_v)``.  It reports local read distortion, attention perturbation, and
exact same-batch one-layer/all-layer loss deltas without changing BamAttention.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import attentions
from layers import normalizations
from v2_c256_rope_gate_diagnostics import (  # pylint: disable=unused-import
    BamLlama2MediumV2C256RopeGateDiagnostics,
)
from v2_c256_rope_structure_diagnostics import (
    _attention_delta,
    _capture_rope_inputs,
    _rope,
    _stack_captures,
)


_LAYER_RE = re.compile(r"layers_(\d+)")
_EPS = 1.0e-12


def _layer_index(module) -> int | None:
  for component in module.scope.path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _packed_local_parts(module, inputs_q):
  packed = module.W_local_qk_packed(inputs_q)
  key_width = module.bam_k + module.bam_v
  mix_width = 2 * module.num_query_heads
  split_points = (
      key_width,
      key_width + 2,
      key_width + 2 + mix_width,
      2 * key_width + 2 + mix_width,
      2 * key_width + 4 + mix_width,
  )
  q_key, q_gate, q_mix, k_key, k_gate, k_mix = jnp.split(
      packed, split_points, axis=-1)
  q_key = q_key + jnp.asarray(module.W_lq_bias, q_key.dtype)
  k_key = k_key + jnp.asarray(module.W_lk_bias, k_key.dtype)
  q_gate = q_gate + jnp.asarray(module.W_lq_gate_b0, q_gate.dtype)
  k_gate = k_gate + jnp.asarray(module.W_lk_gate_b0, k_gate.dtype)
  q_mix = q_mix.reshape(
      q_mix.shape[:-1] + (module.num_query_heads, 2))
  k_mix = k_mix.reshape(
      k_mix.shape[:-1] + (module.num_query_heads, 2))
  return ((q_key, q_gate, q_mix, "W_lq"),
          (k_key, k_gate, k_mix, "W_lk"))


def _factorized_read_pair(
    module, matrix, inputs_q, key, gate, mix, name, output_mode):
  if module._create_grouped_rw_norm or module._use_native_grouped_read_norm:
    raise ValueError("compressed-key mapping is undefined for learned read norms")
  kwargs = module._read_key_kwargs_from_logits(name, gate)
  raw_row, raw_col = jnp.split(key, [module.bam_k], axis=-1)
  _, _, row_key, col_key = attentions._project_bam_read_keys(
      module.bam_k, inputs_q, lambda _x: key, **kwargs)
  current_u, current_v = attentions._contract_bam_read_sides(
      matrix, matrix, row_key, col_key, False,
      module._read_implementation, module.read_side)

  encoder = module.abs_v_cache_projection.astype(matrix.dtype)
  compressed_matrix = jnp.einsum("btkv,vc->btkc", matrix, encoder)
  # Best checkpoint-compatible coordinates z for E_v z ~= raw_col.  Using
  # E_v.T directly would silently assume orthonormal learned columns.
  encoder_coordinate_map = jnp.linalg.pinv(
      encoder.astype(jnp.float32)).T.astype(raw_col.dtype)
  compressed_raw_col = jnp.einsum(
      "btv,vc->btc", raw_col, encoder_coordinate_map)
  compressed_key = jnp.concatenate((raw_row, compressed_raw_col), axis=-1)
  _, _, compressed_row_key, compressed_col_key = (
      attentions._project_bam_read_keys(
          module.bam_k, inputs_q, lambda _x: compressed_key, **kwargs))
  compressed_u, compressed_v = attentions._contract_bam_read_sides(
      compressed_matrix, compressed_matrix,
      compressed_row_key, compressed_col_key, False,
      module._read_implementation, module.read_side)
  if output_mode == "decoded":
    compressed_v = jnp.einsum(
        "btc,cv->btv", compressed_v,
        jnp.linalg.pinv(encoder.astype(jnp.float32)).astype(compressed_v.dtype))

  mix = normalizations.rms_norm(
      mix, dtype=current_u.dtype, epsilon=module._read_key_epsilon, axis=-2)
  row_mix, col_mix = mix[..., 0], mix[..., 1]

  def expand(y_u, y_v):
    y_u = jnp.einsum("btk,btn->btnk", y_u, col_mix)
    y_v = jnp.einsum("btv,btn->btnv", y_v, row_mix)
    return jnp.concatenate((y_u, y_v), axis=-1)

  return (
      expand(current_u, current_v),
      expand(compressed_u, compressed_v),
      current_v,
      encoder,
  )


def _compressed_local_qk(module, matrix, inputs_q, output_mode):
  if not module._pack_factorized_local_qk:
    raise ValueError("diagnostic expects packed factorized LocalQK")
  if module._batch_factorized_local_qk_read:
    raise ValueError("diagnostic expects separate packed Q/K slots")
  q_parts, k_parts = _packed_local_parts(module, inputs_q)
  q_current, q_compressed, q_row, q_encoder = _factorized_read_pair(
      module, matrix, inputs_q, *q_parts, output_mode)
  k_current, k_compressed, k_row, k_encoder = _factorized_read_pair(
      module, matrix, inputs_q, *k_parts, output_mode)
  q_current, k_current = module._fit_local_qk_reads(q_current, k_current)
  q_compressed, k_compressed = module._fit_local_qk_reads(
      q_compressed, k_compressed)
  return (
      (q_current, k_current),
      (q_compressed, k_compressed),
      {"q_row": q_row, "k_row": k_row,
       "q_encoder": q_encoder, "k_encoder": k_encoder},
  )


def _compare(current, target, start, end):
  current = current[..., start:end].astype(jnp.float32)
  target = target[..., start:end].astype(jnp.float32)
  axes = (0, 1, 3)
  current2 = jnp.sum(jnp.square(current), axis=axes)
  target2 = jnp.sum(jnp.square(target), axis=axes)
  error2 = jnp.sum(jnp.square(target - current), axis=axes)
  dot = jnp.sum(target * current, axis=axes)
  denom = jnp.sqrt(jnp.maximum(current2 * target2, _EPS))
  return {
      "target_over_current_norm": jnp.sqrt(
          target2 / jnp.maximum(current2, _EPS)),
      "relative_error": jnp.sqrt(error2 / jnp.maximum(current2, _EPS)),
      "cosine": dot / denom,
  }


def _structural_forward(
    model, config, params, batch, rng, query_count, output_mode):
  def compressed_interceptor(next_fun, args, kwargs, context):
    module = context.module
    if (isinstance(module, attentions.BamAttention)
        and context.method_name == "_read_local_qk"):
      next_fun(*args, **kwargs)
      _, compressed, _ = _compressed_local_qk(
          module, args[0], args[1], output_mode)
      return compressed
    return next_fun(*args, **kwargs)

  dropout_rng, params_rng = jax.random.split(rng)
  apply_kwargs = dict(
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng},
      mutable=["intermediates"],
      capture_intermediates=_capture_rope_inputs,
  )
  (_, _, _), current_collections = model.apply(
      params, batch["inputs"], batch["inputs_position"], **apply_kwargs)
  with nn.intercept_methods(compressed_interceptor):
    (_, _, _), compressed_collections = model.apply(
        params, batch["inputs"], batch["inputs_position"], **apply_kwargs)
  current = _stack_captures(current_collections, config.num_decoder_layers)
  compressed = _stack_captures(
      compressed_collections, config.num_decoder_layers)

  query_indices = ((jnp.arange(query_count) + 1)
                   * (batch["inputs"].shape[1] // query_count) - 1)
  query_segments = batch["inputs_segmentation"][:, query_indices]
  key_index = jnp.arange(batch["inputs"].shape[1])
  valid = (
      (query_segments[:, :, None] != 0)
      & (query_segments[:, :, None] == batch["inputs_segmentation"][:, None])
      & (key_index[None, None] <= query_indices[None, :, None]))

  output = {}
  for layer in range(config.num_decoder_layers):
    q_actual, k_actual = current["q_bam"][layer], current["k_bam"][layer]
    q_comp, k_comp = compressed["q_bam"][layer], compressed["k_bam"][layer]
    layer_output = {
        "q_column": _compare(q_actual, q_comp, 0, config.bam_k),
        "q_row": _compare(q_actual, q_comp, config.bam_k, config.head_dim),
        "q_total": _compare(q_actual, q_comp, 0, config.head_dim),
        "k_column": _compare(k_actual, k_comp, 0, config.bam_k),
        "k_row": _compare(k_actual, k_comp, config.bam_k, config.head_dim),
        "k_total": _compare(k_actual, k_comp, 0, config.head_dim),
    }
    q_std = _rope(
        current["query"][layer], batch["inputs_position"],
        config.rope_min_timescale, config.rope_max_timescale)
    k_std = _rope(
        current["key"][layer], batch["inputs_position"],
        config.rope_min_timescale, config.rope_max_timescale)
    layer_output["attention"] = _attention_delta(
        (q_std + q_actual)[:, query_indices], k_std + k_actual,
        (q_std + q_comp)[:, query_indices], k_std + k_comp,
        current["value"][layer], valid, 1.0)
    output[layer] = layer_output
  return output


def _loss_forward(
    model, config, params, batch, rng, target_layer, output_mode):
  def interceptor(next_fun, args, kwargs, context):
    module = context.module
    if (not isinstance(module, attentions.BamAttention)
        or context.method_name != "_read_local_qk"):
      return next_fun(*args, **kwargs)
    layer = _layer_index(module)
    current = next_fun(*args, **kwargs)
    _, compressed, _ = _compressed_local_qk(
        module, args[0], args[1], output_mode)
    selected = (target_layer == layer) | (target_layer == -2)
    weight = jnp.asarray(selected, current[0].dtype)
    return tuple(
        value + weight * (target - value)
        for value, target in zip(current, compressed, strict=True))

  dropout_rng, params_rng = jax.random.split(rng)
  with nn.intercept_methods(interceptor):
    xent, _, _ = model.apply(
        params,
        batch["inputs"],
        batch["inputs_position"],
        decoder_segment_ids=batch["inputs_segmentation"],
        decoder_target_mask=batch["targets_segmentation"],
        decoder_target_tokens=batch["targets"],
        enable_dropout=False,
        rngs={"dropout": dropout_rng, "params": params_rng},
    )
  mask = batch["targets_segmentation"] != 0
  return jnp.sum(xent * mask), jnp.sum(mask)


def _iter_microbatches(batch, size):
  for start in range(0, int(batch["inputs"].shape[0]), size):
    yield {name: value[start:start + size] for name, value in batch.items()}


def _mean_tree(trees: list[Any]):
  return jax.tree.map(
      lambda *values: np.mean(np.stack(values), axis=0), *trees)


def _json_tree(value):
  if isinstance(value, dict):
    return {str(key): _json_tree(child) for key, child in value.items()}
  return np.asarray(value).tolist()


def run(config):
  if config.scan_layers or config.bam_local_qk_injection != "post_rope":
    raise ValueError("diagnostic expects unscanned V2 with post-RoPE LocalQK")
  if config.bam_abs_v_compression_dim != 8:
    raise ValueError("diagnostic currently targets V2 abs_v=8")
  batches = int(os.environ.get("BAM_COMPRESSED_QK_BATCHES", "4"))
  loss_batches = int(os.environ.get("BAM_COMPRESSED_QK_LOSS_BATCHES", "1"))
  microbatch = int(os.environ.get("BAM_COMPRESSED_QK_MICROBATCH", "2"))
  query_count = int(os.environ.get("BAM_COMPRESSED_QK_QUERY_COUNT", "16"))
  output_mode = os.environ.get("BAM_COMPRESSED_QK_OUTPUT_MODE", "direct")
  output_path = Path(os.environ.get(
      "BAM_COMPRESSED_QK_OUTPUT", "/tmp/v2_c256_compressed_local_qk.json"))
  if not 1 <= loss_batches <= batches:
    raise ValueError("loss batches must be in [1, batches]")
  if output_mode not in ("direct", "decoded"):
    raise ValueError("output mode must be direct or decoded")
  output_path.parent.mkdir(parents=True, exist_ok=True)

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  structural = jax.jit(lambda params, batch, rng: _structural_forward(
      model, config, params, batch, rng, query_count, output_mode))
  loss_fn = jax.jit(lambda params, batch, rng, target: _loss_forward(
      model, config, params, batch, rng, target, output_mode))

  metric_trees = []
  sequence_hashes = []
  loss_totals = {target: [0.0, 0.0]
                 for target in [-2, -1, *range(config.num_decoder_layers)]}
  structural_seconds = []
  loss_seconds = []
  microbatch_index = 0
  for batch_index in range(batches):
    batch = next(eval_iterator)
    host_inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in host_inputs)
    for small_batch in _iter_microbatches(batch, microbatch):
      rng = jax.random.fold_in(init_rng, microbatch_index)
      begin = time.perf_counter()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        metrics = structural(state.params, small_batch, rng)
      metrics = jax.device_get(metrics)
      structural_seconds.append(time.perf_counter() - begin)
      metric_trees.append(metrics)

      if batch_index < loss_batches:
        begin = time.perf_counter()
        for target in loss_totals:
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            loss_sum, weight_sum = loss_fn(
                state.params, small_batch, rng, jnp.asarray(target, jnp.int32))
          loss_sum, weight_sum = jax.device_get((loss_sum, weight_sum))
          loss_totals[target][0] += float(loss_sum)
          loss_totals[target][1] += float(weight_sum)
        loss_seconds.append(time.perf_counter() - begin)
      loss_suffix = (
          f" loss_s={loss_seconds[-1]:.2f}"
          if batch_index < loss_batches else "")
      print(
          f"BAM_COMPRESSED_QK batch={batch_index} microbatch={microbatch_index} "
          f"structural_s={structural_seconds[-1]:.2f}{loss_suffix}", flush=True)
      microbatch_index += 1

  losses = {
      target: total / weight for target, (total, weight) in loss_totals.items()}
  baseline = losses[-1]
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_COMPRESSED_QK_CODE_COMMIT", ""),
          "sequence_count": len(sequence_hashes),
          "unique_sequence_count": len(set(sequence_hashes)),
          "cohort_hash": hashlib.sha256(
              "".join(sequence_hashes).encode()).hexdigest()[:16],
          "sequence_hashes": sequence_hashes,
          "batches": batches,
          "loss_batches": loss_batches,
          "microbatch": microbatch,
          "query_count": query_count,
          "output_mode": output_mode,
          "structural_seconds": structural_seconds,
          "loss_seconds": loss_seconds,
          "elapsed_seconds": time.perf_counter() - started,
      },
      "metrics": _json_tree(_mean_tree(metric_trees)),
      "loss": {
          "baseline": baseline,
          "all_layers": losses[-2],
          "all_layers_dloss": losses[-2] - baseline,
          "one_layer": {str(layer): losses[layer]
                        for layer in range(config.num_decoder_layers)},
          "one_layer_dloss": {str(layer): losses[layer] - baseline
                               for layer in range(config.num_decoder_layers)},
      },
  }
  output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"BAM_COMPRESSED_QK_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

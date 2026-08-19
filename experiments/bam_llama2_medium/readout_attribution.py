"""Loss-grounded and structural BAM readout attribution for the V2 checkpoint.

This is an isolated diagnostic runner.  P1 uses one scalar perturbation per
write record, whose gradient is exactly the Frobenius attribution requested by
the study plan.  P2 reduces exact per-record read contributions on TPU and only
returns compact per-site metrics; no multi-GiB cotangent/readout dump crosses
the host boundary.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import exp
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator


_LAYERS = 24
_HEADS = 16
_K = 32
_V = 32
_C = 8
_QUERY_SAMPLES = 16
_SOURCE_TOPK = 1536
_EPS = 1.0e-12
_LAYER_RE = re.compile(r"layers_(\d+)")


class BamLlama2MediumV2ReadoutAttribution(exp.BamLlama2MediumV2):
  """Read-only V2 P1/P2 diagnostic; never used as a training config."""

  bam_diagnostics = True
  bam_readout_attribution = True
  scan_layers = False
  # Preserve the exact shuffled cohort used by delta_rule_write_reuse.py.  The
  # runner slices each 16-sequence iterator batch into smaller backward passes.
  eval_per_device_batch_size = 16.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_readout_attribution_tb/"


exp.BamLlama2MediumV2ReadoutAttribution = BamLlama2MediumV2ReadoutAttribution


def _unwrap(value: Any) -> Any:
  while isinstance(value, (tuple, list)) and len(value) == 1:
    value = value[0]
  return value


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for part in path:
    match = _LAYER_RE.fullmatch(part)
    if match:
      return int(match.group(1))
  return None


def _group_by_layer(tree: dict[str, Any], leaf_name: str | None = None) -> dict[int, dict[str, Any]]:
  grouped: dict[int, dict[str, Any]] = defaultdict(dict)
  for path, value in traverse_util.flatten_dict(tree).items():
    layer = _layer_from_path(path)
    if layer is None:
      continue
    name = leaf_name or path[-1]
    grouped[layer][name] = _unwrap(value)
  return dict(sorted(grouped.items()))


def _stack_collection(collections: dict[str, Any]) -> dict[str, jax.Array]:
  grouped = _group_by_layer(collections["bam_readout"])
  if set(grouped) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} bam_readout layers, got {sorted(grouped)}")
  names = set(grouped[0])
  for layer, values in grouped.items():
    if set(values) != names:
      raise ValueError(f"layer {layer} readout keys differ: {set(values) ^ names}")
  return {name: jnp.stack([grouped[layer][name] for layer in range(_LAYERS)]) for name in sorted(names)}


def _stack_attributions(grad_perturbations: dict[str, Any]) -> jax.Array:
  grouped = _group_by_layer(grad_perturbations, "write_record_scale")
  if set(grouped) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} perturbation layers, got {sorted(grouped)}")
  return jnp.stack([grouped[layer]["write_record_scale"] for layer in range(_LAYERS)])


def _make_perturbations(params: dict[str, Any], batch_size: int, length: int) -> dict[str, Any]:
  leaves = {}
  for path in traverse_util.flatten_dict(params["params"]):
    if path[-1] == "gw_b0":
      leaves[path[:-1] + ("write_record_scale",)] = jnp.zeros(
          (batch_size, length, _HEADS), jnp.float32)
  if len(leaves) != _LAYERS:
    raise ValueError(f"expected {_LAYERS} BAM modules, found {len(leaves)}")
  return traverse_util.unflatten_dict(leaves)


def _offset_perturbations(
    perturbations: dict[str, Any], direction: np.ndarray, delta: float) -> dict[str, Any]:
  leaves = traverse_util.flatten_dict(perturbations)
  return traverse_util.unflatten_dict({
      path: value + delta * direction[_layer_from_path(path)]
      for path, value in leaves.items()
  })


def _abs_v_projections(params: dict[str, Any]) -> jax.Array:
  by_layer = {}
  for path, value in traverse_util.flatten_dict(params["params"]).items():
    if path[-1] == "abs_v_cache_projection":
      layer = _layer_from_path(path)
      if layer is not None:
        by_layer[layer] = value
  if set(by_layer) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} AbsV projections, got {sorted(by_layer)}")
  return jnp.stack([by_layer[layer] for layer in range(_LAYERS)]).astype(jnp.float32)


def _model_loss_and_capture(model, params, perturbations, batch, rng):
  variables = dict(params)
  variables["perturbations"] = perturbations
  rng1, aqt_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      variables,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=["bam_readout"],
  )
  mask = batch["targets_segmentation"] != 0
  total_loss = jnp.sum(xent * mask)
  total_weights = jnp.sum(mask)
  return total_loss / jnp.maximum(total_weights, 1), (total_weights, collections)


def _gather_sources(values, indices):
  """[L,b,t,h,...] + [b,q,m] -> [b,q,L,m,h,...]."""
  values = jnp.swapaxes(values, 0, 1)
  gathered = jax.vmap(lambda value, index: jnp.take(value, index, axis=1))(
      values, indices)
  axes = (0, 2, 1, 3, 4) + tuple(range(5, gathered.ndim))
  return jnp.transpose(gathered, axes)


def _top_record_metrics(score, norm2, y_truncated, y_actual, attrs, source_layers,
                        self_mask=None):
  """Reduce [b,q,n,L,m,h] record contributions to compact site metrics."""
  actual_norm2 = jnp.sum(jnp.square(y_actual), axis=-1)
  truncated_norm2 = jnp.sum(jnp.square(y_truncated), axis=-1)
  score = score / jnp.maximum(actual_norm2[..., None, None, None], _EPS)
  flat = score.reshape(score.shape[:3] + (-1,))
  _, order = jax.lax.top_k(jnp.abs(flat), min(8, flat.shape[-1]))
  top_signed = jnp.take_along_axis(flat, order, axis=-1)
  abs_score = jnp.abs(score)
  attr_harmful = attrs > 0
  output = {
      "top1_signed_share": top_signed[..., 0],
      "top8_signed_share": jnp.sum(top_signed, axis=-1),
      "top1_abs_share": jnp.abs(top_signed[..., 0]),
      "top8_abs_share": jnp.sum(jnp.abs(top_signed), axis=-1),
      "signed_share_sum": jnp.sum(score, axis=(-3, -2, -1)),
      "absolute_share_sum": jnp.sum(abs_score, axis=(-3, -2, -1)),
      "coherence": truncated_norm2 / jnp.maximum(
          jnp.sum(norm2, axis=(-3, -2, -1)), _EPS),
      "reconstruction_relative_norm": jnp.sqrt(
          jnp.sum(jnp.square(y_truncated - y_actual), axis=-1)
          / jnp.maximum(actual_norm2, _EPS)),
      "harmful_attribution_abs_share": jnp.sum(
          abs_score * attr_harmful, axis=(-3, -2, -1))
          / jnp.maximum(jnp.sum(abs_score, axis=(-3, -2, -1)), _EPS),
      "source_layer_signed_share": jnp.sum(score, axis=(-2, -1)),
      "source_layer_abs_share": jnp.sum(abs_score, axis=(-2, -1)),
      "source_layers": source_layers,
  }
  if self_mask is not None:
    self_mask = self_mask[:, :, None, None, :, None]
    output.update({
        "self_signed_share": jnp.sum(jnp.where(self_mask, score, 0), axis=(-3, -2, -1)),
        "cross_signed_share": jnp.sum(jnp.where(self_mask, 0, score), axis=(-3, -2, -1)),
        "self_abs_share": jnp.sum(jnp.where(self_mask, abs_score, 0), axis=(-3, -2, -1)),
        "cross_abs_share": jnp.sum(jnp.where(self_mask, 0, abs_score), axis=(-3, -2, -1)),
    })
  return output


def _fetch_layer_metrics(layer, raw, attrs, projection, query_indices, *, permute_alpha=False):
  indices = raw["fetch_source_indices"][layer]
  alpha = raw["fetch_source_weights"][layer].astype(jnp.float32)
  if permute_alpha:
    indices = jnp.roll(indices, 1, axis=1)
    alpha = jnp.roll(alpha, 1, axis=1)
  u = _gather_sources(raw["write_u1_norm"].astype(jnp.float32), indices)
  v = _gather_sources(raw["write_u2_norm"].astype(jnp.float32), indices)
  scale = _gather_sources(raw["write_scale"].astype(jnp.float32), indices)
  attr = _gather_sources(attrs.astype(jnp.float32), indices)
  v = jnp.einsum("bqlmhv,vc->bqlmhc", v, projection[layer])

  key = raw["full_post_gate_key"][layer].astype(jnp.float32)
  r_row, r_col = jnp.split(key, [_K], axis=-1)
  layer_mask = (jnp.arange(_LAYERS) < layer).astype(jnp.float32)
  coefficient = (
      alpha[:, :, None, :, None]
      * scale
      * layer_mask[None, None, :, None, None]
  )
  col_dot = jnp.einsum("bqlmhc,bqnc->bqnlmh", v, r_col)
  row_dot = jnp.einsum("bqlmhk,bqnk->bqnlmh", u, r_row)
  a_u = col_dot * coefficient[:, :, None]
  a_v = row_dot * coefficient[:, :, None]

  norm2 = (
      jnp.square(a_u) * jnp.sum(jnp.square(u), axis=-1)[:, :, None]
      + jnp.square(a_v) * jnp.sum(jnp.square(v), axis=-1)[:, :, None]
  )
  y_truncated = jnp.concatenate((
      jnp.einsum("bqnlmh,bqlmhk->bqnk", a_u, u),
      jnp.einsum("bqnlmh,bqlmhc->bqnc", a_v, v),
  ), axis=-1)
  y_actual = raw["y_full"][layer].astype(jnp.float32)
  y_actual = jnp.concatenate((y_actual[..., :_K], y_actual[..., _K:_K + _C]), axis=-1)
  y_reference = y_truncated if permute_alpha else y_actual
  y_u, y_v = jnp.split(y_reference, [_K], axis=-1)
  u_y = jnp.einsum("bqlmhk,bqnk->bqnlmh", u, y_u)
  v_y = jnp.einsum("bqlmhc,bqnc->bqnlmh", v, y_v)
  score = a_u * u_y + a_v * v_y
  attr = attr[:, :, None]
  metrics = _top_record_metrics(
      score, norm2, y_truncated, y_reference,
      attr, jnp.arange(_LAYERS),
      self_mask=indices == query_indices[None, :, None])
  metrics["retained_alpha_abs_mass"] = raw["fetch_retained_abs_mass"][layer]
  metrics["support_99_count"] = raw["fetch_support_99_count"][layer]
  return metrics


def _local_layer_metrics(layer, raw, attrs, query_indices, prefix):
  def gather_queries(values):
    values = jnp.swapaxes(values, 0, 1)
    gathered = jax.vmap(lambda value: jnp.take(value, query_indices, axis=1))(values)
    axes = (0, 2, 1, 3) + tuple(range(4, gathered.ndim))
    return jnp.transpose(gathered, axes)

  u = gather_queries(raw["write_u1_norm"].astype(jnp.float32))
  v = gather_queries(raw["write_u2_norm"].astype(jnp.float32))
  scale = gather_queries(raw["write_scale"].astype(jnp.float32))
  attr = gather_queries(attrs.astype(jnp.float32))
  key = raw[f"{prefix}_post_gate_key"][layer].astype(jnp.float32)
  r_row, r_col = jnp.split(key, [_K], axis=-1)
  head_mix = raw[f"{prefix}_head_mix"][layer].astype(jnp.float32)
  row_mix, col_mix = head_mix[..., 0], head_mix[..., 1]
  layer_mask = (jnp.arange(_LAYERS) < layer).astype(jnp.float32)
  coefficient = scale * layer_mask[None, None, :, None]
  col_dot = jnp.einsum("bqlhv,bqv->bqlh", v, r_col)
  row_dot = jnp.einsum("bqlhk,bqk->bqlh", u, r_row)
  a_u = coefficient[:, :, None] * col_dot[:, :, None] * col_mix[:, :, :, None, None]
  a_v = coefficient[:, :, None] * row_dot[:, :, None] * row_mix[:, :, :, None, None]

  pre_mix = raw[f"{prefix}_pre_mix_read"][layer].astype(jnp.float32)
  pre_u, pre_v = jnp.split(pre_mix, [_K], axis=-1)
  y_actual = jnp.concatenate((
      pre_u[:, :, None] * col_mix[..., None],
      pre_v[:, :, None] * row_mix[..., None],
  ), axis=-1)
  y_u, y_v = jnp.split(y_actual, [_K], axis=-1)
  u_y = jnp.einsum("bqlhk,bqnk->bqnlh", u, y_u)
  v_y = jnp.einsum("bqlhv,bqnv->bqnlh", v, y_v)
  score = a_u * u_y + a_v * v_y
  norm2 = (
      jnp.square(a_u) * jnp.sum(jnp.square(u), axis=-1)[:, :, None]
      + jnp.square(a_v) * jnp.sum(jnp.square(v), axis=-1)[:, :, None]
  )
  y_truncated = jnp.concatenate((
      jnp.einsum("bqnlh,bqlhk->bqnk", a_u, u),
      jnp.einsum("bqnlh,bqlhv->bqnv", a_v, v),
  ), axis=-1)
  # Reuse the generic reducer by adding a singleton source-token axis.
  return _top_record_metrics(
      score[..., None, :], norm2[..., None, :], y_truncated, y_actual,
      attr[:, :, None, :, None, :], jnp.arange(_LAYERS))


def _p2_all_layers(raw, attrs, projections):
  query_indices = (jnp.arange(_QUERY_SAMPLES) + 1) * (
      raw["write_scale"].shape[2] // _QUERY_SAMPLES) - 1

  def one(layer):
    return {
        "fetch": _fetch_layer_metrics(
            layer, raw, attrs, projections, query_indices),
        "fetch_permuted_alpha": _fetch_layer_metrics(
            layer, raw, attrs, projections, query_indices, permute_alpha=True),
        "local_q": _local_layer_metrics(layer, raw, attrs, query_indices, "local_q"),
        "local_k": _local_layer_metrics(layer, raw, attrs, query_indices, "local_k"),
    }

  return jax.lax.map(one, jnp.arange(_LAYERS))


def _flatten_for_npz(tree: Any, prefix: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
  output = {}
  if isinstance(tree, dict):
    for key, value in tree.items():
      output.update(_flatten_for_npz(value, prefix + (str(key),)))
  else:
    output["__".join(prefix)] = np.asarray(tree)
  return output


def _distribution(value: np.ndarray) -> dict[str, float]:
  value = np.asarray(value, np.float64).reshape(-1)
  value = value[np.isfinite(value)]
  if not value.size:
    return {key: float("nan") for key in ("mean", "std", "p10", "p50", "p90", "p99")}
  return {
      "mean": float(np.mean(value)),
      "std": float(np.std(value)),
      "p10": float(np.percentile(value, 10)),
      "p50": float(np.percentile(value, 50)),
      "p90": float(np.percentile(value, 90)),
      "p99": float(np.percentile(value, 99)),
  }


def _to_float32(value: np.ndarray) -> np.ndarray:
  """Decode JAX bf16 arrays saved by NumPy as opaque two-byte values."""
  value = np.asarray(value)
  if value.dtype.kind == "V" and value.dtype.itemsize == 2:
    value = value.view(ml_dtypes.bfloat16)
  return value.astype(np.float32)


def _rankdata(value: np.ndarray) -> np.ndarray:
  order = np.argsort(value, kind="stable")
  ranks = np.empty(order.size, np.float64)
  ranks[order] = np.arange(order.size, dtype=np.float64)
  return ranks


def _correlations(gate: np.ndarray, value: np.ndarray, sample_cap=500_000) -> dict[str, float]:
  gate = np.asarray(gate, np.float64).reshape(-1)
  value = np.asarray(value, np.float64).reshape(-1)
  if gate.size > sample_cap:
    select = np.linspace(0, gate.size - 1, sample_cap, dtype=np.int64)
    gate, value = gate[select], value[select]
  pearson = float(np.corrcoef(gate, value)[0, 1])
  spearman = float(np.corrcoef(_rankdata(gate), _rankdata(value))[0, 1])
  return {"pearson": pearson, "spearman": spearman, "sample_count": int(gate.size)}


def _analyze_p1(output_dir: Path) -> dict[str, Any]:
  files = sorted(output_dir.glob("attribution_batch_*.npz"))
  if not files:
    raise FileNotFoundError("no attribution batches")
  overall_attr_sample = []
  overall_gate_sample = []
  layer_reports = {}
  head_positive = np.zeros(_HEADS, np.float64)
  head_absolute = np.zeros(_HEADS, np.float64)
  total_positive = total_absolute = total_net = total_gate = total_gate_attr = 0.0
  for layer in range(_LAYERS):
    attrs, gates = [], []
    for path in files:
      data = np.load(path)
      valid = data["valid"].astype(bool)
      attrs.append(data["attr_sumloss"][layer][valid])
      gates.append(_to_float32(data["write_gate"][layer][valid]))
    attr = np.concatenate(attrs)
    gate = np.concatenate(gates)
    positive = np.maximum(attr, 0)
    absolute = np.abs(attr)
    layer_reports[f"layer_{layer:02d}"] = {
        "attr": _distribution(attr),
        "attr_per_unit_gate": _distribution(attr / np.maximum(gate, 1e-8)),
        "harmful_mass": float(np.sum(positive) / max(np.sum(absolute), _EPS)),
        "net_attr": float(np.sum(attr)),
        "gate_weighted_mean_attr": float(np.sum(gate * attr) / max(np.sum(gate), _EPS)),
        "gate_vs_helpfulness": _correlations(gate, -attr),
    }
    head_positive += np.sum(positive, axis=0)
    head_absolute += np.sum(absolute, axis=0)
    total_positive += float(np.sum(positive))
    total_absolute += float(np.sum(absolute))
    total_net += float(np.sum(attr))
    total_gate += float(np.sum(gate))
    total_gate_attr += float(np.sum(gate * attr))
    select = np.linspace(0, attr.size - 1, min(100_000, attr.size), dtype=np.int64)
    overall_attr_sample.append(attr.reshape(-1)[select])
    overall_gate_sample.append(gate.reshape(-1)[select])
  attr_sample = np.concatenate(overall_attr_sample)
  gate_sample = np.concatenate(overall_gate_sample)
  return {
      "overall": {
          "attr_sampled": _distribution(attr_sample),
          "harmful_mass_exact": total_positive / max(total_absolute, _EPS),
          "net_attr_exact": total_net,
          "gate_weighted_mean_attr_exact": total_gate_attr / max(total_gate, _EPS),
          "gate_vs_helpfulness_sampled": _correlations(gate_sample, -attr_sample, 2_000_000),
          "per_head_harmful_mass": (
              head_positive / np.maximum(head_absolute, _EPS)).tolist(),
      },
      "layers": layer_reports,
  }


def _analyze_p2(output_dir: Path) -> dict[str, Any]:
  files = sorted(output_dir.glob("p2_batch_*.npz"))
  if not files:
    raise FileNotFoundError("no P2 batches")
  metric_names = (
      "top1_signed_share", "top8_signed_share", "top1_abs_share",
      "top8_abs_share", "signed_share_sum", "absolute_share_sum",
      "coherence", "reconstruction_relative_norm",
      "harmful_attribution_abs_share", "self_signed_share",
      "cross_signed_share", "self_abs_share", "cross_abs_share",
      "retained_alpha_abs_mass",
      "support_99_count",
  )
  sites = ("fetch", "fetch_permuted_alpha", "local_q", "local_k")
  values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
  depth_signed = {site: np.zeros(_LAYERS, np.float64) for site in sites}
  depth_absolute = {site: np.zeros(_LAYERS, np.float64) for site in sites}
  for path in files:
    with np.load(path) as data:
      for site in sites:
        for metric in metric_names:
          key = f"{site}__{metric}"
          if key in data:
            values[(site, metric)].append(_to_float32(data[key]))
        signed_key = f"{site}__source_layer_signed_share"
        absolute_key = f"{site}__source_layer_abs_share"
        signed = _to_float32(data[signed_key]).astype(np.float64)
        absolute = _to_float32(data[absolute_key]).astype(np.float64)
        for use_layer in range(1, _LAYERS):
          for source_layer in range(use_layer):
            gap = use_layer - source_layer
            depth_signed[site][gap] += np.sum(signed[use_layer, ..., source_layer])
            depth_absolute[site][gap] += np.sum(absolute[use_layer, ..., source_layer])

  report = {}
  per_layer_metrics = (
      "top1_abs_share", "top8_abs_share", "coherence",
      "harmful_attribution_abs_share", "reconstruction_relative_norm")
  for site in sites:
    site_report = {}
    for metric in metric_names:
      arrays = values.get((site, metric), [])
      if arrays:
        joined = np.concatenate(arrays, axis=1)
        site_report[metric] = _distribution(joined[1:])
    site_report["by_use_layer"] = {
        f"layer_{layer:02d}": {
            metric: _distribution(np.concatenate(values[(site, metric)], axis=1)[layer])
            for metric in per_layer_metrics
            if values.get((site, metric))
        }
        for layer in range(1, _LAYERS)
    }
    absolute_total = np.sum(depth_absolute[site][1:])
    site_report["depth_gap_profile"] = {
        str(gap): {
            "signed_share_sum": float(depth_signed[site][gap]),
            "absolute_mass_fraction": float(
                depth_absolute[site][gap] / max(absolute_total, _EPS)),
        }
        for gap in range(1, _LAYERS)
    }
    report[site] = site_report
  return report


def run(config) -> None:
  output_dir = Path(os.environ.get("BAM_ATTR_OUTPUT_DIR", "/tmp/bam_readout_attribution"))
  output_dir.mkdir(parents=True, exist_ok=True)
  target_sequences = int(os.environ.get("BAM_ATTR_SEQUENCES", "128"))
  sequence_offset = int(os.environ.get("BAM_ATTR_SEQUENCE_OFFSET", "0"))
  iterator_batch_size = int(
      config.eval_per_device_batch_size * jax.local_device_count())
  batch_size = int(os.environ.get("BAM_ATTR_BATCH_SIZE", "2"))
  if (target_sequences % batch_size or iterator_batch_size % batch_size
      or sequence_offset % iterator_batch_size):
    raise ValueError(
        f"invalid {target_sequences=}, {sequence_offset=}, {batch_size=}, "
        f"{iterator_batch_size=}")
  num_batches = target_sequences // batch_size
  microbatches_per_input = iterator_batch_size // batch_size
  output_batch_offset = sequence_offset // batch_size

  start = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  projections = _abs_v_projections(state.params)
  compiled_grad = jax.jit(jax.value_and_grad(
      lambda perturbations, params, batch, rng: _model_loss_and_capture(
          model, params, perturbations, batch, rng),
      argnums=0, has_aux=True))
  compiled_p2 = jax.jit(_p2_all_layers)
  perturbations = None
  metadata = {
      "checkpoint": config.load_parameters_path,
      "checkpoint_trainer_commit": "1afd942",
      "diagnostic_commit": os.environ.get("BAM_ATTR_DIAGNOSTIC_COMMIT", "unknown"),
      "sequences": target_sequences,
      "sequence_offset": sequence_offset,
      "batch_size": batch_size,
      "iterator_batch_size": iterator_batch_size,
      "num_batches": num_batches,
      "query_positions": ((np.arange(_QUERY_SAMPLES) + 1)
                          * (config.max_target_length // _QUERY_SAMPLES) - 1).tolist(),
      "source_topk": _SOURCE_TOPK,
      "data_shuffle_seed": config.data_shuffle_seed,
      "setup_seconds": time.perf_counter() - start,
      "device": [str(device) for device in jax.devices()],
      "uniform_scale_check": None,
  }

  for _ in range(sequence_offset // iterator_batch_size):
    next(eval_data_iterator)

  input_batch = None
  for batch_index in range(num_batches):
    microbatch_index = batch_index % microbatches_per_input
    if microbatch_index == 0:
      input_batch = next(eval_data_iterator)
    start_index = microbatch_index * batch_size
    batch = jax.tree.map(
        lambda value: value[start_index:start_index + batch_size], input_batch)
    if perturbations is None:
      perturbations = _make_perturbations(
          state.params, batch["inputs"].shape[0], batch["inputs"].shape[1])
    batch_rng = jax.random.fold_in(init_rng, batch_index)
    batch_start = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      (loss_and_aux, grad_perturbations) = compiled_grad(
          perturbations, state.params, batch, batch_rng)
      loss, (total_weights, collections) = loss_and_aux
      attrs = _stack_attributions(grad_perturbations)
      raw = _stack_collection(collections)
      p2 = compiled_p2(raw, attrs, projections)
    jax.block_until_ready((loss, attrs, p2))

    if batch_index == 0:
      # Move exactly one bf16 ULP on either side of one, then divide by the
      # realized scale interval rather than the nominal float32 probe interval.
      epsilon = 2.0**-7
      attrs_for_check = np.asarray(jax.device_get(attrs), np.float32)
      selected_count = min(256, attrs_for_check.size)
      selected = np.argpartition(
          np.abs(attrs_for_check).ravel(), -selected_count)[-selected_count:]
      direction = np.zeros(attrs_for_check.size, np.float32)
      direction[selected] = np.sign(attrs_for_check.ravel()[selected])
      direction = direction.reshape(attrs_for_check.shape)
      plus = _offset_perturbations(perturbations, direction, epsilon)
      minus = _offset_perturbations(perturbations, direction, -epsilon)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        (loss_plus, _), _ = compiled_grad(plus, state.params, batch, batch_rng)
        (loss_minus, _), _ = compiled_grad(minus, state.params, batch, batch_rng)
      scale_plus = float(np.asarray(jnp.asarray(1 + epsilon, config.dtype)))
      scale_minus = float(np.asarray(jnp.asarray(1 - epsilon, config.dtype)))
      numerical = (loss_plus - loss_minus) / (scale_plus - scale_minus)
      analytic = jnp.sum(attrs * direction)
      numerical, analytic = jax.device_get((numerical, analytic))
      metadata["uniform_scale_check"] = {
          "epsilon": epsilon,
          "direction": f"top-{selected_count} absolute attributions",
          "scale_plus": scale_plus,
          "scale_minus": scale_minus,
          "analytic": float(analytic),
          "numerical": float(numerical),
          "relative_error": float(abs(analytic - numerical) / max(abs(numerical), _EPS)),
      }

    attrs_host, raw_small, p2_host, total_weights_host = jax.device_get((
        attrs,
        {"write_gate": raw["write_gate"]},
        p2,
        total_weights,
    ))
    inputs_host, valid_host = jax.device_get((
        batch["inputs"], batch["targets_segmentation"] != 0))
    np.savez_compressed(
        output_dir / f"attribution_batch_{output_batch_offset + batch_index:03d}.npz",
        attr_mean=np.asarray(attrs_host, np.float32),
        attr_sumloss=np.asarray(attrs_host, np.float32) * float(total_weights_host),
        write_gate=np.asarray(raw_small["write_gate"]),
        valid=np.asarray(valid_host),
        sequence_hashes=np.asarray([
            hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
            for sequence in np.asarray(inputs_host)
        ]),
    )
    np.savez_compressed(
        output_dir / f"p2_batch_{output_batch_offset + batch_index:03d}.npz",
        **_flatten_for_npz(jax.device_get(p2_host)))
    print(
        f"ATTR batch={batch_index + 1}/{num_batches} loss={float(loss):.6f} "
        f"seconds={time.perf_counter() - batch_start:.1f}", flush=True)

  metadata["total_seconds"] = time.perf_counter() - start
  report = {
      "metadata": metadata,
      "p1": _analyze_p1(output_dir),
      "p2": _analyze_p2(output_dir),
  }
  report_path = output_dir / "readout_attribution.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"READOUT_ATTRIBUTION_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Residual-space energy and frozen-final-RMS integrated-gradient attribution.

The production model only exposes guarded raw tensors.  This runner projects
the BAM head-space reads through each layer's existing W_O and performs every
statistic on device; no residual-width activation or gradient is saved.
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
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import exp
import pyconfig
import train
import max_utils
from input_pipeline.input_pipeline_interface import create_data_iterator


_BASE_CONFIG_CLASS = os.environ.get(
    "BAM_RESIDUAL_ATTR_BASE_CONFIG",
    "BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2")
_TRAINER_COMMIT = os.environ.get(
    "BAM_RESIDUAL_ATTR_TRAINER_COMMIT", "aef0d97411a1725386ebba1aeae1bf4acb1bb79e")
_LAYERS = int(os.environ.get("BAM_RESIDUAL_ATTR_LAYERS", "24"))
_COMPONENTS = (
    "mlp",
    "mha",
    "bam_col_self",
    "bam_col_cross",
    "bam_row_self",
    "bam_row_cross",
)
_BAM_HEAD_COMPONENTS = (
    "bam_col_self_head",
    "bam_col_cross_head",
    "bam_row_self_head",
    "bam_row_cross_head",
)
_LAYER_RE = re.compile(r"layers_(\d+)")
_EPS = 1.0e-12


class BamResidualAttribution(getattr(exp, _BASE_CONFIG_CLASS)):
  """Read-only residual decomposition diagnostic for a sealed BAM run."""

  bam_diagnostics = False
  bam_readout_attribution = False
  bam_residual_attribution = True
  eval_per_device_batch_size = 16.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_residual_attribution_tb/"


exp.BamResidualAttribution = BamResidualAttribution


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


def _layer_axis_first(value: jax.Array, name: str) -> jax.Array:
  axes = [axis for axis, size in enumerate(value.shape) if size == _LAYERS]
  if len(axes) != 1:
    raise ValueError(
        f"cannot identify unique {_LAYERS}-layer axis for {name}: {value.shape}")
  return jnp.moveaxis(value, axes[0], 0)


def _layer_collection(collections: dict[str, Any]) -> dict[str, jax.Array]:
  expected = {
      "attention_total", "mlp_residual", "layer_delta", "bam_full_head",
      "fetch_self_weight", *_BAM_HEAD_COMPONENTS,
  }
  grouped: dict[int, dict[str, Any]] = {}
  scanned: dict[str, list[jax.Array]] = {}
  for path, value in traverse_util.flatten_dict(
      collections["residual_attribution"]).items():
    if path[-1] not in expected:
      continue
    layer = _layer_from_path(path)
    if layer is not None:
      grouped.setdefault(layer, {})[path[-1]] = _unwrap(value)
    else:
      scanned.setdefault(path[-1], []).append(_unwrap(value))
  if grouped:
    if set(grouped) != set(range(_LAYERS)):
      raise ValueError(
          f"expected {_LAYERS} residual-attribution layers, got {sorted(grouped)}")
    for layer, values in grouped.items():
      if set(values) != expected:
        raise ValueError(
            f"layer {layer} collection mismatch: {set(values) ^ expected}")
    return {
        name: jnp.stack([grouped[layer][name] for layer in range(_LAYERS)])
        for name in sorted(expected)
    }
  if set(scanned) != expected or any(len(values) != 1 for values in scanned.values()):
    raise ValueError(
        f"scanned residual-attribution collection mismatch: "
        f"names={sorted(scanned)}, counts={ {k: len(v) for k, v in scanned.items()} }")
  return {
      name: _layer_axis_first(values[0], name)
      for name, values in scanned.items()
  }


def _top_collection_value(
    collections: dict[str, Any], name: str) -> jax.Array:
  matches = [
      _unwrap(value)
      for path, value in traverse_util.flatten_dict(
          collections["residual_attribution"]).items()
      if _layer_from_path(path) is None and path[-1] == name
  ]
  if len(matches) != 1:
    raise ValueError(f"expected one top-level {name}, got {len(matches)}")
  return matches[0]


def _out_kernels(params: dict[str, Any]) -> jax.Array:
  by_layer = {}
  scanned = []
  for path, value in traverse_util.flatten_dict(params["params"]).items():
    layer = _layer_from_path(path)
    if "self_attention" in path and path[-2:] == ("out", "kernel"):
      if layer is None:
        scanned.append(value)
      else:
        by_layer[layer] = value
  if by_layer:
    if set(by_layer) != set(range(_LAYERS)):
      raise ValueError(f"expected {_LAYERS} W_O kernels, got {sorted(by_layer)}")
    return jnp.stack([by_layer[layer] for layer in range(_LAYERS)])
  if len(scanned) != 1:
    raise ValueError(f"expected one scanned W_O kernel, got {len(scanned)}")
  return _layer_axis_first(scanned[0], "W_O")


def _output_head_parameters(
    params: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
  norm_scales = []
  logits_kernels = []
  for path, value in traverse_util.flatten_dict(params["params"]).items():
    if path[-2:] == ("decoder_norm", "scale"):
      norm_scales.append(value)
    elif path[-1] == "logits_dense" and "lm_head" in path:
      logits_kernels.append(value)
  if len(norm_scales) != 1 or len(logits_kernels) != 1:
    raise ValueError(
        "expected one final RMS scale and one logits kernel; got "
        f"{len(norm_scales)} and {len(logits_kernels)}")
  return norm_scales[0], logits_kernels[0]


def _sequence_mean(values: jax.Array, valid: jax.Array) -> jax.Array:
  """Average a [b,t,...] tensor over valid tokens."""
  weights = valid.astype(jnp.float32)
  weights = weights.reshape(weights.shape + (1,) * (values.ndim - 2))
  count = jnp.maximum(jnp.sum(valid, axis=1), 1)
  return jnp.sum(values * weights, axis=1) / count.reshape(
      count.shape + (1,) * (values.ndim - 2))


def _frozen_head_token_loss(
    hidden: jax.Array,
    frozen_denom: jax.Array,
    norm_scale: jax.Array,
    logits_kernel: jax.Array,
    targets: jax.Array,
    config,
) -> jax.Array:
  """Per-token CE with the final RMS denominator frozen at the real endpoint."""
  dtype = config.dtype
  normalized = jnp.asarray(
      hidden * jax.lax.rsqrt(frozen_denom), dtype)
  scale = jnp.asarray(norm_scale, dtype)
  if not config.direct_scale:
    scale = scale + jnp.asarray(1, dtype)
  normalized = normalized * scale
  kernel = jnp.asarray(logits_kernel, dtype)
  losses = []
  for start in range(0, hidden.shape[1], config.loss_chunk_size):
    end = min(start + config.loss_chunk_size, hidden.shape[1])
    logits = jnp.tensordot(
        normalized[:, start:end], kernel, axes=((-1,), (0,)))
    one_hot_targets = jax.nn.one_hot(
        targets[:, start:end], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(
        logits, one_hot_targets, 0.0)
    losses.append(xent)
  return jnp.concatenate(losses, axis=1)


def _component_token_dot(
    components: jax.Array, gradient: jax.Array) -> jax.Array:
  """Dot [l,b,t,c,d] components with [b,t,d], returning [b,t,l,c]."""
  return jnp.transpose(
      jnp.einsum("lbtcd,btd->lbtc", components, gradient), (1, 2, 0, 3))


def _summarize_capture(
    collections: dict[str, Any],
    params: dict[str, Any],
    batch: dict[str, jax.Array],
    production_xent: jax.Array,
    config,
    quadrature_order: int,
) -> dict[str, jax.Array]:
  captured = _layer_collection(collections)
  embedding = _top_collection_value(collections, "embedding").astype(jnp.float32)
  final_hidden = _top_collection_value(
      collections, "final_hidden").astype(jnp.float32)
  valid = batch["targets_segmentation"] != 0
  valid_count = jnp.maximum(jnp.sum(valid, axis=1), 1)

  w_o = _out_kernels(params)
  bam_heads = jnp.stack(
      [captured[name] for name in _BAM_HEAD_COMPONENTS], axis=3)
  bam_residual = jnp.einsum(
      "lbtcnd,lnde->lbtce",
      bam_heads,
      jnp.asarray(w_o, bam_heads.dtype),
      precision=jax.lax.Precision(config.matmul_precision),
  ).astype(jnp.float32)
  bam_total_residual = jnp.einsum(
      "lbtnd,lnde->lbte",
      captured["bam_full_head"],
      jnp.asarray(w_o, captured["bam_full_head"].dtype),
      precision=jax.lax.Precision(config.matmul_precision),
  ).astype(jnp.float32)
  bam_residual = bam_residual.at[..., -1, :].set(
      bam_total_residual - jnp.sum(bam_residual[..., :-1, :], axis=3))
  layer_delta = captured["layer_delta"].astype(jnp.float32)
  mlp = captured["mlp_residual"].astype(jnp.float32)
  attention_total = captured["attention_total"].astype(jnp.float32)
  mha = layer_delta - mlp - jnp.sum(bam_residual, axis=3)
  components = jnp.concatenate(
      (mlp[..., None, :], mha[..., None, :], bam_residual), axis=3)

  reconstructed = embedding + jnp.sum(layer_delta, axis=0)
  residual_error = reconstructed - final_hidden
  final_l2 = jnp.sqrt(jnp.sum(jnp.square(final_hidden), axis=-1) + _EPS)
  residual_closure_token = (
      jnp.sqrt(jnp.sum(jnp.square(residual_error), axis=-1))
      / final_l2)
  rounding_correction = layer_delta - mlp - attention_total
  rounding_ratio_token = (
      jnp.sqrt(jnp.sum(jnp.square(rounding_correction), axis=-1))
      / jnp.maximum(
          jnp.sqrt(jnp.sum(jnp.square(layer_delta), axis=-1) + _EPS),
          _EPS))

  head_sum = jnp.sum(bam_heads, axis=3)
  head_error = captured["bam_full_head"] - head_sum
  head_closure_token = (
      jnp.sqrt(jnp.sum(jnp.square(
          head_error.astype(jnp.float32)), axis=(-2, -1)))
      / jnp.maximum(
          jnp.sqrt(jnp.sum(jnp.square(
              captured["bam_full_head"].astype(jnp.float32)), axis=(-2, -1)) + _EPS),
          _EPS))

  frozen_denom = jax.lax.stop_gradient(
      jnp.mean(jnp.square(final_hidden), axis=-1, keepdims=True)
      + config.normalization_layer_epsilon)
  norm_scale, logits_kernel = _output_head_parameters(params)

  def path_loss(hidden):
    token_loss = _frozen_head_token_loss(
        hidden, frozen_denom, norm_scale, logits_kernel,
        batch["targets"], config)
    masked = token_loss * valid
    return jnp.sum(masked), jnp.sum(masked, axis=1) / valid_count

  nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
  nodes = jnp.asarray((nodes_np + 1.0) / 2.0, jnp.float32)
  weights = jnp.asarray(weights_np / 2.0, jnp.float32)

  def integrate_node(gradient_sum, node_weight):
    alpha, weight = node_weight
    (_, sequence_loss), gradient = jax.value_and_grad(
        path_loss, has_aux=True)(alpha * final_hidden)
    component_token = -_component_token_dot(components, gradient)
    embedding_token = -jnp.sum(embedding * gradient, axis=-1)
    component_sequence = _sequence_mean(component_token, valid)
    embedding_sequence = _sequence_mean(embedding_token, valid)
    gradient_l2 = _sequence_mean(
        jnp.sqrt(jnp.sum(jnp.square(gradient), axis=-1) + _EPS), valid)
    node_output = (
        sequence_loss,
        gradient_l2,
        component_sequence,
        embedding_sequence,
    )
    return gradient_sum + weight * gradient, node_output

  gradient0 = jnp.zeros_like(final_hidden)
  integrated_gradient, node_values = jax.lax.scan(
      integrate_node, gradient0, (nodes, weights))
  node_loss, node_gradient_l2, node_component_v, node_embedding_v = node_values

  component_norm_token = jnp.transpose(
      jnp.sqrt(jnp.sum(jnp.square(components), axis=-1) + _EPS),
      (1, 2, 0, 3))
  energy_token = component_norm_token / final_l2[..., None, None]
  contribution_token = -_component_token_dot(
      components, integrated_gradient)
  embedding_norm_token = jnp.sqrt(
      jnp.sum(jnp.square(embedding), axis=-1) + _EPS)
  embedding_energy_token = embedding_norm_token / final_l2
  embedding_contribution_token = -jnp.sum(
      embedding * integrated_gradient, axis=-1)

  energy = _sequence_mean(energy_token, valid)
  contribution = _sequence_mean(contribution_token, valid)
  embedding_energy = _sequence_mean(embedding_energy_token, valid)
  embedding_contribution = _sequence_mean(
      embedding_contribution_token, valid)
  contribution_total = embedding_contribution + jnp.sum(
      contribution, axis=(1, 2))
  contribution_normalized = (
      contribution / contribution_total[:, None, None])
  embedding_contribution_normalized = (
      embedding_contribution / contribution_total)

  endpoint_total, endpoint_loss = path_loss(final_hidden)
  del endpoint_total
  zero_total, zero_loss = path_loss(jnp.zeros_like(final_hidden))
  del zero_total
  production_loss = _sequence_mean(production_xent, valid)

  all_components = jnp.concatenate(
      (embedding[:, None],
       jnp.transpose(components, (1, 0, 3, 2, 4)).reshape(
           final_hidden.shape[0], _LAYERS * len(_COMPONENTS),
           final_hidden.shape[1], final_hidden.shape[2])),
      axis=1)
  normalized_components = (
      all_components / final_l2[:, None, :, None]
      * valid[:, None, :, None]
      / jnp.sqrt(valid_count[:, None, None, None]))
  flat_components = normalized_components.reshape(
      normalized_components.shape[:2] + (-1,))
  normalized_gram = jnp.einsum(
      "bix,bjx->bij", flat_components, flat_components,
      precision=jax.lax.Precision.HIGHEST)

  path_contribution = zero_loss - endpoint_loss
  ig_closure_error = contribution_total - path_contribution
  return {
      "energy": energy,
      "component_norm": _sequence_mean(component_norm_token, valid),
      "contribution": contribution,
      "contribution_normalized": contribution_normalized,
      "embedding_energy": embedding_energy,
      "embedding_norm": _sequence_mean(embedding_norm_token, valid),
      "embedding_contribution": embedding_contribution,
      "embedding_contribution_normalized": embedding_contribution_normalized,
      "energy_token": energy_token,
      "component_norm_token": component_norm_token,
      "contribution_token": contribution_token,
      "embedding_energy_token": embedding_energy_token,
      "embedding_norm_token": embedding_norm_token,
      "embedding_contribution_token": embedding_contribution_token,
      "normalized_gram": normalized_gram,
      "contribution_total": contribution_total,
      "path_contribution": path_contribution,
      "ig_closure_error": ig_closure_error,
      "endpoint_loss": endpoint_loss,
      "zero_loss": zero_loss,
      "production_loss": production_loss,
      "endpoint_loss_error": endpoint_loss - production_loss,
      "residual_closure_mean": _sequence_mean(
          residual_closure_token, valid),
      "residual_closure_max": jnp.max(
          jnp.where(valid, residual_closure_token, 0), axis=1),
      "bam_head_closure_mean": jnp.transpose(
          _sequence_mean(jnp.transpose(head_closure_token, (1, 2, 0)), valid),
          (0, 1)),
      "rounding_correction_ratio": jnp.transpose(
          _sequence_mean(jnp.transpose(rounding_ratio_token, (1, 2, 0)), valid),
          (0, 1)),
      "fetch_self_weight_min": jnp.min(
          captured["fetch_self_weight"], axis=(0, 2)),
      "fetch_self_weight_max": jnp.max(
          captured["fetch_self_weight"], axis=(0, 2)),
      "node_loss": jnp.swapaxes(node_loss, 0, 1),
      "node_gradient_l2": jnp.swapaxes(node_gradient_l2, 0, 1),
      "node_component_contribution": jnp.swapaxes(node_component_v, 0, 1),
      "node_embedding_contribution": jnp.swapaxes(node_embedding_v, 0, 1),
      "quadrature_nodes": nodes,
      "quadrature_weights": weights,
      "valid_count": valid_count,
  }


def _model_and_summary(
    model, params, batch, rng, config, quadrature_order):
  rng1, aqt_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      params,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=["residual_attribution"],
  )
  return _summarize_capture(
      collections, params, batch, xent, config, quadrature_order)


def _distribution(values: np.ndarray) -> dict[str, float]:
  values = np.asarray(values, np.float64)
  return {
      "min": float(np.min(values)),
      "mean": float(np.mean(values)),
      "p50": float(np.percentile(values, 50)),
      "p95": float(np.percentile(values, 95)),
      "max": float(np.max(values)),
  }


def _aggregate(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
  files = sorted(output_dir.glob("residual_attribution_batch_*.npz"))
  if not files:
    raise FileNotFoundError("no residual attribution batches")
  keys = (
      "energy", "contribution", "contribution_normalized",
      "component_norm", "embedding_energy", "embedding_contribution",
      "embedding_contribution_normalized", "contribution_total",
      "path_contribution", "ig_closure_error", "endpoint_loss_error",
      "residual_closure_mean", "residual_closure_max",
      "bam_head_closure_mean", "rounding_correction_ratio",
      "normalized_gram",
  )
  joined = {}
  for key in keys:
    values = []
    for path in files:
      with np.load(path) as data:
        values.append(np.asarray(data[key]))
    joined[key] = np.concatenate(values, axis=0)
  energy = joined["energy"]
  contribution = joined["contribution"]
  efficiency = (
      np.mean(contribution, axis=0)
      / np.maximum(np.mean(energy, axis=0), _EPS))
  return {
      "metadata": metadata,
      "closure": {
          name: _distribution(joined[name])
          for name in (
              "ig_closure_error", "endpoint_loss_error",
              "residual_closure_mean", "residual_closure_max",
              "bam_head_closure_mean", "rounding_correction_ratio")
      },
      "component_order": list(_COMPONENTS),
      "mean_energy_by_layer_component": np.mean(energy, axis=0).tolist(),
      "mean_contribution_by_layer_component": np.mean(
          contribution, axis=0).tolist(),
      "mean_normalized_contribution_by_layer_component": np.mean(
          joined["contribution_normalized"], axis=0).tolist(),
      "ratio_of_mean_contribution_to_mean_energy": efficiency.tolist(),
      "embedding": {
          "mean_energy": float(np.mean(joined["embedding_energy"])),
          "mean_contribution": float(
              np.mean(joined["embedding_contribution"])),
          "mean_normalized_contribution": float(
              np.mean(joined["embedding_contribution_normalized"])),
      },
      "normalized_gram_mean": np.mean(
          joined["normalized_gram"], axis=0).tolist(),
      "normalized_gram_closure": _distribution(
          np.sum(joined["normalized_gram"], axis=(1, 2))),
  }


def run(config) -> None:
  if config.logits_via_embedding:
    raise ValueError("residual attribution currently requires logits_dense")
  if config.num_decoder_layers != _LAYERS:
    raise ValueError(
        f"configured {config.num_decoder_layers} layers, expected {_LAYERS}")
  output_dir = Path(os.environ.get(
      "BAM_RESIDUAL_ATTR_OUTPUT_DIR", "/tmp/bam_residual_attribution"))
  output_dir.mkdir(parents=True, exist_ok=True)
  target_sequences = int(os.environ.get("BAM_RESIDUAL_ATTR_SEQUENCES", "128"))
  sequence_offset = int(os.environ.get("BAM_RESIDUAL_ATTR_SEQUENCE_OFFSET", "0"))
  batch_size = int(os.environ.get("BAM_RESIDUAL_ATTR_BATCH_SIZE", "2"))
  quadrature_order = int(os.environ.get("BAM_RESIDUAL_ATTR_IG_NODES", "8"))
  cohort_path = os.environ.get("BAM_RESIDUAL_ATTR_COHORT_PATH")
  iterator_batch_size = int(
      config.eval_per_device_batch_size * jax.local_device_count())
  if (target_sequences % batch_size or iterator_batch_size % batch_size):
    raise ValueError(
        f"invalid {target_sequences=}, {sequence_offset=}, {batch_size=}, "
        f"{iterator_batch_size=}")
  cohort = None
  if cohort_path:
    with np.load(cohort_path) as data:
      cohort = {
          key: np.asarray(data[key])
          for key in (
              "inputs", "targets", "inputs_position", "inputs_segmentation",
              "targets_segmentation", "sequence_hashes")
      }
    if sequence_offset + target_sequences > cohort["inputs"].shape[0]:
      raise ValueError(
          f"cohort has {cohort['inputs'].shape[0]} sequences, requested "
          f"[{sequence_offset}, {sequence_offset + target_sequences})")
  elif sequence_offset % iterator_batch_size:
    raise ValueError(
        "iterator-backed sequence_offset must be divisible by iterator batch "
        f"size {iterator_batch_size}")

  start = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  compiled = jax.jit(
      lambda params, batch, rng: _model_and_summary(
          model, params, batch, rng, config, quadrature_order))

  metadata = {
      "checkpoint": config.load_parameters_path,
      "checkpoint_trainer_commit": _TRAINER_COMMIT,
      "diagnostic_commit": os.environ.get(
          "BAM_RESIDUAL_ATTR_DIAGNOSTIC_COMMIT", "unknown"),
      "config_class": "BamResidualAttribution",
      "base_config_class": _BASE_CONFIG_CLASS,
      "sequences": target_sequences,
      "sequence_offset": sequence_offset,
      "batch_size": batch_size,
      "iterator_batch_size": iterator_batch_size,
      "data_shuffle_seed": config.data_shuffle_seed,
      "cohort_path": cohort_path,
      "cohort_sha256": (
          hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest()
          if cohort_path else None),
      "quadrature_order": quadrature_order,
      "component_order": list(_COMPONENTS),
      "component_semantics": {
          "mha": (
              "complete attention residual minus fetched-BAM components; "
              "includes LocalQK's nonlinear effect through Q/K/alpha and "
              "bf16 residual-add rounding"),
          "self_cross": (
              "same-position M[l,t] versus all other source-position M[l,s]; "
              "not write-record provenance"),
          "positive_contribution": "reduces loss along the frozen-RMS IG path",
      },
      "energy": (
          "per sample mean over valid tokens of ||z[l,c]||_2 / ||h_L||_2"),
      "ig_path": (
          "h(alpha)=alpha*h_L; final RMS denominator frozen at h_L; learned "
          "final RMS scale and logits head retained"),
      "raw_vector_storage": False,
      "device": [str(device) for device in jax.devices()],
      "setup_seconds": time.perf_counter() - start,
  }

  if cohort is None:
    for _ in range(sequence_offset // iterator_batch_size):
      next(eval_data_iterator)
  input_batch = None
  microbatches = iterator_batch_size // batch_size
  num_batches = target_sequences // batch_size
  output_offset = sequence_offset // batch_size
  for batch_index in range(num_batches):
    if cohort is None:
      microbatch_index = batch_index % microbatches
      if microbatch_index == 0:
        input_batch = next(eval_data_iterator)
      start_index = microbatch_index * batch_size
      batch = jax.tree.map(
          lambda value: value[start_index:start_index + batch_size], input_batch)
    else:
      start_index = sequence_offset + batch_index * batch_size
      end_index = start_index + batch_size
      batch = {
          key: jnp.asarray(cohort[key][start_index:end_index])
          for key in (
              "inputs", "targets", "inputs_position", "inputs_segmentation",
              "targets_segmentation")
      }
    rng = jax.random.fold_in(init_rng, output_offset + batch_index)
    batch_start = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      summary = compiled(state.params, batch, rng)
    (summary, inputs, targets, inputs_position, inputs_segmentation,
     targets_segmentation) = jax.device_get((
        summary,
        batch["inputs"],
        batch["targets"],
        batch["inputs_position"],
        batch["inputs_segmentation"],
        batch["targets_segmentation"],
    ))
    valid = targets_segmentation != 0
    sequence_hashes = np.asarray([
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in np.asarray(inputs)
    ])
    if cohort is not None:
      expected_hashes = cohort["sequence_hashes"][start_index:end_index]
      if not np.array_equal(sequence_hashes, expected_hashes):
        raise ValueError("fixed-cohort sequence hash mismatch")
    np.savez_compressed(
        output_dir
        / f"residual_attribution_batch_{output_offset + batch_index:03d}.npz",
        **{key: np.asarray(value) for key, value in summary.items()},
        inputs=np.asarray(inputs),
        targets=np.asarray(targets),
        inputs_position=np.asarray(inputs_position),
        inputs_segmentation=np.asarray(inputs_segmentation),
        targets_segmentation=np.asarray(targets_segmentation),
        valid=np.asarray(valid),
        sequence_hashes=sequence_hashes,
    )
    print(
        f"RESIDUAL_ATTR batch={batch_index + 1}/{num_batches} "
        f"endpoint_loss={np.mean(summary['endpoint_loss']):.6f} "
        f"ig_closure_max={np.max(np.abs(summary['ig_closure_error'])):.3e} "
        f"seconds={time.perf_counter() - batch_start:.1f}",
        flush=True,
    )

  metadata["total_seconds"] = time.perf_counter() - start
  report = _aggregate(output_dir, metadata)
  report_path = output_dir / "residual_attribution.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"RESIDUAL_ATTRIBUTION_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

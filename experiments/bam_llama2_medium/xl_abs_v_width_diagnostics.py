"""Read-only XL fetched-M width diagnostics on a fixed Pile-eval cohort.

The diagnostic stays outside ``BamAttention``.  It measures post-gate fetched
read scale/rank and uses a Flax method interceptor for paired row/column output
scaling ablations.  It can also replace each learned AbsV projection by its
rank-r SVD approximation without changing runtime shapes.
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
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


_LAYER_RE = re.compile(r"layers_(\d+)")
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


def _stack_captures(collections, num_layers: int):
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  flat = flatten_dict(collections.get("intermediates", {}))
  scanned = {}
  for path, raw in flat.items():
    layer = _layer_from_path(path)
    is_read = "_read_fetched_m" in path
    is_attention = "_query_chunk_op" in path or "_attention_block" in path
    if layer is None:
      value = _unwrap(raw)
      if is_read:
        if isinstance(value, (tuple, list)):
          value = value[0]
        scanned["read"] = value
      elif is_attention:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
          raise ValueError(
              f"unexpected scanned attention capture at {path}: {type(value)}")
        scanned["y_std"], scanned["mbar"] = value
      continue
    value = _unwrap(raw)
    if is_read:
      if isinstance(value, (tuple, list)):
        value = value[0]
      grouped[layer]["read"] = value
    elif is_attention:
      if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"unexpected attention capture at {path}: {type(value)}")
      grouped[layer]["y_std"], grouped[layer]["mbar"] = value
  expected = {"read", "y_std", "mbar"}
  if not grouped and set(scanned) == expected:
    for name, value in scanned.items():
      if value.shape[0] != num_layers:
        raise ValueError(
            f"scanned {name} has {value.shape[0]} layers, expected {num_layers}")
    return scanned
  if set(grouped) != set(range(num_layers)):
    shapes = {
        "/".join(path): jax.tree.map(lambda value: getattr(value, "shape", None), raw)
        for path, raw in flat.items()
    }
    raise ValueError(
        f"captured layers differ: {sorted(grouped)}; captures={shapes}")
  for layer, values in grouped.items():
    if set(values) != expected:
      raise ValueError(
          f"layer {layer} captures differ: {sorted(expected ^ set(values))}; "
          f"paths={sorted('/'.join(path) for path in flat)}")
    for name, value in values.items():
      if not hasattr(value, "shape"):
        structure = jax.tree.map(
            lambda leaf: (type(leaf).__name__, getattr(leaf, "shape", None)),
            value)
        raise ValueError(
            f"layer {layer} {name} is not an array: {structure}")
  return {
      name: jnp.stack([grouped[layer][name] for layer in range(num_layers)])
      for name in sorted(expected)
  }


def _stack_unscanned_captures(collections, num_layers: int):
  grouped: dict[int, dict[str, jax.Array]] = defaultdict(dict)
  flat = flatten_dict(collections.get("bam_energy", {}))

  def decode_attention(raw, path):
    value = _unwrap(raw)
    if (isinstance(value, (tuple, list)) and value
        and all(isinstance(chunk, (tuple, list)) and len(chunk) == 2
                for chunk in value)):
      y_chunks, mbar_chunks = zip(*value)
      return (
          jnp.concatenate(y_chunks, axis=1),
          jnp.concatenate(mbar_chunks, axis=-3),
      )
    if not isinstance(value, (tuple, list)) or len(value) != 2:
      raise ValueError(f"unexpected attention capture at {path}: {type(value)}")
    return value

  for path, raw in flat.items():
    layer = _layer_from_path(path)
    if layer is None:
      continue
    if path[-1] == "read":
      grouped[layer]["read"] = _unwrap(raw)
    elif path[-1] == "attention":
      grouped[layer]["y_std"], grouped[layer]["mbar"] = (
          decode_attention(raw, path))

  expected = {"read", "y_std", "mbar"}
  if set(grouped) != set(range(num_layers)):
    raise ValueError(
        f"captured layers differ: {sorted(grouped)}; "
        f"paths={sorted('/'.join(path) for path in flat)}")
  for layer, values in grouped.items():
    if set(values) != expected:
      raise ValueError(
          f"layer {layer} captures differ: {sorted(expected ^ set(values))}")
  return {
      name: jnp.stack([grouped[layer][name] for layer in range(num_layers)])
      for name in sorted(expected)
  }


def _masked_sums(x, mask):
  mask = mask.astype(jnp.float32)
  while mask.ndim < x.ndim:
    mask = mask[..., None]
  square = jnp.square(x.astype(jnp.float32)) * mask
  count = jnp.maximum(jnp.sum(mask) * np.prod(x.shape[3:]), 1.0)
  rms = jnp.sqrt(jnp.sum(square, axis=tuple(range(1, x.ndim))) / count)
  token_l2 = jnp.sqrt(jnp.sum(square, axis=-1))
  token_mask = jnp.broadcast_to(mask[..., 0], token_l2.shape)
  l2_count = jnp.maximum(
      jnp.sum(token_mask, axis=tuple(range(1, token_l2.ndim))), 1.0)
  mean_l2 = jnp.sum(token_l2 * token_mask, axis=tuple(range(1, token_l2.ndim))) / l2_count
  return rms, mean_l2, jnp.sum(square, axis=tuple(range(1, x.ndim)))


def _masked_dot(x, y, mask):
  mask = mask.astype(jnp.float32)
  while mask.ndim < x.ndim:
    mask = mask[..., None]
  return jnp.sum(
      x.astype(jnp.float32) * y.astype(jnp.float32) * mask,
      axis=tuple(range(1, x.ndim)))


def _covariance_spectrum(x, mask):
  """Return ascending covariance eigenvalues for [layer,b,t,item,width]."""
  x = x.astype(jnp.float32)
  mask = mask.astype(jnp.float32)[None, ..., None, None]
  x = x * mask
  covariance = jnp.einsum("lbtic,lbtid->lcd", x, x)
  count = jnp.maximum(
      jnp.sum(mask, axis=(1, 2, 3, 4)) * x.shape[-2], 1.0)
  covariance = covariance / count[:, None, None]
  return jnp.linalg.eigvalsh(covariance)


def _capture_metrics(captured, mask, bam_k, read_v_dim):
  read = captured["read"]
  y_std = captured["y_std"]
  mbar = captured["mbar"]
  y_col = read[..., :bam_k]
  y_row = read[..., bam_k:bam_k + read_v_dim]
  col_rms, col_l2, col_sq = _masked_sums(y_col, mask[None])
  row_rms, row_l2, row_sq = _masked_sums(y_row, mask[None])
  bam_rms, bam_l2, bam_sq = _masked_sums(read, mask[None])
  std_rms, std_l2, std_sq = _masked_sums(y_std, mask[None])
  combined_rms, combined_l2, combined_sq = _masked_sums(
      y_std + read, mask[None])
  bam_std_dot = _masked_dot(read, y_std, mask[None])
  return {
      "col_rms": col_rms,
      "row_rms": row_rms,
      "bam_rms": bam_rms,
      "std_rms": std_rms,
      "combined_rms": combined_rms,
      "col_mean_l2": col_l2,
      "row_mean_l2": row_l2,
      "bam_mean_l2": bam_l2,
      "std_mean_l2": std_l2,
      "combined_mean_l2": combined_l2,
      "col_to_std_frobenius": jnp.sqrt(col_sq / jnp.maximum(std_sq, _EPS)),
      "row_to_std_frobenius": jnp.sqrt(row_sq / jnp.maximum(std_sq, _EPS)),
      "bam_to_std_frobenius": jnp.sqrt(bam_sq / jnp.maximum(std_sq, _EPS)),
      "combined_to_std_frobenius": jnp.sqrt(
          combined_sq / jnp.maximum(std_sq, _EPS)),
      "bam_std_cosine": bam_std_dot / jnp.sqrt(
          jnp.maximum(bam_sq * std_sq, _EPS)),
      "col_spectrum": _covariance_spectrum(y_col, mask),
      "row_spectrum": _covariance_spectrum(y_row, mask),
      "mbar_v_spectrum": _covariance_spectrum(mbar, mask),
  }


def _sequence_loss(xent, mask):
  weights = jnp.sum(mask, axis=-1)
  return jnp.sum(xent * mask, axis=-1) / jnp.maximum(weights, 1)


def _scaled_forward(model, params, batch, rng, bam_k, read_v_dim,
                    col_scale, row_scale):
  def interceptor(next_fun, args, kwargs, context):
    output = next_fun(*args, **kwargs)
    if context.method_name != "_read_fetched_m":
      return output
    read, gate_logits = output
    scaled = jnp.concatenate((
        read[..., :bam_k] * jnp.asarray(col_scale, read.dtype),
        read[..., bam_k:bam_k + read_v_dim]
        * jnp.asarray(row_scale, read.dtype),
        read[..., bam_k + read_v_dim:],
    ), axis=-1)
    return scaled, gate_logits

  dropout_rng, params_rng = jax.random.split(rng)
  with nn.intercept_methods(interceptor):
    (xent, _, _), _ = model.apply(
        params,
        batch["inputs"],
        batch["inputs_position"],
        decoder_segment_ids=batch["inputs_segmentation"],
        decoder_target_mask=batch["targets_segmentation"],
        decoder_target_tokens=batch["targets"],
        enable_dropout=False,
        rngs={"dropout": dropout_rng, "params": params_rng},
        mutable=["bam_energy"],
    )
  return _sequence_loss(xent, batch["targets_segmentation"] != 0)


def _captured_forward(model, config, params, batch, rng, read_v_dim):
  attention_method = (
      "_query_chunk_op" if config.query_chunk_size is not None
      else "_attention_block")

  def capture_intermediates(_module, method_name):
    return method_name in ("_read_fetched_m", attention_method)

  dropout_rng, params_rng = jax.random.split(rng)
  apply_kwargs = dict(
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"], enable_dropout=False,
      rngs={"dropout": dropout_rng, "params": params_rng})
  if config.scan_layers:
    (xent, _, _), collections = model.apply(
        params, batch["inputs"], batch["inputs_position"],
        mutable=["intermediates"],
        capture_intermediates=capture_intermediates, **apply_kwargs)
    captured = _stack_captures(collections, config.num_decoder_layers)
  else:
    def interceptor(next_fun, args, kwargs, context):
      output = next_fun(*args, **kwargs)
      if context.method_name == "_attention_block":
        context.module.sow("bam_energy", "attention", output)
      return output

    with nn.intercept_methods(interceptor):
      (xent, _, _), collections = model.apply(
          params, batch["inputs"], batch["inputs_position"],
          mutable=["bam_energy"], **apply_kwargs)
    captured = _stack_unscanned_captures(
        collections, config.num_decoder_layers)
  mask = batch["targets_segmentation"] != 0
  return {
      "sequence_loss": _sequence_loss(xent, mask),
      "metrics": _capture_metrics(captured, mask, config.bam_k, read_v_dim),
  }


def _install_fetched_read_capture():
  """Sow dense fetched-read outputs without changing BamAttention source."""
  from layers import attentions

  original = attentions.BamAttention._read_fetched_m
  if getattr(original, "_bam_energy_capture", False):
    return

  def captured(self, *args, **kwargs):
    output = original(self, *args, **kwargs)
    if (not self.is_initializing()
        and self.is_mutable_collection("bam_energy")):
      self.sow("bam_energy", "read", output[0])
    return output

  captured._bam_energy_capture = True
  attentions.BamAttention._read_fetched_m = captured


def _stats(values):
  values = np.asarray(values, np.float64)
  return {
      "mean": float(np.mean(values)),
      "std": float(np.std(values)),
      "se": float(np.std(values, ddof=1) / np.sqrt(values.size)),
      "min": float(np.min(values)),
      "p50": float(np.percentile(values, 50)),
      "max": float(np.max(values)),
  }


def _rank_approximation(params, rank, num_layers, selected_layers=None):
  selected_layers = (
      set(range(num_layers)) if selected_layers is None else set(selected_layers))
  flat = flatten_dict(unfreeze(params))
  matches = [path for path in flat if path[-1] == "abs_v_cache_projection"]
  if len(matches) == num_layers:
    for path in matches:
      if _layer_from_path(path) not in selected_layers:
        continue
      matrix = np.asarray(jax.device_get(flat[path]))
      u, singular, vh = np.linalg.svd(
          matrix.astype(np.float32), full_matrices=False)
      singular[rank:] = 0
      flat[path] = jnp.asarray((u * singular[None]) @ vh, matrix.dtype)
    return freeze(unflatten_dict(flat))
  if len(matches) != 1:
    raise ValueError(f"expected one scanned or {num_layers} AbsV projections, found {matches}")
  path = matches[0]
  matrix = np.asarray(jax.device_get(flat[path]))
  layer_axes = [i for i, width in enumerate(matrix.shape) if width == num_layers]
  if len(layer_axes) != 1:
    raise ValueError(f"cannot identify layer axis in AbsV shape {matrix.shape}")
  layer_axis = layer_axes[0]
  matrices = np.moveaxis(matrix, layer_axis, 0)
  approximated = []
  for layer, value in enumerate(matrices):
    if layer in selected_layers:
      u, singular, vh = np.linalg.svd(
          value.astype(np.float32), full_matrices=False)
      singular[rank:] = 0
      value = (u * singular[None]) @ vh
    approximated.append(value)
  flat[path] = jnp.asarray(
      np.moveaxis(np.stack(approximated), 0, layer_axis), matrix.dtype)
  return freeze(unflatten_dict(flat))


def _projection_singular_values(params, num_layers):
  flat = flatten_dict(unfreeze(params))
  matches = [path for path in flat if path[-1] == "abs_v_cache_projection"]
  if not matches:
    return None
  if len(matches) == num_layers:
    ordered = sorted(matches, key=lambda path: _layer_from_path(path))
    return np.stack([
        np.linalg.svd(
            np.asarray(jax.device_get(flat[path]), np.float32),
            compute_uv=False)
        for path in ordered
    ])
  if len(matches) != 1:
    raise ValueError(
        f"expected one scanned or {num_layers} AbsV projections, found {matches}")
  matrix = np.asarray(jax.device_get(flat[matches[0]]), np.float32)
  layer_axes = [i for i, width in enumerate(matrix.shape) if width == num_layers]
  if len(layer_axes) != 1:
    raise ValueError(f"cannot identify layer axis in AbsV shape {matrix.shape}")
  matrices = np.moveaxis(matrix, layer_axes[0], 0)
  return np.stack([
      np.linalg.svd(value, compute_uv=False) for value in matrices
  ])


def _mean_tree(trees):
  return jax.tree.map(
      lambda *values: np.mean(np.stack(values), axis=0), *trees)


def _json_tree(value):
  if isinstance(value, dict):
    return {key: _json_tree(child) for key, child in value.items()}
  return np.asarray(value).tolist()


def run(config):
  if not config.only_eval:
    raise ValueError("xl_abs_v_width_diagnostics.py requires only_eval=True")
  _install_fetched_read_capture()
  fetched_read_heads = config.bam_fetched_read_num_heads or config.num_query_heads
  if fetched_read_heads != config.num_query_heads:
    raise ValueError("diagnostic currently requires one fetched head per MHA head")
  num_batches = int(os.environ.get("BAM_ABSV_DIAG_BATCHES", "32"))
  capture_batches = int(os.environ.get("BAM_ABSV_DIAG_CAPTURE_BATCHES", "8"))
  scales = tuple(float(x) for x in os.environ.get(
      "BAM_ABSV_DIAG_SCALES", "1,0.70710678,0.5,0.25").split(","))
  ranks = tuple(int(x) for x in os.environ.get(
      "BAM_ABSV_DIAG_RANKS", "8,16").split(",") if x)
  layerwise_rank = int(os.environ.get("BAM_ABSV_DIAG_LAYERWISE_RANK", "0"))
  output_path = Path(os.environ.get(
      "BAM_ABSV_DIAG_OUTPUT", "/tmp/xl_abs_v_width_diagnostics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  read_v_dim = config.bam_abs_v_compression_dim or config.bam_v

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  cohort_path = os.environ.get("BAM_ABSV_DIAG_COHORT_PATH")
  if cohort_path:
    with np.load(cohort_path) as cohort_file:
      cohort = {
          key: np.asarray(cohort_file[key])
          for key in (
              "inputs", "targets", "inputs_position", "inputs_segmentation",
              "targets_segmentation", "sequence_hashes")
      }
    batch_size = int(config.eval_per_device_batch_size * jax.device_count())
    required = num_batches * batch_size
    if required > cohort["inputs"].shape[0]:
      raise ValueError(
          f"fixed cohort has {cohort['inputs'].shape[0]} sequences, "
          f"but {required} are required")
    batches = [
        {
            key: jnp.asarray(value[start:start + batch_size])
            for key, value in cohort.items() if key != "sequence_hashes"
        }
        for start in range(0, required, batch_size)
    ]
  else:
    batches = [next(eval_iterator) for _ in range(num_batches)]
  sequence_hashes = []
  for batch in batches:
    inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in inputs)

  scaled = jax.jit(lambda params, batch, rng, col, row: _scaled_forward(
      model, params, batch, rng, config.bam_k, read_v_dim, col, row))
  modes = {"baseline": (1.0, 1.0)}
  for scale in scales:
    if scale == 1:
      continue
    modes[f"col_{scale:g}"] = (scale, 1.0)
    modes[f"row_{scale:g}"] = (1.0, scale)
    modes[f"both_{scale:g}"] = (scale, scale)

  losses = {}
  timings = {}
  for name, (col_scale, row_scale) in modes.items():
    values = []
    begin = time.perf_counter()
    for index, batch in enumerate(batches):
      rng = jax.random.fold_in(init_rng, index)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        result = scaled(state.params, batch, rng, col_scale, row_scale)
      values.extend(np.asarray(jax.device_get(result), np.float64))
    losses[name] = np.asarray(values)
    timings[name] = time.perf_counter() - begin
    print(
        f"BAM_ABSV_DIAG mode={name} loss={np.mean(values):.8f} "
        f"seconds={timings[name]:.1f}", flush=True)

  rank_losses = {}
  if config.bam_abs_v_compression_dim is not None:
    for rank in ranks:
      if rank >= config.bam_abs_v_compression_dim:
        continue
      rank_params = _rank_approximation(
          state.params, rank, config.num_decoder_layers)
      values = []
      begin = time.perf_counter()
      for index, batch in enumerate(batches):
        rng = jax.random.fold_in(init_rng, index)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          result = scaled(rank_params, batch, rng, 1.0, 1.0)
        values.extend(np.asarray(jax.device_get(result), np.float64))
      rank_losses[f"rank_{rank}"] = np.asarray(values)
      timings[f"rank_{rank}"] = time.perf_counter() - begin
      print(
          f"BAM_ABSV_DIAG mode=rank_{rank} loss={np.mean(values):.8f} "
          f"seconds={timings[f'rank_{rank}']:.1f}", flush=True)
    if 0 < layerwise_rank < config.bam_abs_v_compression_dim:
      for layer in range(1, config.num_decoder_layers):
        rank_params = _rank_approximation(
            state.params, layerwise_rank, config.num_decoder_layers, (layer,))
        values = []
        begin = time.perf_counter()
        for index, batch in enumerate(batches):
          rng = jax.random.fold_in(init_rng, index)
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            result = scaled(rank_params, batch, rng, 1.0, 1.0)
          values.extend(np.asarray(jax.device_get(result), np.float64))
        name = f"layer_{layer:02d}_rank_{layerwise_rank}"
        rank_losses[name] = np.asarray(values)
        timings[name] = time.perf_counter() - begin
        print(
            f"BAM_ABSV_DIAG mode={name} loss={np.mean(values):.8f} "
            f"seconds={timings[name]:.1f}", flush=True)

  capture = jax.jit(lambda params, batch, rng: _captured_forward(
      model, config, params, batch, rng, read_v_dim))
  capture_trees = []
  for index, batch in enumerate(batches[:capture_batches]):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      result = capture(
          state.params, batch, jax.random.fold_in(init_rng, index))
    capture_trees.append(jax.device_get(result["metrics"]))
  capture_metrics = _mean_tree(capture_trees)

  baseline = losses["baseline"]
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "exp_class": config.exp_class,
          "code_commit": os.environ.get("BAM_ABSV_DIAG_CODE_COMMIT", ""),
          "checkpoint_trainer_commit": os.environ.get(
              "BAM_ABSV_DIAG_TRAINER_COMMIT", ""),
          "cohort_path": cohort_path,
          "cohort_sha256": (
              hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest()
              if cohort_path else None),
          "num_sequences": len(sequence_hashes),
          "capture_sequences": capture_batches * config.eval_per_device_batch_size,
          "sequence_hashes": sequence_hashes,
          "bam_k": config.bam_k,
          "bam_v": config.bam_v,
          "read_v_dim": read_v_dim,
          "timing_seconds": timings,
          "elapsed_seconds": time.perf_counter() - started,
      },
      "modes": {},
      "capture_metrics": _json_tree(capture_metrics),
      "abs_v_projection_singular_values": _json_tree(
          _projection_singular_values(state.params, config.num_decoder_layers)),
  }
  for name, values in {**losses, **rank_losses}.items():
    delta = values - baseline
    report["modes"][name] = {
        "loss": _stats(values),
        "delta_vs_baseline": _stats(delta),
        "sequence_loss": values.tolist(),
        "sequence_delta_vs_baseline": delta.tolist(),
    }
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_ABSV_DIAG_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

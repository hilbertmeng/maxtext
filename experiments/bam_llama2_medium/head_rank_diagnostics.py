"""Layerwise head-rank capture for BAM fetch reads and MHA residual contributions.

Only BAM tensors are compared in their native shared U/V coordinates.  Cross-model
comparisons use each head's contribution after its own W_O block, which places BAM
and MHA heads in the common residual space R^D.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import exp
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator


_LAYERS = 24
_HEADS = 16
_HEAD_DIM = 64
_EMBED = 1024
_K = 32
_C = 8
_QUERY_SAMPLES = 32
_LAYER_RE = re.compile(r"layers_(\d+)")


class BamLlama2MediumV2HeadRankDiagnostics(exp.BamLlama2MediumV2):
  """Read-only BAM head-rank capture on the completed V2 checkpoint."""

  bam_head_rank_diagnostics = True
  bam_head_rank_ablation = False
  bam_readout_attribution = False
  bam_diagnostics = False
  scan_layers = False
  eval_per_device_batch_size = 8.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_head_rank_tb/"


class Llama2MediumHeadRankDiagnostics(exp.Llama2Medium):
  """Read-only MHA contribution-space control."""

  bam_head_rank_diagnostics = True
  scan_layers = False
  eval_per_device_batch_size = 8.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/mha_head_rank_tb/"


exp.BamLlama2MediumV2HeadRankDiagnostics = BamLlama2MediumV2HeadRankDiagnostics
exp.Llama2MediumHeadRankDiagnostics = Llama2MediumHeadRankDiagnostics


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


def _stack_collection(collections: dict[str, Any]) -> dict[str, jax.Array]:
  grouped: dict[int, dict[str, Any]] = defaultdict(dict)
  for path, value in traverse_util.flatten_dict(
      collections["bam_head_rank"]).items():
    layer = _layer_from_path(path)
    if layer is not None:
      grouped[layer][path[-1]] = _unwrap(value)
  if set(grouped) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} layers, got {sorted(grouped)}")
  names = set(grouped[0])
  for layer, values in grouped.items():
    if set(values) != names:
      raise ValueError(f"layer {layer} capture mismatch: {set(values) ^ names}")
  return {
      name: jnp.stack([grouped[layer][name] for layer in range(_LAYERS)])
      for name in sorted(names)
  }


def _out_kernels(params: dict[str, Any]) -> jax.Array:
  by_layer = {}
  for path, value in traverse_util.flatten_dict(params["params"]).items():
    layer = _layer_from_path(path)
    if (layer is not None and "self_attention" in path
        and path[-2:] == ("out", "kernel")):
      by_layer[layer] = value
  if set(by_layer) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} W_O kernels, got {sorted(by_layer)}")
  kernels = jnp.stack([by_layer[layer] for layer in range(_LAYERS)])
  if kernels.shape != (_LAYERS, _HEADS, _HEAD_DIM, _EMBED):
    raise ValueError(f"unexpected W_O shape {kernels.shape}")
  return kernels.astype(jnp.float32)


def _query_indices(length: int) -> jax.Array:
  if length % _QUERY_SAMPLES:
    raise ValueError(f"length {length} is not divisible by {_QUERY_SAMPLES}")
  return ((jnp.arange(_QUERY_SAMPLES) + 1)
          * (length // _QUERY_SAMPLES) - 1)


def _residual_contributions(head_output: jax.Array, w_o: jax.Array) -> jax.Array:
  return jnp.einsum("lbqnh,lnhe->lbqne", head_output.astype(jnp.float32), w_o)


def _gram_summary(values: jax.Array, valid: jax.Array) -> dict[str, jax.Array]:
  """Return compact per-sequence and per-token head-space summaries."""
  values = values.astype(jnp.float32)
  gram = jnp.einsum("lbqnd,lbqmd->lbqnm", values, values)
  mask = valid[None, :, :, None, None].astype(jnp.float32)
  gram = gram * mask
  sequence_gram = jnp.sum(gram, axis=2)

  center = (jnp.eye(_HEADS, dtype=jnp.float32)
            - jnp.ones((_HEADS, _HEADS), jnp.float32) / _HEADS)
  centered = jnp.einsum("ij,lbqjk,km->lbqim", center, gram, center)
  eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(gram), 0)[..., ::-1]
  centered_eigenvalues = jnp.maximum(
      jnp.linalg.eigvalsh(centered), 0)[..., ::-1]

  diagonal = jnp.maximum(jnp.diagonal(gram, axis1=-2, axis2=-1), 0)
  denominator = jnp.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
  cosine = jnp.where(denominator > 0, gram / denominator, 0)
  off_diagonal = 1 - jnp.eye(_HEADS, dtype=jnp.float32)
  count = _HEADS * (_HEADS - 1)
  cosine_mean = jnp.sum(cosine * off_diagonal, axis=(-2, -1)) / count
  cosine_abs_mean = jnp.sum(
      jnp.abs(cosine) * off_diagonal, axis=(-2, -1)) / count
  cosine_mean *= valid[None]
  cosine_abs_mean *= valid[None]
  return {
      "sequence_gram": sequence_gram,
      "local_eigenvalues": eigenvalues,
      "local_centered_eigenvalues": centered_eigenvalues,
      "local_cosine_mean": cosine_mean,
      "local_cosine_abs_mean": cosine_abs_mean,
  }


def _summarize_capture(
    captured: dict[str, jax.Array], w_o: jax.Array, valid: jax.Array,
    model_kind: str,
) -> dict[str, jax.Array]:
  mha_contribution = _residual_contributions(
      captured["mha_head_output"], w_o)
  values = {f"{model_kind}_mha_residual": mha_contribution}

  if model_kind == "bam":
    native = captured["fetch_native_output"].astype(jnp.float32)
    col_output, row_output = jnp.split(native, [_K], axis=-1)
    row_width = row_output.shape[-1]
    if row_width != _C:
      raise ValueError(f"expected compressed row width {_C}, got {row_width}")

    post_rms = captured["fetch_key_post_rms_pre_gate"].astype(jnp.float32)
    post_gate = captured["fetch_key_post_gate"].astype(jnp.float32)
    row_key_rms, col_key_rms = jnp.split(post_rms, [_K], axis=-1)
    row_key_gate, col_key_gate = jnp.split(post_gate, [_K], axis=-1)

    zeros_k = jnp.zeros(col_output.shape[:-1] + (_HEAD_DIM - _K,), jnp.float32)
    col_head = jnp.concatenate((col_output, zeros_k), axis=-1)
    row_head = jnp.pad(
        row_output,
        [(0, 0)] * (row_output.ndim - 1)
        + [(_K, _HEAD_DIM - _K - row_width)])
    col_contribution = _residual_contributions(col_head, w_o)
    row_contribution = _residual_contributions(row_head, w_o)

    values.update({
        "bam_row_key_post_rms": row_key_rms,
        "bam_col_key_post_rms": col_key_rms,
        "bam_row_key_post_gate": row_key_gate,
        "bam_col_key_post_gate": col_key_gate,
        "bam_row_native_read": row_output,
        "bam_col_native_read": col_output,
        "bam_row_residual": row_contribution,
        "bam_col_residual": col_contribution,
        "bam_fetch_residual": row_contribution + col_contribution,
    })

  output = {}
  for name, value in values.items():
    for metric, array in _gram_summary(value, valid).items():
      output[f"{name}__{metric}"] = array
  return output


def _model_loss_and_capture(model, params, batch, rng):
  rng1, aqt_rng = jax.random.split(rng)
  (xent, _, _), collections = model.apply(
      dict(params),
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable=["bam_head_rank"],
  )
  mask = batch["targets_segmentation"] != 0
  return (
      jnp.sum(xent * mask) / jnp.maximum(jnp.sum(mask), 1),
      collections,
  )


def run(config) -> None:
  model_kind = os.environ.get("BAM_HEAD_RANK_MODEL", "bam").lower()
  if model_kind not in ("bam", "mha"):
    raise ValueError(f"unknown BAM_HEAD_RANK_MODEL={model_kind}")
  output_dir = Path(os.environ.get("BAM_HEAD_RANK_OUTPUT_DIR", "/tmp/bam_head_rank"))
  output_dir.mkdir(parents=True, exist_ok=True)
  target_sequences = int(os.environ.get("BAM_HEAD_RANK_SEQUENCES", "128"))

  start = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  w_o = _out_kernels(state.params)
  batch_size = int(config.eval_per_device_batch_size * jax.local_device_count())
  if target_sequences % batch_size:
    raise ValueError(f"{target_sequences=} not divisible by {batch_size=}")
  num_batches = target_sequences // batch_size

  compiled_capture = jax.jit(_model_loss_and_capture, static_argnums=0)
  compiled_summary = jax.jit(
      lambda captured, kernels, valid: _summarize_capture(
          captured, kernels, valid, model_kind))
  metadata = {
      "model_kind": model_kind,
      "checkpoint": config.load_parameters_path,
      "diagnostic_commit": os.environ.get("BAM_HEAD_RANK_COMMIT", "unknown"),
      "sequences": target_sequences,
      "batch_size": batch_size,
      "query_samples_per_sequence": _QUERY_SAMPLES,
      "query_positions": np.asarray(
          _query_indices(config.max_target_length)).tolist(),
      "data_shuffle_seed": config.data_shuffle_seed,
      "device": [str(device) for device in jax.devices()],
      "setup_seconds": time.perf_counter() - start,
      "space_policy": {
          "bam_native": "shared U/V coordinates; direct head comparison is valid",
          "cross_model": "per-head y @ W_O[h] contribution in residual R^D",
          "mha_native": "not analyzed because each head has private coordinates",
      },
  }

  for batch_index in range(num_batches):
    batch_start = time.perf_counter()
    batch = next(eval_data_iterator)
    rng = jax.random.fold_in(init_rng, batch_index)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      loss, collections = compiled_capture(model, state.params, batch, rng)
      captured = _stack_collection(collections)
      query_indices = _query_indices(batch["inputs"].shape[1])
      valid = (batch["targets_segmentation"][:, query_indices] != 0)
      summaries = compiled_summary(captured, w_o, valid)
    loss, summaries, valid_host, inputs_host = jax.device_get((
        loss, summaries, valid, batch["inputs"]))
    hashes = np.asarray([
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in np.asarray(inputs_host)
    ])
    np.savez_compressed(
        output_dir / f"{model_kind}_head_rank_batch_{batch_index:03d}.npz",
        loss=np.asarray(loss, np.float32),
        valid=np.asarray(valid_host),
        sequence_hashes=hashes,
        **{name: np.asarray(value, np.float32)
           for name, value in summaries.items()},
    )
    print(
        f"HEAD_RANK model={model_kind} batch={batch_index + 1}/{num_batches} "
        f"loss={float(loss):.6f} seconds={time.perf_counter() - batch_start:.1f}",
        flush=True)

  metadata["total_seconds"] = time.perf_counter() - start
  (output_dir / f"{model_kind}_capture_metadata.json").write_text(
      json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(f"HEAD_RANK_DONE model={model_kind} output={output_dir}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

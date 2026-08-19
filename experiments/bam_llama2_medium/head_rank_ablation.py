"""Held-out loss test for fixed low-rank BAM fetch-head bases."""

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
_RANKS = (2, 4, 8, 12)
_LAYER_RE = re.compile(r"layers_(\d+)")


class BamLlama2MediumV2HeadRankAblation(exp.BamLlama2MediumV2):
  """Read-only V2 with diagnostic head projections on fetched runtime keys."""

  bam_head_rank_diagnostics = False
  bam_head_rank_ablation = True
  bam_readout_attribution = False
  bam_diagnostics = False
  scan_layers = False
  eval_per_device_batch_size = 8.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_head_rank_ablation_tb/"


exp.BamLlama2MediumV2HeadRankAblation = BamLlama2MediumV2HeadRankAblation


def _layer_from_path(path: tuple[str, ...]) -> int | None:
  for part in path:
    match = _LAYER_RE.fullmatch(part)
    if match:
      return int(match.group(1))
  return None


def _attention_paths(params: dict[str, Any]) -> dict[int, tuple[str, ...]]:
  paths = {}
  for path in traverse_util.flatten_dict(params["params"]):
    layer = _layer_from_path(path)
    if layer is not None and path[-1] == "gw_b0" and "self_attention" in path:
      paths[layer] = path[:-1]
  if set(paths) != set(range(_LAYERS)):
    raise ValueError(f"expected {_LAYERS} BAM attention paths, got {sorted(paths)}")
  return paths


def _projection(vectors: np.ndarray, rank: int) -> np.ndarray:
  basis = vectors[:, :rank]
  return basis @ basis.T


def _make_perturbations(
    params: dict[str, Any], row_vectors: np.ndarray, col_vectors: np.ndarray,
    arm: str, rank: int,
) -> dict[str, Any]:
  if arm not in ("baseline", "row", "col", "both"):
    raise ValueError(arm)
  identity = np.eye(_HEADS, dtype=np.float32)
  leaves = {}
  for layer, parent in _attention_paths(params).items():
    row_projection = (
        _projection(row_vectors[layer], rank)
        if arm in ("row", "both") else identity)
    col_projection = (
        _projection(col_vectors[layer], rank)
        if arm in ("col", "both") else identity)
    leaves[parent + ("fetch_row_head_projection_delta",)] = jnp.asarray(
        row_projection - identity, jnp.bfloat16)
    leaves[parent + ("fetch_col_head_projection_delta",)] = jnp.asarray(
        col_projection - identity, jnp.bfloat16)
  return traverse_util.unflatten_dict(leaves)


def _loss(model, params, perturbations, batch, rng):
  variables = dict(params)
  variables["perturbations"] = perturbations
  rng1, aqt_rng = jax.random.split(rng)
  xent, _, _ = model.apply(
      variables,
      batch["inputs"],
      batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      decoder_target_mask=batch["targets_segmentation"],
      decoder_target_tokens=batch["targets"],
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
  )
  mask = batch["targets_segmentation"] != 0
  return jnp.sum(xent * mask), jnp.sum(mask)


def run(config) -> None:
  output_dir = Path(os.environ.get(
      "BAM_HEAD_RANK_ABLATION_OUTPUT_DIR", "/tmp/bam_head_rank_ablation"))
  output_dir.mkdir(parents=True, exist_ok=True)
  basis_path = Path(os.environ["BAM_HEAD_RANK_BASIS_PATH"])
  with np.load(basis_path) as basis_data:
    row_vectors = basis_data["row_key_vectors"]
    col_vectors = basis_data["col_key_vectors"]
    expected_hashes = basis_data["heldout_sequence_hashes"].tolist()
  if row_vectors.shape != (_LAYERS, _HEADS, _HEADS):
    raise ValueError(f"bad row basis shape {row_vectors.shape}")
  if col_vectors.shape != (_LAYERS, _HEADS, _HEADS):
    raise ValueError(f"bad col basis shape {col_vectors.shape}")

  start = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  batch_size = int(config.eval_per_device_batch_size * jax.local_device_count())
  if len(expected_hashes) % batch_size:
    raise ValueError("held-out cohort is not divisible by eval batch size")
  train_sequences = len(expected_hashes)
  for _ in range(train_sequences // batch_size):
    next(eval_data_iterator)

  variants = {"baseline": _make_perturbations(
      state.params, row_vectors, col_vectors, "baseline", _HEADS)}
  for arm in ("row", "col", "both"):
    for rank in _RANKS:
      variants[f"{arm}_rank_{rank}"] = _make_perturbations(
          state.params, row_vectors, col_vectors, arm, rank)
  compiled_loss = jax.jit(_loss, static_argnums=0)
  totals = defaultdict(float)
  total_weights = 0.0
  observed_hashes = []
  num_batches = len(expected_hashes) // batch_size

  for batch_index in range(num_batches):
    batch_start = time.perf_counter()
    batch = next(eval_data_iterator)
    batch_hashes = [
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
        for sequence in np.asarray(jax.device_get(batch["inputs"]))
    ]
    observed_hashes.extend(batch_hashes)
    rng = jax.random.fold_in(init_rng, batch_index)
    batch_results = {}
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      for name, perturbations in variants.items():
        batch_results[name] = compiled_loss(
            model, state.params, perturbations, batch, rng)
    batch_results = jax.device_get(batch_results)
    weights = float(batch_results["baseline"][1])
    total_weights += weights
    for name, (loss_sum, variant_weights) in batch_results.items():
      if float(variant_weights) != weights:
        raise ValueError(f"weight mismatch for {name}")
      totals[name] += float(loss_sum)
    print(
        f"HEAD_RANK_ABLATION batch={batch_index + 1}/{num_batches} "
        f"baseline={float(batch_results['baseline'][0]) / weights:.6f} "
        f"seconds={time.perf_counter() - batch_start:.1f}", flush=True)

  if observed_hashes != expected_hashes:
    raise ValueError("held-out cohort hashes do not match the capture")
  losses = {name: total / total_weights for name, total in totals.items()}
  baseline = losses["baseline"]
  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "diagnostic_commit": os.environ.get("BAM_HEAD_RANK_COMMIT", "unknown"),
          "basis_path": str(basis_path),
          "heldout_sequences": len(expected_hashes),
          "batch_size": batch_size,
          "sequence_hashes": expected_hashes,
          "total_weights": total_weights,
          "total_seconds": time.perf_counter() - start,
      },
      "loss": losses,
      "loss_delta_vs_baseline": {
          name: loss - baseline for name, loss in losses.items()
      },
  }
  report_path = output_dir / "head_rank_ablation.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"HEAD_RANK_ABLATION_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv) -> None:
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

"""Basis-invariant held-out rank ablation of the fetched-M AbsV axis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


def _sequence_loss(xent, mask):
  weights = jnp.sum(mask, axis=-1)
  return jnp.sum(xent * mask, axis=-1) / jnp.maximum(weights, 1)


def _loss(model, params, batch, rng):
  dropout_rng, params_rng = jax.random.split(rng)
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
  return _sequence_loss(xent, batch["targets_segmentation"] != 0)


def _projected_loss(
    model, params, batch, rng, rank: int, calibration_first: bool
):
  batch_size = batch["inputs"].shape[0]
  if batch_size % 2:
    raise ValueError(f"rank ablation requires an even batch, got {batch_size}")
  midpoint = batch_size // 2
  calibration = slice(0, midpoint) if calibration_first else slice(midpoint, None)

  def interceptor(next_fun, args, kwargs, context):
    if context.method_name != "_read_fetched_m":
      return next_fun(*args, **kwargs)
    mbar = args[0]
    width = mbar.shape[-1]
    if not 0 < rank < width:
      raise ValueError(f"rank must be in (0, {width}), got {rank}")
    calibration_mbar = mbar[calibration].astype(jnp.float32)
    covariance = jnp.einsum(
        "btkc,btkd->cd", calibration_mbar, calibration_mbar,
        preferred_element_type=jnp.float32)
    covariance /= np.prod(calibration_mbar.shape[:-1])
    _, eigenvectors = jnp.linalg.eigh(covariance)
    basis = eigenvectors[:, -rank:]
    projector = basis @ basis.T
    projected = jnp.einsum(
        "btkc,cd->btkd", mbar, projector.astype(mbar.dtype))
    return next_fun(projected, *args[1:], **kwargs)

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
  return _sequence_loss(xent, batch["targets_segmentation"] != 0)


def _fixed_batches(path: Path, batch_size: int) -> tuple[list[dict], np.ndarray]:
  with np.load(path) as cohort_file:
    cohort = {
        key: np.asarray(cohort_file[key])
        for key in (
            "inputs", "targets", "inputs_position", "inputs_segmentation",
            "targets_segmentation", "sequence_hashes")
    }
  if cohort["inputs"].shape[0] % batch_size:
    raise ValueError(
        f"cohort size {cohort['inputs'].shape[0]} is not divisible by {batch_size}")
  batches = [
      {
          key: jnp.asarray(value[start:start + batch_size])
          for key, value in cohort.items() if key != "sequence_hashes"
      }
      for start in range(0, cohort["inputs"].shape[0], batch_size)
  ]
  return batches, cohort["sequence_hashes"]


def _stats(values):
  values = np.asarray(values, np.float64)
  return {
      "mean": float(np.mean(values)),
      "se": float(np.std(values, ddof=1) / np.sqrt(values.size)),
      "min": float(np.min(values)),
      "p50": float(np.percentile(values, 50)),
      "max": float(np.max(values)),
  }


def _bootstrap_ci(values, seed=20260904, draws=20_000):
  values = np.asarray(values, np.float64)
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, values.size, size=(draws, values.size))
  return np.quantile(np.mean(values[indices], axis=1), (0.025, 0.975)).tolist()


def run(config):
  if not config.only_eval:
    raise ValueError("subspace ablation requires only_eval=True")
  cohort_path = Path(os.environ["BAM_ABSV_SUBSPACE_COHORT_PATH"])
  output_path = Path(os.environ.get(
      "BAM_ABSV_SUBSPACE_OUTPUT", "/tmp/xl_abs_v_subspace_ablation.json"))
  ranks = tuple(
      int(value) for value in os.environ.get(
          "BAM_ABSV_SUBSPACE_RANKS", "8,16").split(",") if value)
  batch_size = int(os.environ.get("BAM_ABSV_SUBSPACE_BATCH_SIZE", "16"))

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, _ = create_data_iterator(config, mesh)
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  batches, sequence_hashes = _fixed_batches(cohort_path, batch_size)

  baseline_fn = jax.jit(lambda params, batch, rng: _loss(
      model, params, batch, rng))
  rank_fns = {
      (rank, calibration_first): jax.jit(
          lambda params, batch, rng, rank=rank,
          calibration_first=calibration_first: _projected_loss(
              model, params, batch, rng, rank, calibration_first))
      for rank in ranks
      for calibration_first in (True, False)
      if rank < (config.bam_abs_v_compression_dim or config.bam_v)
  }

  baseline_values = []
  projected_values = {rank: [] for rank, _ in rank_fns}
  projected_baselines = {rank: [] for rank, _ in rank_fns}
  for batch_index, batch in enumerate(batches):
    rng = jax.random.fold_in(init_rng, batch_index)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      baseline = np.asarray(jax.device_get(
          baseline_fn(state.params, batch, rng)), np.float64)
    baseline_values.extend(baseline)
    midpoint = batch_size // 2
    for rank in projected_values:
      for calibration_first, heldout in (
          (True, slice(midpoint, None)), (False, slice(0, midpoint))):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          projected = np.asarray(jax.device_get(
              rank_fns[(rank, calibration_first)](
                  state.params, batch, rng)), np.float64)
        projected_values[rank].extend(projected[heldout])
        projected_baselines[rank].extend(baseline[heldout])
    print(
        f"BAM_ABSV_SUBSPACE batch={batch_index + 1}/{len(batches)}",
        flush=True)

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "checkpoint_trainer_commit": os.environ.get(
              "BAM_ABSV_SUBSPACE_TRAINER_COMMIT", ""),
          "diagnostic_commit": os.environ.get(
              "BAM_ABSV_SUBSPACE_DIAGNOSTIC_COMMIT", ""),
          "exp_class": config.exp_class,
          "cohort_path": str(cohort_path),
          "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
          "sequence_hashes": sequence_hashes.tolist(),
          "batch_size": batch_size,
          "calibration_sequences_per_fold": batch_size // 2,
          "heldout_sequences": len(baseline_values),
          "elapsed_seconds": time.perf_counter() - started,
      },
      "baseline_loss": _stats(baseline_values),
      "rank_ablation": {},
  }
  for rank in sorted(projected_values):
    projected = np.asarray(projected_values[rank])
    baseline = np.asarray(projected_baselines[rank])
    delta = projected - baseline
    report["rank_ablation"][f"rank_{rank}"] = {
        "projected_loss": _stats(projected),
        "paired_delta": _stats(delta),
        "paired_delta_bootstrap_95ci": _bootstrap_ci(delta, seed=20260904 + rank),
        "sequence_delta": delta.tolist(),
    }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_ABSV_SUBSPACE_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

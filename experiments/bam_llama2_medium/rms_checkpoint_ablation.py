"""Paired checkpoint loss ablation for BAM fp32 versus historical bf16 RMS statistics."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import normalizations
import train


_MODE_CALLERS = {
    "current": frozenset(),
    "legacy_read": frozenset(("_transform_bam_read_key",)),
    "legacy_write": frozenset(("_write",)),
    "legacy_read_write": frozenset(("_transform_bam_read_key", "_write")),
}


def _legacy_rms_norm(x, *, dtype, epsilon=1e-6, axis=-1):
  return jnp.asarray(
      x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=axis, keepdims=True) + epsilon),
      dtype,
  )


def _selective_rms_norm(mode, current_rms_norm):
  legacy_callers = _MODE_CALLERS[mode]

  def rms_norm(x, *, dtype, epsilon=1e-6, axis=-1):
    caller = inspect.currentframe().f_back.f_code.co_name
    fn = _legacy_rms_norm if caller in legacy_callers else current_rms_norm
    return fn(x, dtype=dtype, epsilon=epsilon, axis=axis)

  return rms_norm


def _forward(model, params, batch, rng):
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
  mask = batch["targets_segmentation"] != 0
  sequence_weights = jnp.sum(mask, axis=-1)
  sequence_loss = jnp.sum(xent * mask, axis=-1) / jnp.maximum(sequence_weights, 1)
  return sequence_loss, sequence_weights


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


def run(config):
  if not config.only_eval:
    raise ValueError("rms_checkpoint_ablation.py requires only_eval=True")
  num_batches = int(os.environ.get("BAM_RMS_ABLATION_BATCHES", "4"))
  output_dir = Path(os.environ.get("BAM_RMS_ABLATION_OUTPUT_DIR", "/tmp/bam_rms_ablation"))
  output_dir.mkdir(parents=True, exist_ok=True)

  init_rng, writer, checkpoint_manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  if eval_data_iterator is None:
    raise ValueError("Pile eval iterator is disabled; eval_interval must be positive")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)

  batches = [next(eval_data_iterator) for _ in range(num_batches)]
  sequence_hashes = []
  for batch in batches:
    inputs = np.asarray(jax.device_get(batch["inputs"]))
    sequence_hashes.extend(
        hashlib.sha256(sequence.tobytes()).hexdigest()[:16] for sequence in inputs)

  original_rms_norm = normalizations.rms_norm
  losses = {}
  timings = {}
  try:
    for mode in _MODE_CALLERS:
      normalizations.rms_norm = (
          original_rms_norm
          if mode == "current"
          else _selective_rms_norm(mode, original_rms_norm)
      )
      jax.clear_caches()
      compiled_forward = jax.jit(
          lambda params, batch, rng: _forward(model, params, batch, rng))
      mode_losses = []
      start = time.perf_counter()
      for batch_index, batch in enumerate(batches):
        rng = jax.random.fold_in(init_rng, batch_index)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          sequence_loss, _ = compiled_forward(state.params, batch, rng)
        mode_losses.extend(np.asarray(jax.device_get(sequence_loss), np.float64))
      timings[mode] = time.perf_counter() - start
      losses[mode] = np.asarray(mode_losses)
      print(
          f"BAM_RMS_ABLATION mode={mode} loss={np.mean(losses[mode]):.8f} "
          f"seconds={timings[mode]:.1f}",
          flush=True,
      )
  finally:
    normalizations.rms_norm = original_rms_norm

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "num_batches": num_batches,
          "num_sequences": len(sequence_hashes),
          "eval_per_device_batch_size": config.eval_per_device_batch_size,
          "eval_shuffle_buffer_size": config.eval_shuffle_buffer_size,
          "data_shuffle_seed": config.data_shuffle_seed,
          "sequence_hashes": sequence_hashes,
          "device_count": jax.device_count(),
          "timing_seconds": timings,
      },
      "modes": {},
  }
  current = losses["current"]
  for mode, values in losses.items():
    delta = values - current
    report["modes"][mode] = {
        "loss": _stats(values),
        "delta_vs_current": _stats(delta),
        "sequence_loss": values.tolist(),
        "sequence_delta_vs_current": delta.tolist(),
    }
  report_path = output_dir / "rms_checkpoint_ablation.json"
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_RMS_ABLATION_DONE report={report_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

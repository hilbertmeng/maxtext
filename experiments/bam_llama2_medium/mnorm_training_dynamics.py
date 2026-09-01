"""Trace early training dynamics for whole-M read-normalization ablations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import max_utils
import maxtext_utils
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
import train


def _global_l2(tree):
  return jnp.sqrt(sum(
      jnp.sum(jnp.square(value.astype(jnp.float32)))
      for value in jax.tree.leaves(tree)))


def _w_r_stats(tree, bam_k):
  row_square = jnp.asarray(0.0, jnp.float32)
  col_square = jnp.asarray(0.0, jnp.float32)
  row_count = 0
  col_count = 0
  layer_rms = []
  for path, value in flatten_dict(tree).items():
    names = tuple(str(part) for part in path)
    if names[-2:] != ("W_R", "kernel"):
      continue
    value = value.astype(jnp.float32)
    row = value[..., :bam_k]
    col = value[..., bam_k:]
    row_square += jnp.sum(jnp.square(row))
    col_square += jnp.sum(jnp.square(col))
    row_count += row.size
    col_count += col.size
    layer_rms.append(jnp.sqrt(jnp.mean(jnp.square(value))))
  if not layer_rms:
    raise ValueError("No fetched W_R/kernel parameters found")
  return {
      "rms": jnp.sqrt((row_square + col_square) / (row_count + col_count)),
      "row_rms": jnp.sqrt(row_square / row_count),
      "col_rms": jnp.sqrt(col_square / col_count),
      "layer_rms": jnp.stack(layer_rms),
  }


def _batch_fingerprint(batch):
  inputs = np.asarray(jax.device_get(batch["inputs"]), dtype=np.int32)
  return {
      "shape": list(inputs.shape),
      "sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
      "first_tokens": inputs.reshape(-1)[:32].tolist(),
  }


def run(config):
  output_path = Path(os.environ.get(
      "BAM_DYNAMICS_OUTPUT", "/tmp/mnorm_training_dynamics.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  trace_steps = int(os.environ.get("BAM_DYNAMICS_STEPS", "21"))

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  first_batch = next(eval_iterator)

  def step_fn(state, batch, rng):
    grad_fn = jax.value_and_grad(train.loss_fn, argnums=4, has_aux=True)
    (loss, _), raw_grads = grad_fn(
        model, config, batch, rng, state.params, is_train=True)
    raw_norm = _global_l2(raw_grads)
    clipped_grads = maxtext_utils.apply_gradient_clipping(
        raw_grads, state, config.gradient_clipping_threshold)
    clipped_norm = _global_l2(clipped_grads)
    new_state = state.apply_gradients(grads=clipped_grads)
    return new_state, {
        "loss": loss,
        "raw_grad_norm": raw_norm,
        "clip_multiplier": clipped_norm / jnp.maximum(raw_norm, 1e-30),
        "w_r": _w_r_stats(raw_grads, config.bam_k),
    }

  compiled_step = jax.jit(step_fn)
  report = {
      "metadata": {
          "exp_class": config.exp_class,
          "trace_steps": trace_steps,
          "batch": _batch_fingerprint(first_batch),
          "jax_version": jax.__version__,
          "git_commit": os.environ.get("BAM_DYNAMICS_GIT_COMMIT", ""),
      },
      "steps": {},
  }
  batch = first_batch
  for step in range(trace_steps):
    if step:
      batch = next(eval_iterator)
    rng = jax.random.fold_in(init_rng, step)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics = compiled_step(state, batch, rng)
    host = jax.device_get(metrics)
    report["steps"][str(step)] = {
        "loss": float(host["loss"]),
        "raw_grad_norm": float(host["raw_grad_norm"]),
        "clip_multiplier": float(host["clip_multiplier"]),
        "w_r_rms": float(host["w_r"]["rms"]),
        "w_r_row_rms": float(host["w_r"]["row_rms"]),
        "w_r_col_rms": float(host["w_r"]["col_rms"]),
        "w_r_layer_rms": np.asarray(host["w_r"]["layer_rms"]).tolist(),
    }
    print(
        f"BAM_DYNAMICS step={step} loss={float(host['loss']):.8f} "
        f"raw={float(host['raw_grad_norm']):.6f} "
        f"clip={float(host['clip_multiplier']):.6f} "
        f"wr={float(host['w_r']['rms']):.8f}",
        flush=True)

  report["metadata"]["elapsed_seconds"] = time.perf_counter() - started
  output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"BAM_DYNAMICS_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

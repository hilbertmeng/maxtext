"""Same-batch, one-layer block-RoPE loss ablation for V2 C256.

The diagnostic intercepts only the selected BamAttention layer.  It leaves the
production attention implementation unchanged and uses a dynamic target-layer
scalar so all 24 interventions share one compiled executable per mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
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
import train
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import attentions
from v2_c256_rope_gate_diagnostics import (  # pylint: disable=unused-import
    BamLlama2MediumV2C256RopeGateDiagnostics,
)
from v2_c256_rope_structure_diagnostics import _block_rope


_LAYER_RE = re.compile(r"layers_(\d+)")
_MODES = (
    ("std_block_only_lambda_0.1", "std", 0.1),
    ("bam_block_only_lambda_0.1", "bam", 0.1),
    ("full_block_lambda_0.1", "full", 0.1),
    ("full_block_lambda_1.0", "full", 1.0),
)


def _layer_index(module) -> int | None:
  for component in module.scope.path:
    match = _LAYER_RE.fullmatch(component)
    if match:
      return int(match.group(1))
  return None


def _intervened_loss(model, config, params, batch, rng, target_layer,
                     mode, strength):
  positions_by_layer = {}
  split = config.bam_k + config.bam_abs_v_compression_dim

  def interpolate(current, target, layer):
    weight = jnp.asarray(strength, current.dtype) * jnp.asarray(
        target_layer == layer, current.dtype)
    return current + weight * (target - current)

  def interceptor(next_fun, args, kwargs, context):
    module = context.module
    if not isinstance(module, attentions.BamAttention):
      return next_fun(*args, **kwargs)
    layer = _layer_index(module)
    if layer is None:
      return next_fun(*args, **kwargs)

    if context.method_name == "apply_rotary_embedding":
      inputs, positions = args[:2]
      positions_by_layer[layer] = positions
      current = next_fun(*args, **kwargs)
      if mode not in ("std", "full"):
        return current
      target = _block_rope(
          inputs, positions, split,
          config.rope_min_timescale, config.rope_max_timescale)
      return interpolate(current, target, layer)

    if context.method_name == "_read_local_qk":
      q_bam, k_bam = next_fun(*args, **kwargs)
      if mode not in ("bam", "full"):
        return q_bam, k_bam
      positions = positions_by_layer[layer]
      q_target = _block_rope(
          q_bam, positions, split,
          config.rope_min_timescale, config.rope_max_timescale)
      k_target = _block_rope(
          k_bam, positions, split,
          config.rope_min_timescale, config.rope_max_timescale)
      return (
          interpolate(q_bam, q_target, layer),
          interpolate(k_bam, k_target, layer),
      )

    return next_fun(*args, **kwargs)

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


def run(config):
  if config.scan_layers:
    raise ValueError("dynamic layer intervention currently requires unscanned layers")
  if config.bam_local_qk_injection != "post_rope":
    raise ValueError("diagnostic expects production post-RoPE LocalQK")
  if config.bam_abs_v_compression_dim is None:
    raise ValueError("block split requires bam_abs_v_compression_dim")

  microbatch = int(os.environ.get("BAM_ROPE_LOSS_MICROBATCH", "2"))
  layer_spec = os.environ.get("BAM_ROPE_LOSS_LAYERS", "")
  layers = (
      [int(value) for value in layer_spec.split(",") if value]
      if layer_spec else list(range(config.num_decoder_layers)))
  output_path = Path(os.environ.get(
      "BAM_ROPE_LOSS_OUTPUT", "/tmp/v2_c256_block_rope_loss.json"))
  output_path.parent.mkdir(parents=True, exist_ok=True)

  started = time.perf_counter()
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = (
      train.setup_mesh_and_model(config))
  data_iterator, eval_iterator = create_data_iterator(config, mesh)
  if eval_iterator is None:
    raise ValueError("Pile eval iterator is unavailable")
  state, _, _, _ = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager)
  batch = next(eval_iterator)
  host_inputs = np.asarray(jax.device_get(batch["inputs"]))
  sequence_hashes = [
      hashlib.sha256(sequence.tobytes()).hexdigest()[:16]
      for sequence in host_inputs
  ]

  totals = {
      name: {layer: [0.0, 0.0] for layer in layers}
      for name, _, _ in _MODES
  }
  baseline = [0.0, 0.0]
  timings = {}
  for name, mode, strength in _MODES:
    compiled = jax.jit(
        lambda params, small_batch, rng, target: _intervened_loss(
            model, config, params, small_batch, rng, target, mode, strength))
    mode_started = time.perf_counter()
    for microbatch_index, small_batch in enumerate(
        _iter_microbatches(batch, microbatch)):
      rng = jax.random.fold_in(init_rng, microbatch_index)
      if name == _MODES[0][0]:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          loss_sum, weight_sum = compiled(
              state.params, small_batch, rng, jnp.asarray(-1, jnp.int32))
        loss_sum, weight_sum = jax.device_get((loss_sum, weight_sum))
        baseline[0] += float(loss_sum)
        baseline[1] += float(weight_sum)
      for layer in layers:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          loss_sum, weight_sum = compiled(
              state.params, small_batch, rng, jnp.asarray(layer, jnp.int32))
        loss_sum, weight_sum = jax.device_get((loss_sum, weight_sum))
        totals[name][layer][0] += float(loss_sum)
        totals[name][layer][1] += float(weight_sum)
      print(
          f"BAM_BLOCK_ROPE_LOSS mode={name} microbatch={microbatch_index}",
          flush=True)
    timings[name] = time.perf_counter() - mode_started

  baseline_loss = baseline[0] / baseline[1]
  results = {}
  for name, _, _ in _MODES:
    layer_loss = {
        str(layer): totals[name][layer][0] / totals[name][layer][1]
        for layer in layers
    }
    results[name] = {
        "loss": layer_loss,
        "dloss": {
            layer: value - baseline_loss for layer, value in layer_loss.items()
        },
    }

  report = {
      "metadata": {
          "checkpoint": config.load_parameters_path,
          "code_commit": os.environ.get("BAM_ROPE_LOSS_CODE_COMMIT", ""),
          "sequence_count": len(sequence_hashes),
          "unique_sequence_count": len(set(sequence_hashes)),
          "cohort_hash": hashlib.sha256(
              "".join(sequence_hashes).encode()).hexdigest()[:16],
          "sequence_hashes": sequence_hashes,
          "microbatch": microbatch,
          "layers": layers,
          "block_rope_split": config.bam_k + config.bam_abs_v_compression_dim,
          "timings_seconds": timings,
          "elapsed_seconds": time.perf_counter() - started,
      },
      "baseline_loss": baseline_loss,
      "results": results,
  }
  output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"BAM_BLOCK_ROPE_LOSS_DONE report={output_path}", flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  run(config)


if __name__ == "__main__":
  app.run(main)

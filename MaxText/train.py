"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import sys
import functools
import pickle
import time
from collections import defaultdict
import re
import queue

from typing import Sequence
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import
import tensorflow as tf

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models

from gcp_workload_monitor import GCPWorkloadMonitor

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
from input_pipeline._pile_data_processing import record_file_and_step # lsp
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import orbax_utils, train_state
import optimizers

from ml_goodput_measurement import monitoring

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3


def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""

  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization == "fp8":
    # pylint: disable=line-too-long
    assert (
        config.gradient_accumulation_steps == 1
    ), "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set gradient_accumulation_steps to 1"


def get_first_step(state):
  with jax.spmd_mode("allow_all"):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr, per_device_tokens):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": 1 / step_time_delta.total_seconds()}) # s/step -> step/s
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tokens": per_device_tokens})
  metrics["scalar"].update({"perf/per_device_tokens_per_sec": per_device_tokens / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


_buffered_step = None
_buffered_metrics = None


def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config, is_training=True):
  """Entry point for all metrics writing in Train's Main.
  TODO: would be better as a Class in the future (that initialized all state!)

  To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
  onto the last metrics and step and only publish when we receive a new metrics and step.
  The logic is that this ensures that Jax is able to queues train_steps and we
  don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  metrics_to_write, steps_to_write = None, None
  if is_training:
    global _buffered_step, _buffered_metrics
    if _buffered_metrics is not None:
      if _buffered_step is None:
        raise ValueError(f"When writing metrics, {_buffered_step=} was none")
      metrics_to_write = _buffered_metrics
      steps_to_write = _buffered_step
  else:
    metrics_to_write = metrics
    steps_to_write = step

  if metrics_to_write:
    if config.enable_tensorboard:
      write_metrics_to_tensorboard(writer, metrics_to_write, steps_to_write, config, is_training)

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics_to_write, steps_to_write, config, local_metrics_file, is_training)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(
          metrics_to_write, steps_to_write, config, running_gcs_metrics, is_training
      )

  if is_training:
    _buffered_step = step
    _buffered_metrics = metrics


def write_metrics_to_tensorboard(writer, metrics, step, config, is_training=True):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode("allow_all"):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        # fp32 write to tensorboard
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name], dtype=np.float32), step)
      for metric_name in metrics.get("scalars", []):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0
    # max_logging.log(
    #     f"completed step: {step}, steps/s: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
    #     f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
    #     f"loss: {metrics['scalar']['learning/loss']:.3f}, "
    #     f"aux_loss: {metrics['scalar']['learning/aux_loss']:.3f}, "
    #     f"accuracy: {metrics['scalar']['learning/accuracy']:.4f}"
    # )
    if full_log and jax.process_index() == 0:
      max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
      writer.flush()


def save_checkpoint(checkpoint_manager, step, state, dataset_type="c4", data_iterator=None):
  """Wrapper for saving checkpoint"""
  if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
    return checkpoint_manager.save(
      step, args=orbax.checkpoint.args.PyTreeSave(state)
  )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
    )
  elif dataset_type in ["pile", 'novel_4_32k', 'pretrain_4k', 'instruct']: # lsp
    return checkpoint_manager.save(step, {'state': state})
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
    )


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------
def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""
  if 'intermediates' not in intermediate_outputs: return
  # lsp
  if 'eos_sum' in intermediate_outputs["intermediates"]["decoder"]:
    output_metrics["scalar"]["eos_sum"] = intermediate_outputs["intermediates"]["decoder"]["eos_sum"]
  if 'eos_sum_mean' in intermediate_outputs["intermediates"]["decoder"]:
    output_metrics["scalar"]["eos_sum_mean"] = intermediate_outputs["intermediates"]["decoder"]["eos_sum_mean"]

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["layers"] # decode -> layers
    for layer_num in range(0, config.base_num_decoder_layers, 1):
      if config.n_shared_experts > 0:
        output_metrics["scalar"][f"shared_mlp_l2norm/layer_{layer_num:03d}"] = metrics_dict["shared_mlp_l2norm"][0][layer_num]
      # moe metrics
      sub_layer_num = layer_num % config.num_layers_per_block # 0,1,2,3
      if config.num_experts >= 1 and sub_layer_num % config.insert_moe_divisor == 0:
        main_layer_num = layer_num // config.num_layers_per_block # [0, ..., 11]
        sub_layer_num_values = metrics_dict[f"unshared_mlp_{sub_layer_num}"]
        temp_dict = {
          f"unshared_mlp_output/l2norm/layer_{layer_num:03d}": metrics_dict["unshared_mlp/l2norm"][0][layer_num],
          f"router_logits/l2norm/unshared_mlp_layer_{layer_num:03d}": sub_layer_num_values["router_logits/l2norm"][0][main_layer_num],
          f"router_logits/token_to_expert_score_up/unshared_mlp_layer_{layer_num:03d}":
                           sub_layer_num_values["router_logits/token_to_expert_score"][0][main_layer_num],
          f"router_logits/expert_to_token_score_down/unshared_mlp_layer_{layer_num:03d}":
                           sub_layer_num_values["router_logits/expert_to_token_score"][0][main_layer_num],
          f"sfm_after_topn/token_to_expert_score_up/unshared_mlp_layer_{layer_num:03d}":
                           sub_layer_num_values["sfm_after_topn/token_to_expert_score"][0][main_layer_num],
          f"sfm_after_topn/expert_to_token_score_down/unshared_mlp_layer_{layer_num:03d}":
                           sub_layer_num_values["sfm_after_topn/expert_to_token_score"][0][main_layer_num],
          # f"router_logits/noise_before_{layer_num:03d}/max": sub_layer_num_values["router_logits/noise_before/max"][0][main_layer_num],
          # f"router_logits/noise_before_{layer_num:03d}/min": sub_layer_num_values["router_logits/noise_before/min"][0][main_layer_num],
          # f"router_logits/noise_after_{layer_num:03d}/max": sub_layer_num_values["router_logits/noise_after/max"][0][main_layer_num],
          # f"router_logits/noise_after_{layer_num:03d}/min": sub_layer_num_values["router_logits/noise_after/min"][0][main_layer_num],
        }
        output_metrics["scalar"].update(temp_dict)

        temp_dict = {f"expert_top/selected_expert_{i}_token_nums/unshared_mlp_layer_{layer_num:03d}": 
                sub_layer_num_values[f"top/selected_expert_token_nums"][0][main_layer_num][i] 
                for i in range(0, config.num_experts, 1)}
        output_metrics["scalar"].update(temp_dict)
      
  else:
    if config.dense_conn:
      for layer_num in range(config.num_decoder_layers):
        output_metrics["scalar"][f"mudd/dyn_dense_w/max/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"dyn_dense_w/max/layer_{layer_num}"]
        output_metrics["scalar"][f"mudd/dyn_dense_w/mean/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"dyn_dense_w/mean/layer_{layer_num}"]
        output_metrics["scalar"][f"mudd/dyn_dense_w/min/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"dyn_dense_w/min/layer_{layer_num}"]
        output_metrics["scalar"][f"mudd/dyn_dense_w/std/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"dyn_dense_w/std/layer_{layer_num}"]
        output_metrics["scalar"][f"mudd/dyn_dense_w/norm/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"dyn_dense_w/norm/layer_{layer_num}"]
        output_metrics["scalar"][f"mudd/layer_output/norm/layer_{layer_num:03d}"] = intermediate_outputs["intermediates"]["decoder"][f"layer_output/norm/layer_{layer_num}"]


def compute_accuracy(logits, targets, masks):
  batch_weights = jnp.maximum(jnp.sum(masks, axis=-1), 1e-10)
  correct = jnp.where(
        masks > 0.0,
        jnp.argmax(logits, axis=-1) == targets,
        jnp.array(False)
    )
  correct = jnp.sum(correct, axis=-1)
  accuracy = jnp.mean(correct / batch_weights)
  return correct, accuracy


def compute_accuracy(logits, targets, masks):
  batch_weights = jnp.maximum(jnp.sum(masks, axis=-1), 1e-10)
  correct = jnp.where(
        masks > 0.0,
        jnp.argmax(logits, axis=-1) == targets,
        jnp.array(False)
    )
  correct = jnp.sum(correct, axis=-1)
  accuracy = jnp.mean(correct / batch_weights)
  return correct, accuracy


def _split_dpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_dpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))


def dpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]

  # for DPO we don't support packed sequence (they shouldn't be present in the first place)
  data["chosen_segmentation"] = (data["chosen_segmentation"] == 1).astype(jnp.int32)
  data["rejected_segmentation"] = (data["rejected_segmentation"] == 1).astype(jnp.int32)
  data["chosen_position"] = data["chosen_position"] * (data["chosen_segmentation"] == 1)
  data["rejected_position"] = data["rejected_position"] * (data["rejected_segmentation"] == 1)

  # concatenated model and reference model forward pass
  inputs = jnp.concatenate([data["chosen"], data["rejected"]], 0)
  inputs_position = jnp.concatenate([data["chosen_position"], data["rejected_position"]], 0)
  inputs_segmentation = jnp.concatenate([data["chosen_segmentation"], data["rejected_segmentation"]], 0)

  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  ref_logits = model.apply(
      {"params": reference_params},
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
  )
  ref_logits = jax.lax.stop_gradient(ref_logits)

  # extract token ids, segmentation and logits for chosen and rejected sequences
  chosen_ids = data["chosen"][..., 1:]
  rejected_ids = data["rejected"][..., 1:]
  chosen_segmentation = data["chosen_segmentation"][..., 1:]
  rejected_segmentation = data["rejected_segmentation"][..., 1:]
  n_logits = logits.shape[-3] // 2  # [B, S, E] - [batch, sequence, embedding/vocab]
  chosen_logits, rejected_logits = logits[:n_logits, :, :], logits[n_logits:, :, :]  # [B, S, E], [B, S, E]
  chosen_ref_logits, rejected_ref_logits = ref_logits[:n_logits, :, :], ref_logits[n_logits:, :, :]  # [B, S, E], [B, S, E]

  # common subsequence and padding mask
  common_prefix_mask = jnp.cumsum(chosen_ids != rejected_ids, axis=-1) == 0  # [B, S]
  valid_seq_mask = (chosen_segmentation != 0) & (rejected_segmentation != 0) & ~common_prefix_mask  # [B, S]

  # compute logratios from the sequence-reduced observed token log-probability
  chosen_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_logps = jnp.sum(chosen_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_ref_logps = jnp.sum(chosen_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

  rejected_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_logps = jnp.sum(rejected_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_ref_logps = jnp.sum(rejected_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

  # DPO loss from chosen and rejected logratios
  LABEL_SMOOTHING, BETA = config.dpo_label_smoothing, config.dpo_beta
  logratios_delta = BETA * (chosen_logratios - rejected_logratios)  # [B]
  losses = (  # [B]
      -jax.nn.log_sigmoid(BETA * logratios_delta) * (1 - LABEL_SMOOTHING)
      - jax.nn.log_sigmoid(-BETA * logratios_delta) * LABEL_SMOOTHING
  )
  total_loss, total_weights = jnp.mean(losses), losses.shape[0]
  loss = total_loss

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "reward_accuracy": reward_accuracy,
  }
  return loss, aux


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  correct, accuracy = compute_accuracy(logits, data["targets"], data["targets_segmentation"])
  flat_intermediate = flatten_dict(intermediate_outputs)

  # ('intermediates', 'decoder', 'layers', 'mlp_0/1/2/3', 'aux_loss')
  if config.num_experts > 1 and config.moe_type != 'mistral':
      _aux_losses = [(v.value, v.weight) for k, v in flat_intermediate.items() if 'aux_loss' in k]
      _aux_losses = jnp.array(_aux_losses)
      aux_losses, aux_weights = _aux_losses[:, 0], _aux_losses[:, 1]
      aux_loss = aux_losses.sum() / aux_weights.sum()
  else:
    aux_loss = 0

  for k, v in flat_intermediate.items():
    max_logging.log(k)
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)
  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "aux_loss": moe_lb_loss,
      "accuracy": accuracy, 
      "correct": jnp.sum(correct)
  }
  return loss, aux


params_fir_dirs = ['norm', 'scale', 'attention', 'mlp']
def compute_params_norm(params):
  def param_norm(param):
      return jnp.sqrt(jnp.sum(jnp.square(param)))
  # 记录每个参数的norm
  param_norms = jax.tree_util.tree_map(param_norm, params)
  flat_param_norms = flatten_dict(param_norms)
  scalar_vales = {}
  for k, v in flat_param_norms.items():
    k = '/'.join(k)
    newk = k.replace('params', 'total_params')
    scalar_vales[newk] = v
  return scalar_vales


def train_step(model, config, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = [], [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  if config.gradient_accumulation_steps > 1:

    def accumulate_gradient(acc_grad_and_loss, data):
      grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
      (_, aux), cur_batch_gradient = grad_func(
          model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True
      )
      acc_grad_and_loss["loss"] += aux["total_loss"]
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x * aux["total_weights"] + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      acc_grad_and_loss["total_loss"] += aux["total_loss"]
      acc_grad_and_loss["aux_loss"] += aux["aux_loss"]
      acc_grad_and_loss["accuracy"] += aux["accuracy"]
      return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
      """Reshape global batch to microbatches, assuming batch axis is leading."""
      microbatches = config.gradient_accumulation_steps
      microbatch_shape = (microbatches, batch_arr.shape[0] // microbatches) + batch_arr.shape[1:]
      return jnp.reshape(batch_arr, microbatch_shape)
    # reshape之后多一个维度
    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_grad_and_loss = {"loss": 0.0, "accuracy": 0.0, "grad": init_grad, "total_weights": 0, "aux_loss": 0.0}
    # 默认按行scan， init_grad_and_loss将结果保存下来
    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], grad_and_loss["grad"])
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)
    aux['aux_loss'] = grad_and_loss["aux_loss"] / config.gradient_accumulation_steps
    aux['accuracy'] = grad_and_loss["accuracy"] / config.gradient_accumulation_steps
  else:
    train_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=True)
    grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
    (loss, aux), raw_grads = grad_fn(state.params)

  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="device"), state_mesh_shardings.opt_state),
        )
    )
  new_state = state.apply_gradients(grads=grads)
  
  scalar_values = {
          "learning/loss": aux['total_loss'] / aux['total_weights'],
          "learning/aux_loss": aux['aux_loss'],  # lsp
          "learning/accuracy": aux['accuracy'],
          "learning/grad_norm": max_utils.l2norm_pytree(grads),
          "learning/raw_grad_norm": max_utils.l2norm_pytree(raw_grads),
          "learning/param_norm": max_utils.l2norm_pytree(new_state.params),
          "learning/train_batch_weights": aux['total_weights'],
      }

  params_scalar_values = compute_params_norm(new_state.params)
  scalar_values.update(params_scalar_values)

  metrics = {
      "scalar": scalar_values,
      "scalars": {},
      # "aux": intermediate_outputs
  }
  # intermediate_outputs = jax.tree_map(lambda x: jax.device_put(x, device=jax.devices("cpu")[0]), intermediate_outputs)
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""
  eval_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params)

  metrics = {
      "scalar": 
      {
      "evaluation/loss": aux["total_loss"] / aux["total_weights"],  # lsp: batch token mean loss
      "evaluation/total_loss": aux["total_loss"], 
      "evaluation/total_weights": aux["total_weights"],
      "evaluation/aux_loss": aux["aux_loss"],
      "evaluation/accuracy": aux["accuracy"],
      "evaluation/correct": aux["correct"],
      }
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]
  return metrics


def create_goodput_recorder(config):
  if config.enable_goodput_recording:
    logger_name = f"goodput_{config.run_name}"
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, jax.process_index() == 0)
    return recorder
  return None


def record_goodput(
    recorder,
    config,
    record_func,
    *args,
):
  """Record data for Goodput and Badput computation."""
  if recorder and config.enable_goodput_recording:
    record_func(*args)


def check_example_batch(config, example_batch):
  if config.max_checkify:
    jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
    # Check if inputs in batch contains bad synthetic data.
    # pylint: disable=not-callable
    err, _ = jax.jit(jittable_f)(example_batch["inputs"][: config.global_batch_size_to_train_on, :])
    err.throw()


def model_init(model, config, key):
  input_shape = (config.global_batch_size_to_load, config.max_target_length - 1)
  params = model.init(
      {"params": key, "dropout": key, "aqt": key},
      jnp.ones(input_shape, dtype=jnp.int32),
      jnp.ones(input_shape, dtype=jnp.int32),
  )
  return params
  

def get_wd_tree(config, params):
  '''example:
  config.wd_mults = [(".*/scale$", 0.001), (".*/attention$", 0.1)]
  '''
  if not config.wd_mults: return None
  assert isinstance(config.wd_mults, list)
  wd_tree = {}
  for name in flatten_dict(params):
      name_str = '/'.join(name)
      wd = config.adam_weight_decay
      for pat, user_wd in config.wd_mults:
          if re.findall(pat, name_str):
            wd = user_wd
      wd_tree[name] = wd
  wd_tree = unflatten_dict(wd_tree)
  return wd_tree


def setup_mesh_and_model(config):
  """Set up the mesh and the model for training

  Args:
    config

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  # lsp: add rule param weight decay
  if config.wd_mults:
    params_shape = jax.eval_shape(functools.partial(model_init, model, config), init_rng)
    max_logging.log(f'wd_mults is not None, -> {config.wd_mults}')
    wd_tree = get_wd_tree(config=config, params=params_shape)
  else:
    wd_tree = None
  max_logging.log(f'wd_tree: {wd_tree}')
  tx = optimizers.get_optimizer(config, learning_rate_schedule, wd_tree)

  if config.enable_emergency_checkpoint:
    abstract_state, _, _ = max_utils.get_abstract_state(
      model, tx, config, init_rng, mesh, is_training=True
    )
    checkpoint_manager = (
      checkpointing.create_orbax_emergency_checkpoint_manager(
          config.local_checkpoint_directory,
          config.checkpoint_dir,
          mesh,
          abstract_state,
          config.local_checkpoint_period,
          config.checkpoint_period,
      )
    )
  else:
    # logger = checkpointing.setup_checkpoint_logger(config)
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config)

  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config

  Returns:
    init_rng:
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_tpu_init_start_time if recorder else None)
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  record_goodput(recorder, config, recorder.record_tpu_init_end_time if recorder else None)
  record_goodput(recorder, config, recorder.record_training_preparation_start_time if recorder else None)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)

  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

  if config.use_dpo:
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    max_logging.log(f"Restoring reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'")
    try:
      step0_restored, _ = checkpointing.load_state_if_possible(
          checkpoint_manager,
          data_iterator,
          load_parameters_from_path="",
          load_full_state_from_path="",
          abstract_unboxed_pre_state=abstract_state,
          enable_single_replica_ckpt_restoring=False,
          dataset_type=config.dataset_type,
          step=0,
      )
    except FileNotFoundError:
      step0_restored = None
    if step0_restored is not None:
      reference_params = step0_restored["items"].params["params"]
      state = _merge_dpo_state(state, reference_params)
    else:
      max_logging.log(
          f"Could not restore reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'"
      )

  record_goodput(recorder, config, recorder.record_training_preparation_end_time if recorder else None)
  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def train_loop(config, state=None):
  """Main Training loop.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  # Create a GoodputRecorder to log information
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_job_start_time if recorder else None)

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_annotations, model, config)
  # lsp
  if isinstance(state, dict):
    state = train_state.TrainState(
      step=state['step'],
      params=state['params'],
      opt_state=state.get('opt_state'),
      apply_fn=model.apply,
      tx=None,
    )
  for k, v in flatten_dict(state.params).items():
    print(k, v.shape)
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    # TODO: p_eval_step is not yet supported in load_compiled
    p_eval_step = None
    max_logging.log("Loaded compiled function!", flush=True)
  else:
    if config.only_eval:
      p_train_step = None
    else:
      # max_logging.log(f'in_shard_train: {in_shard_train}')
      p_train_step = jax.jit(
          functional_train,
          in_shardings=in_shard_train,
          out_shardings=out_shard_train,
          static_argnums=static_argnums_train,
          donate_argnums=donate_argnums_train,
      )

  if eval_data_iterator:
    max_logging.log(f'eval_data_iterator is not None')
    p_eval_step = jax.jit(
        functional_eval,
        in_shardings=in_shard_eval,
        out_shardings=out_shard_eval,
        static_argnums=static_argnums_eval,
        donate_argnums=donate_argnums_eval,
    )
  else:
    p_eval_step = None

  local_metrics_file = open(config.metrics_file, "a", encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  first_profiling_step = prof.start_initial_profile_step
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = prof.finished_initial_profile_step

  example_batch = None
  last_step_completion = datetime.datetime.now()
  prof = profiler.Profiler(config)

  def should_eval(step, eval_start_step):
    eval_loss = 10000.0
    start_time = time.time()
    if config.eval_interval > 0 and step > start_step and step % config.eval_interval == 0 or config.only_eval or eval_start_step:
      if eval_data_iterator is None: return eval_loss, False
      eval_data_iterator.reset()
      assert eval_data_iterator
      cumulative_eval_metrics = defaultdict(int)
      count = 0
      for edx in range(config.eval_loop_num_batches):
        try:
          eval_batch = next(eval_data_iterator)
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, nextrng)
            _eval_loss = float(eval_metrics['scalar']['evaluation/total_loss'])
            _weight = float(eval_metrics['scalar']['evaluation/total_weights'])
            _correct = float(eval_metrics['scalar']['evaluation/correct'])
            _accuracy = float(eval_metrics['scalar']['evaluation/accuracy'])
            _aux_loss = float(eval_metrics['scalar']['evaluation/aux_loss'])

          cumulative_eval_metrics['total_loss'] += _eval_loss
          cumulative_eval_metrics['total_weights'] += _weight
          cumulative_eval_metrics['total_correct'] += _correct

          cumulative_eval_metrics['aux_loss'] += _aux_loss
          cumulative_eval_metrics['accuracy'] += _accuracy # batch acc

          mean_eval_loss = _eval_loss / _weight
          cumulative_eval_metrics['total_batch_loss'] += mean_eval_loss
          count += 1
          max_logging.log(f'eval_step: {count} loss: {mean_eval_loss:.4f} aux_loss: {_aux_loss:.4f} accuracy: {_accuracy:.4f} \
          correct: {_correct} weight: {_weight} take: {time.time() - start_time:.3f}s')
        except Exception as e:
          max_logging.log(f'error: {e} now start to reset eval dataloader')
      aux_loss = cumulative_eval_metrics['aux_loss'] / count
      # token mean loss
      eval_loss = cumulative_eval_metrics["total_loss"] / (cumulative_eval_metrics["total_weights"] + EPS)
      accuracy = cumulative_eval_metrics['total_correct'] / cumulative_eval_metrics['total_weights']
      # batch token mean loss
      batch_eval_loss = cumulative_eval_metrics["total_batch_loss"] / count
      # batch acc
      batch_accuracy = cumulative_eval_metrics["accuracy"] / count
      max_logging.log(f"average loss after {step=}, eval_loss={eval_loss:.4f}, aux_loss={aux_loss:.4f}, accuracy={accuracy:.4f},\
      batch_eval_loss={batch_eval_loss:.4f}, batch_accuracy={batch_accuracy:.4f} total_weights={cumulative_eval_metrics['total_weights']}")
      
      if jax.process_index() == 0:
        writer.add_scalar('learning/eval_loss', eval_loss, step)
        writer.add_scalar('learning/eval_accuracy', accuracy, step)
        writer.add_scalar('learning/batch_eval_loss', batch_eval_loss, step)
        writer.add_scalar('learning/batch_eval_accuracy', batch_accuracy, step)
        max_logging.log(f"Write step {step} eval loss: {eval_loss:4f} accuracy: {accuracy:.4f} to tensorboard ")
        writer.flush()

      if config.only_eval:
        max_logging.log(f"Current mode is only eval, so don't run train, now start exit......")
        exit(0)
    return eval_loss, False

  eval_start_step = config.eval_start_step
  count = 0
  for step in np.arange(start_step, config.steps):
    count += 1
    if count % 5000 == 0: # 每隔5000步新建一个tensorboard文件，不然如果tensorboard文件过大，写入会很慢
      max_utils.close_summary_writer(writer)
      writer = max_utils.initialize_summary_writer(config)

    if step == first_profiling_step:
      prof.activate()
    nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
    eval_loss, eval_start_step = should_eval(step, eval_start_step=eval_start_step)
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
      example_batch = load_next_batch(data_iterator, example_batch, config)
      record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
      check_example_batch(config, example_batch=example_batch)
      record_goodput(recorder, config, step=step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)
    step_time_delta = datetime.datetime.now() - last_step_completion
    last_step_completion = datetime.datetime.now()
    record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
    if performance_metric_queue:
      performance_metric_queue.put(step_time_delta.total_seconds())

    if checkpoint_manager is not None:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, data_iterator, config):
        checkpointing.print_save_message(step, config.async_checkpointing)
        record_file_and_step(step, config, data_iterator) # lsp
        max_logging.log(f"saved a checkpoint at step {step}")

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()
    # 每步打印
    max_logging.log(
        f"completed step: {step}, steps/s: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
        f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
        f"loss: {metrics['scalar']['learning/loss']:.3f}, "
        f"aux_loss: {metrics['scalar']['learning/aux_loss']:.3f}, "
        f"accuracy: {metrics['scalar']['learning/accuracy']:.4f}"
    )
    if step % 5 == 0:
      # 每隔5步进行写入，每隔log_period进行flush
      write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config)

    if config.dump_hlo and step == start_step:
      jax.block_until_ready(state)  # Ensure compilation has finished.
      max_utils.upload_dump(
          config.dump_hlo_local_dir,
          config.dump_hlo_gcs_dir,
          module_name=config.dump_hlo_module_name,
          delete_local_after=config.dump_hlo_delete_local_after,
          all_host_upload=config.dump_hlo_upload_all,
      )

    if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
      assert eval_data_iterator
      cumulative_eval_metrics = {
          "scalar": {
              "eval/total_loss": 0.0,
              "eval/total_weights": 0.0,
              "eval/avg_loss": 0.0,
              "eval/moe_lb_loss": 0.0,
          }
      }
      eval_dpo_reward_accuracy = 0.0
      eval_step_count = 0
      # pylint: disable=not-callable
      for eval_batch in eval_data_iterator:
        if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
          break
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          eval_metrics = p_eval_step(state, eval_batch, nextrng)
        cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
        eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      if config.use_dpo:
        cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
      write_metrics(
          writer, local_metrics_file, running_gcs_metrics, cumulative_eval_metrics, step, config, is_training=False
      )
      max_logging.log(
          f"average loss after {step=}: {eval_step_count=}, {eval_loss=},"
          f" total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
      )
      if eval_loss <= config.target_eval_loss:
        max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
        prof.deactivate()
        break

    if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
      prof.deactivate(blocking_object=state)

    if step == start_step:
      max_utils.print_mem_stats("After params initialized")

  if checkpoint_manager is not None:
    checkpoint_manager.wait_until_finished()

  write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, config.steps - 1, config)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)
  clear_buffered_metrics()
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    # pytype: disable=attribute-error
    compiled = p_train_step.lower(state, example_batch, nextrng).compile()
    compiled_stats = compiled.memory_analysis()
    if compiled_stats is not None:
      max_logging.log(
          f"Output size: {compiled_stats.output_size_in_bytes}, "
          f"temp size: {compiled_stats.temp_size_in_bytes}, "
          f"argument size: {compiled_stats.argument_size_in_bytes}, "
          f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
      )
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=logger_name,
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=config.enable_pathways_goodput,
        include_badput_breakdown=True,
        include_step_deviation=config.monitor_step_time_deviation,
        step_deviation_interval_seconds=config.step_deviation_interval_seconds,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard in the background!")
    if config.monitor_step_time_deviation:
      goodput_monitor.start_step_deviation_uploader()
      max_logging.log("Started step time deviation upload to Tensorboard in the background!")
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for vocabulary tiling (VT)"""

import functools

from flax import linen as nn

import jax
import jax.numpy as jnp
import max_utils
import enum


class ShardMode(enum.Enum):
  AUTO = "auto"  # default
  EXPLICIT = "explicit"


def maybe_shard_with_name(inputs, named_sharding, shard_mode):
  if shard_mode == ShardMode.EXPLICIT:
    return jax.sharding.reshard(inputs, named_sharding)
  else:
    return jax.lax.with_sharding_constraint(inputs, named_sharding)


def get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules):
  def remove_fsdp_sharding(sharding_tree):
    """Recursively traverses the sharding tree to remove fsdp axes."""
    def _remove_fsdp_from_partition_spec(named_sharding):
      """Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec."""
      if isinstance(named_sharding, jax.sharding.NamedSharding):
        new_spec = []
        # Iterate through each axis in the original PartitionSpec.
        for axis in named_sharding.spec:
          if axis is None:
            new_spec.append(None)
          elif isinstance(axis, str):
            # If the axis is 'fsdp', replace it with None to signify replication.
            if axis not in ("fsdp", "fsdp_transpose"):
              new_spec.append(axis)
            else:
              new_spec.append(None)
          elif isinstance(axis, (list, tuple)):
            # If the axis is a collection, filter out 'fsdp'.
            new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
            new_spec.append(tuple(new_axis))
          else:
            raise ValueError(f"Unsupported_axis_type: {type(axis)}")
          # Return a new sharding object with the modified spec.
        return jax.sharding.NamedSharding(named_sharding.mesh, jax.sharding.PartitionSpec(*new_spec))
      return named_sharding

    return jax.tree.map(_remove_fsdp_from_partition_spec, sharding_tree)

  # Convert the high-level logical spec to a physical one using default rules.
  physical = nn.logical_to_mesh_sharding(full_logical, mesh=mesh, rules=logical_axis_rules)
  # Apply the function to remove the FSDP sharding, defining our target layout.
  physical_no_fsdp = remove_fsdp_sharding(physical)
  return physical_no_fsdp


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules):
  # 获取no fsdp 的 sharding tree
  physical_constraint_no_fsdp = get_physical_spec_no_fsdp(sharding_info, mesh, logical_axis_rules)
  # 然后重新shard
  return jax.lax.with_sharding_constraint(variables, physical_constraint_no_fsdp)


def vocab_tiling_loss(hidden_states, labels, segmentation, config, model, lm_head_params):
  """Calculates a tiled MTP loss from hidden states."""
  if config.logits_via_embedding:
    raise NotImplementedError("VT MTP currently only supports logits_via_embedding=False.")

  param_spec = nn.get_partition_spec(lm_head_params)
  hidden_spec = jax.sharding.NamedSharding(
      model.mesh,
      nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed")),
  )
  label_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp"))
  )
  reshaped_hidden_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("num_tile", "activation_embed_and_logits_batch", "activation_embed"))
  )
  reshaped_data_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("num_tile", "activation_embed_and_logits_batch"))
  )
  chunked_hidden_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_embed"))
  )
  chunked_data_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch",))
  )
  chunked_logits_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_vocab"))
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name, shard_mode=config.shard_mode
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)
  gathered_lm_head_params = all_gather_over_fsdp(
      lm_head_params, param_spec, model.mesh, config.logical_axis_rules
  )

  @jax.custom_vjp
  def chunked_cross_entropy_loss(gathered_lm_head_params, hidden_states, labels, segmentation):
    total_loss, _ = _chunked_cross_entropy_loss_fwd(gathered_lm_head_params, hidden_states, labels, segmentation)
    return total_loss

  def _chunked_cross_entropy_loss_fwd(gathered_lm_head_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = _reshape(
        hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

    def _fwd_scan_body(loss_accumulator, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      chunk_logits = model.apply(
          {"params": {"decoder": {"lm_head": gathered_lm_head_params}}},
          hidden_chunk,
          deterministic=True,
          mtp_layer=True,
          method=model.logits_from_hidden_states,
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
      chunk_xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk, 0.0)
      masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
      loss_accumulator += masked_xent
      return loss_accumulator, None

    initial_loss = 0.0
    total_loss, _ = jax.lax.scan(
        _fwd_scan_body, initial_loss, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    residuals = (
        gathered_lm_head_params,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    )
    return total_loss, residuals

  def _chunked_cross_entropy_loss_bwd(residuals, loss_cotangent):
    gathered_lm_head_params, reshaped_hidden_states, reshaped_labels, reshaped_segmentation, batch_size, seq_len, emb_dim = (
        residuals
    )

    def _single_chunk_loss_fn(input_lm_head_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk):
      chunk_logits = model.apply(
          {"params": {"decoder": {"lm_head": input_lm_head_params}}},
          input_hidden_chunk,
          deterministic=True,
          mtp_layer=True,
          method=model.logits_from_hidden_states,
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(input_label_chunk, config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk, 0.0)
      return jnp.sum(xent * (input_segmentation_chunk != 0))

    def _bwd_scan_body(grad_params_acc, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      loss_fn_for_vjp = lambda p, h: _single_chunk_loss_fn(p, h, label_chunk, segmentation_chunk)
      _, vjp_fn = jax.vjp(loss_fn_for_vjp, gathered_lm_head_params, hidden_chunk)
      grad_lm_head_params_update, grad_hidden_chunk = vjp_fn(1.0)
      grad_hidden_chunk = _maybe_shard_with_name(grad_hidden_chunk, chunked_hidden_spec)

      grad_params_acc = jax.tree_util.tree_map(
          lambda acc, update: acc + update,
          grad_params_acc,
          grad_lm_head_params_update,
      )
      return grad_params_acc, grad_hidden_chunk

    initial_grad_params_acc = jax.tree_util.tree_map(jnp.zeros_like, gathered_lm_head_params)
    grad_params, grad_reshaped_hidden_states = jax.lax.scan(
        _bwd_scan_body, initial_grad_params_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    grad_reshaped_hidden_states = _maybe_shard_with_name(grad_reshaped_hidden_states, reshaped_hidden_spec)
    grad_params = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_params)
    grad_reshaped_hidden_states = grad_reshaped_hidden_states * loss_cotangent # gpt-5.4
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)
    return (
        grad_params,
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,
        None,
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)
  return chunked_cross_entropy_loss(gathered_lm_head_params, hidden_states, labels, segmentation)
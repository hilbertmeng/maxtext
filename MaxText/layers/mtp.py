"""
Copyright 2025 Google LLC

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

"""JAX implementation of the Multi Token Predicition https://arxiv.org/pdf/2412.19437 """

from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from common_types import Config, MODEL_MODE_TRAIN
from layers.normalizations import RMSNorm as rms_norm
import max_utils
import maxtext_utils
from layers import initializers
from layers import linears
from layers import mudd


EPS = 1e-8
def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """
  Performs a leftward roll on the sequence axis (axis=1) and masks the
  newly created invalid positions at the end of the sequence.
  Assumes input `x` has a batch dimension at axis 0 and sequence at axis 1.

  Args:
    x: The input array of shape [batch, seq_len, ...].
    shift: The number of positions to shift left.

  Returns:
    The rolled array of the same shape as x.
  """
  # If shift is 0, it's a no-op. Return the original array.
  if shift == 0:
    return x

  # to set the last `abs(shift)` elements of the sequence to zero.
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


class MultiTokenPredictionLayer(nn.Module):
  """
  Implements Multi-Token Prediction (MTP) step:
      1. Normalization of previous hidden state and target token embedding.
      2. Concatenation and Projection of normalized features.
      3. Processing through a Transformer Decoder Layer.

      Equation Representation (Conceptual):
          norm_h = RMSNorm(h_prev)
          norm_e = RMSNorm(e_target)
          h_proj = W_p(concat(norm_h, norm_e))
          h_next = TransformerLayer(h_proj, pos_ids, segment_ids, ...)

      It takes the previous hidden state and target embedding as input and outputs the
      processed hidden state from its internal transformer block.
  """

  config: Config
  mesh: Mesh
  layer_number: int
  transformer_layer_module: None

  @nn.compact
  def __call__(
      self,
      prev_hidden_state: jnp.ndarray, # It is a list if use mudd.
      target_token_embedding: jnp.ndarray,
      position_ids: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      deterministic: bool,
      hids: list = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    """
    Applies the MTP combination, projection, and internal transformer processing.

    Args:
        prev_hidden_state: Hidden state from the previous step/layer.
                           Shape: [batch, seq_len, hidden_size]
        target_token_embedding: Embedding of the target token. In the context of MTP,
                                this often refers to a token at a position relative
                                to the current step, where the offset is determined
                                by the layer number `k` (i.e., token t+k).
                                Shape: [batch, seq_len, embed_dim]
        position_ids: Original position IDs for the sequence.
                      Shape: [batch, seq_len]
        decoder_segment_ids: Original segment IDs for the sequence (for attention mask).
                             Shape: [batch, seq_len]
        deterministic: If true, disable dropout.
        model_mode: The current operational mode (train, eval, decode).

    Returns:
        next_hidden_state: The hidden state produced by this MTP step's internal transformer.
                           Shape: [batch, seq_len, hidden_size]
    """
    cfg = self.config
    mesh = self.mesh
    k = self.layer_number

    projected_features = []
    # --- 1. Normalize Hidden State and Embedding ---
    embedding_norm_layer = rms_norm(
        # num_features=target_token_embedding.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"embedding_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    embedding_norm = embedding_norm_layer(target_token_embedding)

    hidden_state_norm_layer = rms_norm(
        # num_features=_prev_hidden_state.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"hidden_state_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    hidden_state_norm = hidden_state_norm_layer(prev_hidden_state)

    # --- 2. Concatenate Normalized Representations ---
    # Shape: [B, S, 2*H]
    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)

    # --- 3. Project Concatenated Features ---
    # Projects from 2*H back down to H
    projection_layer = linears.DenseGeneral(
        features=cfg.base_emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        use_bias=False,
        kernel_axes=("concat_embed", "embed"),
        # kernel_init=initializers.nd_dense_init_normal(0.006), # lsp?
        name=f"projection",
    )
    # Shape: [B, S, H]
    projected_features = projection_layer(concatenated_features)

    # compose
    if cfg.dense_conn and cfg.mtp_use_compose:
        projected_features, hids = mudd.Compose(
            cfg, self.mesh, None, k + cfg.num_decoder_layers, 
            name=f'compose'
            )(
              layer_output=projected_features, 
              hids=hids
            )

    # --- 4. Pass through MTP Transformer Block ---
    y = self.transformer_layer_module(
        config=cfg, mesh=mesh, 
        name=f"layers_{k - 1 + cfg.num_decoder_layers}")(
          inputs=projected_features, # single array
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=position_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          hids=hids,
    )
    output, hids = y

    if isinstance(output, tuple): # mudd is list type, so ignore it.
      # Handles the scan=True case, where the output is a tuple.
      next_hidden_state = output[0]
    else:
      # Handles the scan=False case, where the output is a single tensor.
      next_hidden_state = output

    # Shape: [B, S, H]
    # --- Return Processed Hidden State ---
    return next_hidden_state, hids


class MultiTokenPredictionBlock(nn.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  config: Config
  mesh: Mesh
  quant: None
  transformer_layer_module: None
  shared_embedding: None

  @nn.compact
  def __call__(
      self,
      output_layer,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      position_ids,
      decoder_segment_ids,
      deterministic=False,
      model_mode=MODEL_MODE_TRAIN,
      hids=None,
  ):
    cfg = self.config
    # The initial hidden state for the MTP chain is the raw output from the main model.
    mtp_hidden_state = main_hidden_state

    # These variables are updated sequentially in each loop iteration,
    # moving the prediction window one token to the right each time.
    rolled_input_ids = input_ids
    rolled_target_ids = target_ids
    rolled_target_mask = target_mask
    # rolled_position_id = position_ids

    # Range chosen to align with the naming convention of the paper
    for k in range(1, cfg.mtp_num_layers + 1):
      # Sequentially roll all tensors to prepare data for predicting the k-th future token.
      rolled_input_ids = roll_and_mask(rolled_input_ids)
      rolled_target_ids = roll_and_mask(rolled_target_ids)
      rolled_target_mask = roll_and_mask(rolled_target_mask)
    #   rolled_position_id = roll_and_mask(rolled_position_id)

      # Embed the k-th future input tokens using the shared embedding module
      target_token_embedding = self.shared_embedding(rolled_input_ids)

      RematMTPLayer = nn.remat(  # pylint: disable=invalid-name
          MultiTokenPredictionLayer,
          prevent_cse=True,
          policy=None,
          static_argnums=(5, ), # 务必注意：参数中有默认值的不能作为静态参数
          rngs={"params": True, "aqt": True, "dropout": True},
      )
       # Instantiate and apply the MTP layer for this step
      mtp_layer = RematMTPLayer(
          config=cfg,
          mesh=self.mesh,
          layer_number=k,
          name=f"mtp_{k}",
          # lsp: Should get prev token's history status in unmtp layers when use mudd, because cur token no history status.
          # but in mtp layers, should get current token's history status. in fact, we can get the position correspond to generated token directly.
          transformer_layer_module=self.transformer_layer_module,
      )
      next_mtp_hidden_state, hids = mtp_layer(
          mtp_hidden_state, target_token_embedding, position_ids, decoder_segment_ids, deterministic, hids
      )
      # Project to logits using the shared embedding transpose
      mtp_logits = output_layer(
        next_mtp_hidden_state[0] if isinstance(next_mtp_hidden_state, tuple|list) else next_mtp_hidden_state, 
        deterministic
        )

      # Calculate cross-entropy loss for this specific layer's prediction
      mtp_xent, _ = max_utils.cross_entropy_with_logits(mtp_logits, jax.nn.one_hot(rolled_target_ids, cfg.vocab_size), 0.0)
      mtp_xent_masked = mtp_xent * rolled_target_mask # BL

      # This logic doesn't run during model initialization to avoid unwated population of the mutable collections.
      if not self.is_initializing(): # don't excute here when model.init
        print(f'MTP loss record.....')
        # For evaluation, save the top prediction and a valid token mask.
        # This is only active for the target layer during an eval run.
        if cfg.mtp_eval_target_module == k:
          print(f'mtp_eval_target_module={k}, compute mtp preds and masks......')
          mtp_top_1_pred = jnp.argmax(mtp_logits, axis=-1) # blv -> bl
          self.sow("intermediates", "mtp_preds", mtp_top_1_pred)
          self.sow("intermediates", "mtp_mask", rolled_target_mask)

        # For training, save the loss components for this MTP head.
        # This is only active during a training run.
        # if self.is_mutable_collection("mtp_losses"):
        self.sow("intermediates", "mtp_losses", jnp.sum(mtp_xent_masked))
        self.sow("intermediates", "mtp_weights", jnp.sum(rolled_target_mask))

      # The output of this layer is the input for the next, maintaining the causal chain.
      mtp_hidden_state = next_mtp_hidden_state


def calculate_mtp_loss(intermediate_outputs, config):
  """Calculates the Multi Token Prediction loss from intermediate outputs."""
  losses_path = ("intermediates", "decoder", "mtp_block", "mtp_losses")
  weights_path = ("intermediates","decoder", "mtp_block", "mtp_weights")
  mtp_losses = maxtext_utils.get_nested_value(intermediate_outputs, losses_path, default=())
  mtp_weights = maxtext_utils.get_nested_value(intermediate_outputs, weights_path, default=())

  if not mtp_losses:  # MTP heads did not run
    return 0.0

  sum_of_all_mtp_losses = jnp.sum(jnp.array(mtp_losses))
  sum_of_all_mtp_weights = jnp.sum(jnp.array(mtp_weights))

  avg_mtp_loss = sum_of_all_mtp_losses / (sum_of_all_mtp_weights + EPS)
  scaled_mtp_loss = avg_mtp_loss * config.mtp_loss_scaling_factor
  return scaled_mtp_loss


def calculate_mtp_acceptance_rate(intermediate_outputs, config, logits):
  """Calculates the MTP acceptance rate from intermediate outputs."""
  preds_path = ("intermediates", "decoder", "mtp_block", "mtp_preds")
  masks_path = ("intermediates","decoder", "mtp_block", "mtp_mask")

  mtp_preds = maxtext_utils.get_nested_value(intermediate_outputs, preds_path, default=())[0]
  valid_mask = maxtext_utils.get_nested_value(intermediate_outputs, masks_path, default=())[0]

  # These values are only "sown" (saved) during an evaluation run and only for the specific
  # MTP layer specified by `config.mtp_eval_target_module`. This check handles cases
  # where the required data is absent (e.g., during a training step) and prevents errors.
  if mtp_preds is None or valid_mask is None:
    print(f'mtp_preds or valid_mask is None....')
    return 0.0
#   main_logits_path =  ("intermediates","decoder", "logits") # lsp
#   main_logits = maxtext_utils.get_nested_value(intermediate_outputs, main_logits_path, default=())[0] # tuple type, such as (main_logits, )
  # Get the main model's greedy predictions from the logits.
  main_model_preds = jnp.argmax(logits, axis=-1)

  # Roll the main model's predictions to align them in time with the MTP head's target.
  rolled_main_preds = main_model_preds
  for _ in range(config.mtp_eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)

  # Compare the aligned predictions. The `valid_mask` ensures that the comparison
  # only happens on valid tokens, ignoring the placeholder values introduced at the
  # end of the sequence by the `roll_and_mask` operation.
  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  # Return acceptance rate as a percentage
  return (correct_predictions / (total_valid_tokens + EPS)) * 100
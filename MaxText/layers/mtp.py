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
import max_logging
import aqt.jax.v2.aqt_dot_general as aqt


dot_general_int8 = aqt.dot_general_make(8, 8)

EPS = 1e-8
def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  # If shift is 0, it's a no-op. Return the original array.
  if shift == 0:
    return x
  # to set the last `abs(shift)` elements of the sequence to zero.
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


class MultiTokenPredictionLayer(nn.Module):

  config: Config
  mesh: Mesh
  quant: None
  layer_number: int
  transformer_layer_module: None
  sliding_window_size: int
  mtp_de: jnp.ndarray

  def setup(self):
    cfg = self.config
    kernel_init_shard = nn.with_logical_partitioning(
      initializers.get_init_method(self.config.init_method), 
      ('concat_embed', 'embed'),
      )
    projection_layer = self.param('projection_layer', kernel_init_shard, (2*cfg.emb_dim, cfg.emb_dim), cfg.weight_dtype)
    self.projection_layer = jnp.asarray(projection_layer, cfg.dtype)

  @nn.compact
  def __call__(
      self,
      prev_hidden_state: jnp.ndarray, # It is a list if use mudd.
      target_token_embedding: jnp.ndarray,
      position_ids: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      deterministic: bool,
      hids: list = None,
      rolled_input_ids: jnp.ndarray = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    cfg = self.config
    mesh = self.mesh
    k = self.layer_number

    projected_features = []
    embedding_norm_layer = rms_norm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"embedding_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    embedding_norm = embedding_norm_layer(target_token_embedding)

    hidden_state_norm_layer = rms_norm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"hidden_state_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    hidden_state_norm = hidden_state_norm_layer(prev_hidden_state)
    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)
    projected_features = jnp.einsum('btH,Hh->bth', concatenated_features, self.projection_layer, 
        _dot_general=dot_general_int8.__call__ 
        if cfg.quantization == 'int8' and cfg.mtp_head_int8 else jax.lax.dot_general
        )
    if self.mtp_de is not None and self.mtp_de.shape[-1] == cfg.mlp_dim:
      print(f'mtp_de: {self.mtp_de.shape}')
      d1 = 32 if cfg.mlp_dim < 4096 else 64
      d2 = cfg.mlp_dim // d1
      mtp_de = self.mtp_de.reshape(*projected_features.shape[:2], d1, d2)
    else:
      mtp_de = None

    if cfg.dense_conn and cfg.partial_scan_layers:
      projected_features = [projected_features] * len(cfg.dynamic_dense_type)
    
    y, _ = self.transformer_layer_module(
        config=cfg, mesh=mesh, quant=self.quant, scan_length=2, # scan_length set >1 means no compose before layer
        sliding_window_size=self.sliding_window_size,
        name=f"layers_{k - 1 + cfg.num_decoder_layers}")(
          projected_features,
          decoder_segment_ids,
          position_ids,
          rolled_input_ids,
          mtp_de,
          deterministic,
          model_mode,
          hids=None, # mtp compose after layer
    )

    if cfg.dense_conn and cfg.partial_scan_layers:
      y, hids = mudd.Compose(
        cfg, self.mesh, self.quant, 
        name=f'compose_final',
        C=1,
        compose=True,
      )(
        layer_output=y if isinstance(y, jnp.ndarray) else y[0], 
        hids=hids,
      )
      
    next_hidden_state = y if isinstance(y, jnp.ndarray) else y[0]
    return next_hidden_state, hids


class MultiTokenPredictionBlock(nn.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  config: Config
  mesh: Mesh
  quant: None
  transformer_layer_module: None
  shared_embedding: None
  sliding_window_size: int

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
      deterministic,
      model_mode,
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
      target_token_embedding, mtp_de = self.shared_embedding
      if cfg.mtp_use_remat:
        RematMTPLayer = nn.remat(  # pylint: disable=invalid-name
            MultiTokenPredictionLayer,
            prevent_cse=cfg.remat_prevent_cse,
            policy=None,
            static_argnums=(5, ), # 务必注意：参数中有默认值的不能作为静态参数
            rngs={"params": True, "aqt": True, "dropout": True},
        )
      else:
        RematMTPLayer = MultiTokenPredictionLayer
       # Instantiate and apply the MTP layer for this step
      mtp_layer = RematMTPLayer(
          config=cfg,
          quant=self.quant,
          mesh=self.mesh,
          layer_number=k,
          name=f"mtp_{k - 1}",
          sliding_window_size=self.sliding_window_size,
          # lsp: Should get prev token's history status in unmtp layers when use mudd, because cur token no history status.
          # but in mtp layers, should get current token's history status. in fact, we can get the position correspond to generated token directly.
          transformer_layer_module=self.transformer_layer_module,
          mtp_de=mtp_de,
      )
      next_mtp_hidden_state, hids = mtp_layer(
          mtp_hidden_state, target_token_embedding, position_ids, decoder_segment_ids, deterministic, hids, rolled_input_ids
      )
      # Project to logits using the shared embedding transpose
      mtp_xent, correct, mtp_top_1_pred = output_layer(
        next_mtp_hidden_state[0] if isinstance(next_mtp_hidden_state, tuple|list) else next_mtp_hidden_state, 
        rolled_target_ids,
        rolled_target_mask,
        cfg.max_target_length,
        deterministic,
        mtp_layer=True
        )
      mtp_xent_masked = mtp_xent * rolled_target_mask # BL

      # This logic doesn't run during model initialization to avoid unwated population of the mutable collections.
      if not self.is_initializing(): # don't excute here when model.init
        max_logging.log(f'MTP loss record.....', debug=cfg.debug)
        # For evaluation, save the top prediction and a valid token mask.
        # This is only active for the target layer during an eval run.
        if cfg.mtp_eval_target_module == k:
          max_logging.log(f'mtp_eval_target_module={k}, compute mtp preds and masks......', debug=cfg.debug)
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


def calculate_mtp_acceptance_rate(intermediate_outputs, config, main_model_preds):
  """Calculates the MTP acceptance rate from intermediate outputs."""
  preds_path = ("intermediates", "decoder", "mtp_block", "mtp_preds")
  masks_path = ("intermediates","decoder", "mtp_block", "mtp_mask")

  mtp_preds = maxtext_utils.get_nested_value(intermediate_outputs, preds_path, default=())[0]
  valid_mask = maxtext_utils.get_nested_value(intermediate_outputs, masks_path, default=())[0]

  if mtp_preds is None or valid_mask is None:
    max_logging.log(f'mtp_preds or valid_mask is None....', debug=config.debug)
    return 0.0
  # main_model_preds = jnp.argmax(logits, axis=-1)
  # Roll the main model's predictions to align them in time with the MTP head's target.
  rolled_main_preds = main_model_preds
  for _ in range(config.mtp_eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)

  # end of the sequence by the `roll_and_mask` operation.
  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  # Return acceptance rate as a percentage
  return (correct_predictions / (total_valid_tokens + EPS)) * 100
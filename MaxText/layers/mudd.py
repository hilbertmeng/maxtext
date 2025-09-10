from typing import Any, Optional

import numpy as np
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh
from einops import rearrange

from layers import initializers
from layers import normalizations
from layers import linears
from layers import quantizations

# Type alias for quantization
Quant = quantizations.AqtQuantization

# Constants
DEFAULT_HIDDEN_ROUND = 64
DEFAULT_POST_NORM_SCALE = 0.001
def l2norm(x: jnp.ndarray) -> jnp.ndarray:
  """Compute L2 norm of input tensor."""
  return jnp.sqrt(jnp.sum(jnp.square(x)))


def weighted_sum(weights: jnp.ndarray,  # Shape: CBTL1
                 hidden_states: list[jnp.ndarray],  # List of BTD tensors
                 seq_chunk_size: int = None  # Currently unused
                 ) -> jnp.ndarray:  # Output shape: CBTD
  """Compute weighted sum of hidden states.
  
  Args:
    weights: Weight tensor of shape (C, B, T, L, 1)
    hidden_states: List of hidden state tensors, each of shape (B, T, D)
    seq_chunk_size: Sequence chunk size (currently unused)
    
  Returns:
    Weighted sum tensor of shape (C, B, T, D)
  """
  C, B, T, L, _ = weights.shape
  D = hidden_states[0].shape[-1]
  output = jnp.zeros((C, B, T, D), dtype=hidden_states[0].dtype)
  
  for layer_idx in range(L):
    output += weights[..., layer_idx, :] * hidden_states[layer_idx]
    
  return output


class Norm(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
      raise ValueError("Input must be a tuple or list of 3 tensors (q, k, v)")
    
    base_name = 'pre_self_attention_layer_norm'
    constraint = ("activation_batch", "activation_norm_length", "activation_embed")
    
    normalized_tensors = []
    for tensor, suffix in zip(inputs, ['q', 'k', 'v']):
      norm_layer = normalizations.get_rmsnorm(f'{base_name}_{suffix}', cfg)
      normalized = nn.with_logical_constraint(norm_layer(tensor), constraint)
      normalized_tensors.append(normalized)
    
    return tuple(normalized_tensors)


class Mlp(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  layer_inx: int = None
  use_bias: bool = True
  C: int = 4

  def _setup_norm_layer(self, cfg) -> normalizations.RMSNorm:
    """Setup the normalization layer."""
    norm_kwargs = {
      "dtype": cfg.dtype,
      "weight_dtype": cfg.weight_dtype,
      "name": "pre_dense_proj1_norm",
      "epsilon": cfg.normalization_layer_epsilon,
    }
    
    if not getattr(cfg, 'mudd_prenorm', False):
      norm_kwargs['scale_init'] = None  # Disable scaling
      
    return normalizations.get_rmsnorm(**norm_kwargs)

  def _compute_dynamic_dimensions(self, cfg):
    """Compute dynamic weight shape and intermediate dimensions."""
    factor = 1
    dw_shape = (self.C, self.layer_inx * factor + 1)
    
    # Determine expansion factor based on layer position
    is_last_layer = (self.layer_inx == 
                     cfg.num_decoder_layers - 1 + cfg.mtp_num_layers)
    dynamic_dense_hidden_expand = (
      len(cfg.dynamic_dense_type) if is_last_layer else 1
    )
    
    # Calculate intermediate dimension
    dynamic_dense_inter_dim = int(
      np.prod(dw_shape) * dynamic_dense_hidden_expand
    )
    
    # Round to nearest multiple if configured
    if cfg.dynamic_dense_hidden_round:
      dynamic_dense_inter_dim = (
        (dynamic_dense_inter_dim // DEFAULT_HIDDEN_ROUND + 1) * 
        DEFAULT_HIDDEN_ROUND
      )
    
    return dw_shape, dynamic_dense_inter_dim

  def _setup_dense_layers(self, cfg, dw_shape, dynamic_dense_inter_dim):
    """Setup the dense projection layers."""
    layer_kwargs = dict(
      dtype=cfg.dtype, 
      weight_dtype=cfg.weight_dtype, 
      quant=self.quant
    )
    
    # First projection layer (expansion)
    self.dense_proj1 = linears.DenseGeneral(
      dynamic_dense_inter_dim,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=('embed', 'kv'),
      use_bias=False,
      name='dynamic_dense_conn1',
      **layer_kwargs
    )
    
    self.dense_activation = linears._convert_to_activation_function(
      cfg.dynamic_dense_act_cls
    )
    
    # Second projection layer (compression)
    output_dim = np.prod(dw_shape) if cfg.mudd_use_muon else dw_shape
    kernel_axes = ('kv', 'mlp') if cfg.mudd_use_muon else ('kv', None, 'mlp')
    
    self.dense_proj2 = linears.DenseGeneral(
      output_dim,
      kernel_init=initializers.contant_dense_init(0.0),
      kernel_axes=kernel_axes,
      use_bias=False,
      name='dynamic_dense_conn2',
      **layer_kwargs
    )

  def _setup_bias(self, cfg, dw_shape):
    """Setup bias parameters if needed."""
    if not self.use_bias:
      return
      
    # Determine bias initialization value
    bias_init_value = (
      0.0 if cfg.mudd_prenorm and cfg.mudd_postnorm else 1.0
    )
    
    # Create bias initialization array
    bias_values = [0] * (dw_shape[1] - 1) + [bias_init_value]
    init_array = jnp.array(bias_values).astype(cfg.weight_dtype)
    init_array = init_array[None].repeat(self.C, axis=0)
    
    self.dense_proj2_bias = self.param(
      "dense_proj2.bias", 
      init_fn=lambda rng: init_array
    )

  def setup(self):
    """Setup all components of the MLP."""
    cfg = self.config
    if not cfg.dense_conn:
      return
    
    # Setup normalization
    self.pre_dense_proj1_norm = self._setup_norm_layer(cfg)
    
    # Compute dimensions
    self.dw_shape, self.dynamic_dense_inter_dim = self._compute_dynamic_dimensions(cfg)
    
    # Setup dense layers
    self._setup_dense_layers(cfg, self.dw_shape, self.dynamic_dense_inter_dim)
    
    # Setup bias if needed
    self._setup_bias(cfg, self.dw_shape)

  @nn.compact
  def __call__(self, layer_output):
    """Forward pass of the MLP.
    
    Args:
      layer_output: Input tensor from the previous layer
      
    Returns:
      Dynamic dense weights or None if dense connection is disabled
    """
    cfg = self.config
    
    # Early return if dense connection is disabled
    if not cfg.dense_conn or cfg.dynamic_dense_type != 'qkvm':
      return None
    
    # Forward pass through MLP layers
    normalized_input = self.pre_dense_proj1_norm(layer_output)
    hidden = self.dense_activation(self.dense_proj1(normalized_input))
    output = self.dense_proj2(hidden)
    
    # Reshape for muon configuration
    if cfg.mudd_use_muon:
      output = output.reshape(*output.shape[:-1], *self.dw_shape)
    
    # Apply scaling if configured
    if cfg.dynamic_dense_scale_dw:
      output /= jnp.sqrt(self.dynamic_dense_inter_dim)
    
    # Add bias if configured
    if self.use_bias:
      bias = self.dense_proj2_bias.astype(output.dtype)
      output = output + bias
    
    return output


class Compose(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  layer_inx: int = None
  C: int = 4

  def _compute_channel_count(self, cfg):
    """Compute the number of channels based on layer position."""
    is_last_layer = (
      self.layer_inx == cfg.num_decoder_layers + cfg.mtp_num_layers - 1
    )
    return 1 if is_last_layer else len(cfg.dynamic_dense_type)

  def _record_metrics(self, cfg, layer_output, dynamic_weights, layer_idx):
    """Record internal metrics if configured."""
    if not cfg.record_internal_nn_metrics:
      return
      
    # Record dynamic weight metrics
    for operation in [jnp.max, jnp.mean, jnp.min, jnp.std, l2norm]:
      metric_name = f'dyn_dense_w/{operation.__name__}/layer_{layer_idx}'
      metric_value = operation(dynamic_weights.astype(jnp.float32))
      self.sow('intermediates', metric_name, metric_value)
    
    # Record layer output norm
    output_norm = l2norm(layer_output.astype(jnp.float32))
    self.sow('intermediates', f'layer_output/norm/layer_{layer_idx}', output_norm)

  def _apply_prenorm_and_update_hidden_states(self, cfg, layer_output, hidden_states):
    """Apply prenormalization and update hidden states list."""
    if cfg.mudd_prenorm:
      normed_output = normalizations.get_rmsnorm(
        name="mudd_prenorm", cfg=cfg
      )(layer_output)
    else:
      normed_output = layer_output
      
    hidden_states.append(normed_output)
    return hidden_states

  def _compute_weighted_combinations(self, cfg, dynamic_weights, hidden_states, 
                                   channel_count, layer_output):
    """Compute weighted combinations of hidden states."""
    # Reshape weights: B T C L -> C B T L 1
    reshaped_weights = rearrange(
      dynamic_weights, 'B T C L -> C B T L 1', C=channel_count
    )
    combinations = []
    for channel_idx in range(channel_count):
      channel_weights = reshaped_weights[channel_idx:channel_idx + 1]
      weighted_combination = weighted_sum(
        channel_weights, hidden_states, cfg.ddw_gen_chunk_size
      ).squeeze(0)
      
      if cfg.mudd_postnorm:
        if channel_idx == channel_count - 1:
          # Apply post-normalization for the last channel
          post_norm = normalizations.get_rmsnorm(
            name="mudd_postnorm", 
            cfg=cfg, 
            scale_init=nn.initializers.constant(DEFAULT_POST_NORM_SCALE)
          )
          weighted_combination = post_norm(weighted_combination)
        
        # Add to original layer output for residual connection
        combination = layer_output + weighted_combination
      else:
        combination = weighted_combination
        
      combinations.append(combination)
    
    return tuple(combinations)

  @nn.compact
  def __call__(self, layer_output, hidden_states):
    """Compose hidden states using dynamic weights.
    
    Args:
      layer_output: Output from the current layer
      hidden_states: List of previous layer hidden states
      
    Returns:
      Tuple of (composed_outputs, updated_hidden_states)
    """
    cfg = self.config
    
    # Early return if dense connection is disabled
    if not cfg.dense_conn:
      return layer_output, hidden_states
    
    # Compute channel count -> C
    channel_count = self._compute_channel_count(cfg)
    
    # Generate dynamic weights using MLP
    mlp = Mlp(
      self.config, self.mesh, self.quant, self.layer_inx, 
      name='mlp', C=channel_count
    )
    dynamic_weights = mlp(layer_output)
    
    # Record metrics if configured
    self._record_metrics(cfg, layer_output, dynamic_weights, self.layer_inx)
    
    # Apply prenormalization and update hidden states
    hidden_states = self._apply_prenorm_and_update_hidden_states(
      cfg, layer_output, hidden_states
    )
    
    # Compute weighted combinations
    composed_outputs = self._compute_weighted_combinations(
      cfg, dynamic_weights, hidden_states, channel_count, layer_output
    )
    
    return composed_outputs, hidden_states
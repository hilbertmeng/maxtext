#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import common_types
from layers import initializers
from layers import normalizations
from layers import quantizations
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
import max_logging
from flax import struct

try:
  from jax.experimental.pallas.ops.tpu import megablox as mblx
except ImportError:
  max_logging.log("JAX megablox is available for TPU only.")
  pass

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization

BATCH = "activation_batch"

def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _entroy(probs):
  log_probs = jnp.log2(jnp.maximum(1.0e-30, probs))
  mean_sum_plogp = jnp.mean(- jnp.sum(log_probs * probs, axis=-1))
  return mean_sum_plogp


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[str, ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    def compute_dot_general(inputs, kernel, axis, contract_ind):
      """Computes a dot_general operation that may be quantized."""
      dot_general = lax.dot_general
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls(mesh_axes=self.kernel_axes)
        dot_general = dot_general_cls()
      return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      kernel = jnp.zeros(kernel_shape)
    else:
      kernel = self.param(
          "kernel",
          nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    output = compute_dot_general(inputs, kernel, axis, contract_ind)

    if self.use_bias:
      bias_axes, bias_shape = self.kernel_axes[-len(features) :], kernel_shape[-len(features) :]
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ("relu",)
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def fake_mgate(self, inputs):
    gate_scores = DenseGeneral(
            self.config.mgate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name='mgate',
            quant=self.quant,
            use_bias=self.use_bias,
        )(inputs)
    gate_scores = jax.nn.softmax(gate_scores.astype(jnp.float32), axis=-1)
    gate_scores = gate_scores.astype(self.dtype)
    if self.config.record_internal_nn_metrics:
      expert_to_token_score = gate_scores.mean(axis=(0,1))
      sum_value = jnp.sum(expert_to_token_score, axis=-1)
      expert_to_token_score = expert_to_token_score / (sum_value + 1e-6)
      self.sow('intermediates', 'expert_to_token_score', _entroy(expert_to_token_score)) # 熵越大越好 max: 5.45
      self.sow('intermediates', 'token_to_expert_score', _entroy(gate_scores)) # 熵越小越好
    return gate_scores

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm: # False
      inputs = self.get_norm_layer()(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    gate_scores = self.fake_mgate(inputs) if cfg.mgate else None
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    print(f'self.intermediate_dim: {self.intermediate_dim}')
    if cfg.fused_mlp:
      x = DenseGeneral(
          (len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="wi",
          quant=self.quant,
          use_bias=self.use_bias,
      )(inputs)
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations): # activations: ["silu", "linear"]# 说明wi_0是gate
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        x = DenseGeneral(
            self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
        )(inputs)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    x = checkpoint_name(x, "mlpwi")
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, (BATCH, "activation_length", "activation_mlp"))
    print(f'inputs.shape: {inputs.shape} xshape: {x.shape}')

    if gate_scores is not None:
      assert isinstance(cfg.mgate_dim, int)
      print(f'gate_scores is not None and mgate_dim={cfg.mgate_dim}')
      B, T, F = x.shape
      x = x.reshape(B, T, cfg.mgate_dim, F // cfg.mgate_dim)
      x = jnp.einsum('BTE,BTEM->BTEM', gate_scores, x)
      x = x.reshape(B, T, F)

    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output


class MoeBlock(nn.Module):
  """Mixture of Experts (MoE) block.

  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32

  def generate_kernels(self, num_experts, emb_dim, mlp_dim):

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')

    # The first axes is expert
    kernel_axes = (None, 'embed', 'mlp')
    wo_kernel_axes = (None, 'mlp', 'embed')

    w0_kernel = self.param(
        'wi_0',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, emb_dim, mlp_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    w0_kernel = jnp.asarray(w0_kernel, self.dtype)
    w1_kernel = self.param(
        'wi_1',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (num_experts, emb_dim, mlp_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)
    wo_kernel = self.param(
        'wo',
        nn.with_logical_partitioning(kernel_init, wo_kernel_axes),
        (num_experts, mlp_dim, emb_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  def permute(self, inputs, gate_logits, emb_dim):
    """Permute tokens to group by expert to fit gmm call."""

    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_2d = jnp.reshape(inputs, (-1, emb_dim))
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(self.weight_dtype), axis=-1).astype(self.dtype)
    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size

  def unpermute(self, intermediate, sorted_selected_experts, weights):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(unsort_intermediate, (-1, self.num_experts_per_tok, self.config.emb_dim))
    with jax.named_scope("weight_sum"):
      output = jnp.einsum("BKE,BK -> BE", reshaped_intermediate, reshaped_weights)
    return output.reshape(-1, self.config.max_target_length, self.config.emb_dim).astype(self.dtype)

  def megablox(self, inputs, gate_logits, config, w0_kernel, w1_kernel, wo_kernel):
    # TODO(ranran): need to changes in JAX repo to enable optimized tile_size
    #               instead of the static default tile_size (512, 512, 512)
    tile_size = (512, 512, 512)

    def gmm(inputs, kernel, group_sizes):
      hs_shape = inputs.shape
      # pad length is the 1st dimension of tiling size in gmm call
      pad_length = 512
      if hs_shape[0] % pad_length:
        pad_length = pad_length - hs_shape[0] % pad_length
        inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0,0,0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.weight_dtype)
      output = mblx.gmm(lhs=inputs,
                        rhs=kernel,
                        group_sizes=group_sizes,
                        preferred_element_type=jnp.bfloat16,
                        tiling=tile_size)

      if hs_shape[0] % pad_length:
        output = output[:hs_shape[0]]
      return output

    # Currently, we only support data parallelism with Megablox (sharding on batch dimensions)
    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(
              (nn.logical_to_mesh_axes((BATCH, None, None))),
              (nn.logical_to_mesh_axes((BATCH, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
              (nn.logical_to_mesh_axes((None, None, None))),
          ),
        out_specs=(nn.logical_to_mesh_axes((BATCH, None, None))),
        check_rep=False,
    )
    def wrapper(x, logits, w0, w1, wo):
      x, sorted_selected_experts, weights, group_sizes = self.permute(x, logits, config.emb_dim)

      layer_w0 = gmm(x, w0, group_sizes)
      layer_w1 = gmm(x, w1, group_sizes)
      layer_act = _convert_to_activation_function(config.mlp_activations[0])(layer_w0)
      intermediate_layer = jnp.multiply(layer_act, layer_w1)
      intermediate_output = gmm(intermediate_layer, wo, group_sizes)
      output = self.unpermute(intermediate_output,
                              sorted_selected_experts,
                              weights)
      return output
    return wrapper(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name="gate")(inputs)

    top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    flattened_top_k_weights = top_k_weights.reshape(-1, self.num_experts_per_tok)

    softmax_probs = jax.nn.softmax(flattened_top_k_weights.astype(jnp.float32), axis=-1).astype(self.weight_dtype)
    softmax_probs = softmax_probs.reshape(gate_logits.shape[:-1] + (self.num_experts_per_tok,))

    weights = jnp.zeros_like(gate_logits)
    index_update = (jnp.arange(gate_logits.shape[0])[:, None, None], jnp.arange(gate_logits.shape[1])[:, None], top_k_indices)
    weights = weights.at[index_update].set(softmax_probs)

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts,
                                                            cfg.emb_dim,
                                                            cfg.mlp_dim)

    if cfg.megablox:
      max_logging.log("Running MoE megablox implementation.")
      return self.megablox(inputs, gate_logits, cfg, w0_kernel, w1_kernel, wo_kernel)
    else:
      max_logging.log("Running MoE matmul implementation.")
      with jax.named_scope("wi_0"):
        layer_w0 = jnp.einsum("BLE,NEH -> BLNH", inputs, w0_kernel)
      with jax.named_scope("wi_1"):
        layer_w1 = jnp.einsum("BLE,NEH -> BLNH", inputs, w1_kernel)
      layer_w0_act = _convert_to_activation_function(cfg.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1)
      with jax.named_scope("wo"):
        intermediate_layer = jnp.einsum("BLNH,NHE -> BLNE", layer_multiply, wo_kernel)
      with jax.named_scope("w_sum"):
        output = jnp.einsum("BLNE,BLN -> BLE", intermediate_layer, weights)

    return output


Array = jnp.ndarray

def _take_along_axis(array: Array, indices: Array, axis: int) -> Array:
    if array.ndim != indices.ndim:
        raise ValueError(
            'indices and array must have the same number of dimensions; '
            f'{indices.ndim} vs. {array.ndim}.')

    if (axis != -1 and axis != array.ndim - 1 and  # Not last dimension
        axis != 1 and axis != -array.ndim + 1):  # Not second dimension
        raise ValueError(
            'Only slices along the second or last dimension are supported; '
            f'array.ndim = {array.ndim}, while axis = {axis}.')

    if _favor_one_hot_slices():
        one_hot_length = array.shape[axis]
        one_hot_indices = jax.nn.one_hot(indices, one_hot_length, axis=axis)

        if axis == -1 or array.ndim == 1:
            result = jnp.einsum(
                '...s,...is->...i',
                array,
                one_hot_indices,
                precision=jax.lax.Precision.HIGHEST)
        else:
            result = jnp.einsum(
                'ns...,nis...->ni...',
                array,
                one_hot_indices,
                precision=jax.lax.Precision.HIGHEST)
        return jax.lax.convert_element_type(result, array.dtype)
    else:
        return jnp.take_along_axis(array, indices, axis=axis)
        
def _favor_one_hot_slices() -> bool:
  return jax.default_backend() == 'tpu' or jax.devices()[0].platform == 'tpu'


def _load_balancing_loss(router_probs: Array, expert_indices: Array) -> float:
  num_experts = router_probs.shape[-1]
  # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
  expert_mask = jax.nn.one_hot(expert_indices, num_experts, dtype=jnp.int32)
  # For a given token, determine if it was routed to a given expert.
  # Shape: [num_groups, tokens_per_group, num_experts]
  expert_mask = jnp.max(expert_mask, axis=-2)

  tokens_per_group_and_expert = jnp.mean(
      expert_mask, dtype=jnp.float32, axis=-2)
  router_prob_per_group_and_expert = jnp.mean(
      router_probs, dtype=jnp.float32, axis=-2)
  return (
      jnp.mean(  # pytype: disable=bad-return-type  # jnp-type
          tokens_per_group_and_expert * router_prob_per_group_and_expert,
          dtype=jnp.float32,
      )
      * num_experts**2
  )


def _top_k(array, k: int):
    if _favor_one_hot_slices():
        top_k_indices = jax.lax.top_k(array, k)[-1]
        top_k_values = _take_along_axis(array, top_k_indices, axis=-1)
        return top_k_values, top_k_indices
    else:
        return jax.lax.top_k(array, k)


@struct.dataclass
class AuxLossStruct:
    value: Array
    weight: Array


class DcMoeBlock(nn.Module):

    config: Config
    mesh: Mesh
    kernel_init: NdInitializer
    kernel_axes: Tuple[str, ...]
    weight_dtype: DType = jnp.float32
    dtype: DType = jnp.float32
    num_experts: int = 0
    intermediate_dim: int = 4096

    def setup(self):

        kernel_in_axis = np.arange(1)
        kernel_out_axis = np.arange(1, 2)
        kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')

        # The first axes is expert
        kernel_axes = (None, 'embed', 'mlp')
        wo_kernel_axes = (None, 'mlp', 'embed')
       
        # self.num_experts = self.config.num_experts - n_shared_experts
        mlp_dim = self.intermediate_dim # moe dim
        emb_dim = self.config.base_emb_dim  # model dim

        self.num_experts_per_tok = self.config.num_experts_per_tok
        self.expert_capacity_factor = self.config.expert_capacity_factor
        self.min_group_size = self.config.min_group_size
        self.router_z_loss_coef = self.config.router_z_loss_coef
        self.aux_loss_coef = self.config.aux_loss_coef

        self.expert_chunk_size = self.config.expert_chunk_size
        self.num_groups = self.config.num_groups

        w0_kernel = self.param(
            'wi_0',
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            (self.num_experts, emb_dim, mlp_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
        self.wi_0 = jnp.asarray(w0_kernel, self.dtype)
        
        w1_kernel = self.param(
            'wi_1',
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            (self.num_experts, emb_dim, mlp_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
        self.wi_gate_0 = jnp.asarray(w1_kernel, self.dtype)
        
        wo_kernel = self.param(
            'wo',
            nn.with_logical_partitioning(kernel_init, wo_kernel_axes),
            (self.num_experts, mlp_dim, emb_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
        self.wo_0 = jnp.asarray(wo_kernel, self.dtype)

        self._is_ffn1_gated = True if self.config.mlp_activations[0] != 'linear' else False
        # silu
        self.activation = _convert_to_activation_function(self.config.mlp_activations[0])

    @nn.compact
    def __call__(self, inputs, paddings, enable_dropout=True):
        inputs = inputs.astype(self.dtype)
        combined_outputs, aux_loss = self._dispatch_and_combine_expert_outputs_openmoe(inputs, paddings)
        return combined_outputs, aux_loss

    @nn.nowrap
    def add_aux_loss(self, name: str, value: Array, weight=None):
        # Accumulate by summing aux_loss.
        if weight is None:
            weight = jnp.ones_like(value)

        def reduce_fn(x, y):
            assert isinstance(x, AuxLossStruct)
            assert isinstance(y, AuxLossStruct)
            return AuxLossStruct(value=x.value + y.value, weight=x.weight + y.weight)

        self.sow(
            'intermediates',  # 会在最后的结果中返回
            name,
            AuxLossStruct(value, weight),
            init_fn=lambda: AuxLossStruct(
                0.0, 0.0
            ), 
            reduce_fn=reduce_fn,
        )

    def _split(self, x, specs):
        return x

    def _call_experts(self, expert_inputs, expert_index, compute_n_expert):
        """
        expert_inputs: gecm
        """
        theta_wi, theta_wo = self.wi_0[expert_index: expert_index + compute_n_expert], self.wo_0[expert_index: expert_index + compute_n_expert]
        if self._is_ffn1_gated:
            theta_wi_gated = self.wi_gate_0[expert_index: expert_index + compute_n_expert]

        num_groups, num_experts, capacity, *hidden_dims = expert_inputs.shape
        assert num_experts == theta_wi.shape[0]
       
        expert_inputs = self._split(expert_inputs, (('replica', 'data'), None, None,'mdl'))

        if self._is_ffn1_gated:
            print(f'expert_inputs: {expert_inputs.shape} theta_wi: {theta_wi.shape}')
            hidden0 = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi)
            hidden1 = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi_gated)
            hidden1 = self.activation(hidden1)
            hidden = hidden1 * hidden0
            hidden = self._split(hidden, (('replica', 'data'), None, None, 'mdl'))
        else:
            hidden = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi)
            hidden = self._split(hidden, (('replica', 'data'), None, None, 'mdl'))
            hidden = self.activation(hidden)

        expert_output = jnp.einsum("gech,ehm->gecm", hidden, theta_wo)
        expert_output = self._split(expert_output, (('replica', 'data'), None, None, 'mdl'))
        
        return expert_output
        
    def _dispatch_and_combine_expert_outputs_openmoe(self, inputs, paddings):

        print(f'Enter openmoe top2 router.....')
        topn = self.num_experts_per_tok
        token_shape = inputs.shape[:-1]
        num_tokens = np.prod(token_shape)
        m_dim = inputs.shape[-1]
       
        num_groups = self.num_groups
        tokens_per_group = num_tokens // num_groups
        assert num_tokens % num_groups == 0, print(f'‘num_tokens % num_groups -> {num_tokens} % {num_groups} != 0’')

        print(f'expert_capacity_factor: {self.expert_capacity_factor}')
        expert_capacity = int(self.expert_capacity_factor * tokens_per_group / self.num_experts)
        max_group_size = float(inputs.shape[1]) * self.expert_capacity_factor
        expert_capacity = min(expert_capacity, max_group_size)
        expert_capacity = max(expert_capacity, self.min_group_size)
        print(f'expert_capacity: {expert_capacity}')
       
        # gsm
        grouped_inputs = jnp.reshape(inputs, (num_groups, tokens_per_group, self.config.base_emb_dim))
        # grouped_inputs = self._split(grouped_inputs, (('replica', 'data'), None, 'mdl'))
        token_inputs = jax.lax.convert_element_type(grouped_inputs, jnp.float32)
        print(f'token_inputs: {token_inputs.shape}')

        router_logits = DenseGeneral(
                self.num_experts,
                dtype=self.dtype,
                weight_dtype=self.weight_dtype,
                kernel_init=self.kernel_init,
                kernel_axes=self.kernel_axes,
                name="gate")(token_inputs)
        # gse
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # g * s * top2
        expert_gate, expert_index = _top_k(router_probs, k=topn)
    
        if paddings is not None:
            print(f'paddings: {paddings.shape}')
            print(f'token_shape: {token_shape}')
            
            assert paddings.shape == token_shape
            # 如果paddings中的0表示保留，则 nonpaddings = 1.0 - paddings  
            nonpaddings = paddings
            nonpaddings = jnp.reshape(nonpaddings, grouped_inputs.shape[:2])
            gate_mask = jnp.expand_dims(nonpaddings, axis=-1)
            expert_gate *= gate_mask
    
            expert_index *= (2 * gate_mask - 1.)
            expert_index += jnp.repeat(gate_mask - 1., topn, axis=-1)
            router_probs *= gate_mask

        if self.aux_loss_coef is not None:
            aux_loss = _load_balancing_loss(router_probs, expert_index)  # 各个专家之间实现均衡的负载分配
            aux_loss *= self.aux_loss_coef
        else:
            aux_loss = 0.0
        
        # lsp
        if self.router_z_loss_coef is not None:  # 目的是避免路由器的输出变得过于极端或不稳定，确保概率分布不会集中在极少数的专家上
            # <=> torch.logsumexp(logits, dim = -1)
            router_z_loss = jnp.log(jnp.sum(jnp.exp(router_logits), axis=-1))
            router_z_loss = jnp.square(router_z_loss)            
            router_z_loss = self.router_z_loss_coef * router_z_loss.mean()
        else:
          router_z_loss = 0.0

        aux_loss += router_z_loss

        # g * 2 * s
        expert_index = jnp.swapaxes(expert_index, 1, 2)
        # g * 2s
        expert_index = expert_index.reshape(num_groups, -1)
        # g * 2s * e, expert_index 负值的地方忽略了?
        expert_mask = jax.nn.one_hot(expert_index, self.num_experts, dtype=jnp.int32)
    
        # g * 2s * e
        token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
        # g * 2 * s * e
        token_priority = token_priority.reshape(num_groups, topn, -1, self.num_experts)
        # g * s * 2 * e
        token_priority = jnp.swapaxes(token_priority, 1, 2)
        # g * s *  e
        token_priority = jnp.max(token_priority, axis=2)
        # g * s *  e
        token_priority = self._split(token_priority, (('replica', 'data'), None, None))
    
        if self.expert_chunk_size is None:
            compute_n_expert = self.num_experts
        else:
            compute_n_expert = self.num_experts // self.expert_chunk_size
            assert self.num_experts % self.expert_chunk_size == 0
    
        combined_outputs = None
        print(f'compute_n_expert: {compute_n_expert}')
        for expert_index in range(0, token_priority.shape[2], compute_n_expert):
            # print(f'expert_index: {expert_index}')
            _token_priority = token_priority[..., expert_index: expert_index+compute_n_expert]
            _router_probs = router_probs[..., expert_index: expert_index+compute_n_expert]
            # 专家概率mask： g * s * e * c
            _dispatch_mask = jax.nn.one_hot(_token_priority, expert_capacity, dtype=jnp.bool_)
            # gsec
            _combine_array = jnp.einsum('...se,...sec->...sec', _router_probs, _dispatch_mask)
            _combine_array = jax.lax.convert_element_type(_combine_array, self.dtype)
            # 专家的输入mask：gsm x gsec -> gecm
            _expert_inputs = jnp.einsum('gs...,gsec->gec...', token_inputs, _dispatch_mask)
            _expert_inputs = jax.lax.convert_element_type(_expert_inputs, self.dtype)
            # gecm
            # print(f'_expert_inputs: {_expert_inputs.shape}')
            # g * e * c * m
            _expert_outputs = self._call_experts(_expert_inputs, expert_index, compute_n_expert)
            _combined_outputs = jnp.einsum('gec...,gsec->gs...', _expert_outputs, _combine_array)
            combined_outputs = _combined_outputs if combined_outputs is None else combined_outputs + _combined_outputs
            # print(f'combined_outputs-{expert_index}: {combined_outputs}')

        self.add_aux_loss("aux_loss", aux_loss)
        # Return to batched shape.
        combined_outputs = combined_outputs.reshape(*inputs.shape)
        return combined_outputs, aux_loss

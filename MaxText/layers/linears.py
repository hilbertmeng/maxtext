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

import flax
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
import math
import max_logging
import max_utils
from aqt.jax.v2 import aqt_tensor
from kernels import megablox as mblx


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization
QTensor = aqt_tensor.QTensor


DISPATCH = "dispatch"
COMBINE = "combine"


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


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation.
    bias_norm: whether to add normalization before adding bias.
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  kernel_axes: Tuple[Optional[str], ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False
  bias_norm: str = ""
  matmul_precision: str = "default"

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
      matmul_precision = lax.Precision(self.matmul_precision)
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls(mesh_axes=self.kernel_axes)
        dot_general = dot_general_cls()
        return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
      return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)

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
      bias_axes, bias_shape = (
          self.kernel_axes[-len(features) :],
          kernel_shape[-len(features) :],
      )
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)

      if self.bias_norm:
        output = _convert_to_activation_function(self.bias_norm)(output)
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
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma", "deepseek"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = self.get_norm_layer()(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
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
          matmul_precision=self.config.matmul_precision,
      )(inputs)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
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
            matmul_precision=self.config.matmul_precision,
        )(inputs)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))

    # lsp mgate
    x = Mgate(config=self.config,
              kernel_init=self.kernel_init,
              weight_dtype=self.weight_dtype,
              dtype=self.dtype,
              quant=self.quant,
              name='mgate',
            )(layer_inputs=inputs, hidden=x, unsqueeze=True)

    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
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
    intermediate_dim: Intermediate dimension of MoE.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  intermediate_dim: int = 2048
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  # The first axes is expert
  wi_kernel_axes = ("exp", "embed_no_exp", "mlp")
  wo_kernel_axes = ("exp", "mlp", "embed_no_exp")

  def generate_kernels(self, num_experts, emb_dim, mlp_dim):

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    kernel_init = nd_dense_init(1.0, "fan_in", "truncated_normal")

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w0_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w0_kernel = self.param(
          "wi_0",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )

    w0_kernel = jnp.asarray(w0_kernel, self.dtype)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      w1_kernel = jnp.zeros((num_experts, emb_dim, mlp_dim))
    else:
      w1_kernel = self.param(
          "wi_1",
          nn.with_logical_partitioning(kernel_init, self.wi_kernel_axes),
          (num_experts, emb_dim, mlp_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    w1_kernel = jnp.asarray(w1_kernel, self.dtype)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retrieved from the tensors stored in the 'aqt' collection.
      wo_kernel = jnp.zeros((num_experts, mlp_dim, emb_dim))
    else:
      wo_kernel = self.param(
          "wo",
          nn.with_logical_partitioning(kernel_init, self.wo_kernel_axes),
          (num_experts, mlp_dim, emb_dim),
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
    wo_kernel = jnp.asarray(wo_kernel, self.dtype)
    return w0_kernel, w1_kernel, wo_kernel

  def deepseek_scale_weights(self, weights):
    """Scales weights according to DeepSeek's v3 reference implementation.
    https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594
    """
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def permute(self, inputs, gate_logits):
    """Permute tokens to group by expert to fit gmm call."""

    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    inputs_2d = jnp.reshape(inputs, (inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    if self.config.decoder_block == "deepseek":
      weights = self.deepseek_scale_weights(weights)
    else:
      weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.dtype)
    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size

  def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = lax.Precision(self.config.matmul_precision)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate.astype(jnp.float32),
          reshaped_weights.astype(jnp.float32),
          precision=matmul_precision,
      )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  def sparse_matmul(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    tile_size = (512, 1024, 1024)  # (m, k, n)

    def gmm(inputs, kernel, group_sizes):
      hs_shape = inputs.shape
      # pad length is the 1st dimension of tiling size in gmm call
      pad_length = 512
      if hs_shape[0] % pad_length:
        pad_length = pad_length - hs_shape[0] % pad_length
        inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.dtype)

      lhs_quantize_dtype, rhs_quantize_dtype = None, None
      if self.quant is not None:
        quant_dg = self.quant.quant_dg
        lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
        rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()

      if self.config.megablox:
        m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
        output = mblx.gmm(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            preferred_element_type=jnp.bfloat16,
            tiling=(min(tile_size[0], m), min(tile_size[1], k), min(tile_size[2], n)),
            lhs_quantize_dtype=lhs_quantize_dtype,
            rhs_quantize_dtype=rhs_quantize_dtype,
        )
      else:
        if self.quant is not None:
          raise NotImplementedError("Quantization is not yet supported with ragged_dot, please set" " megablox=True")
        output = jax.lax.ragged_dot(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            preferred_element_type=jnp.bfloat16,
        )
      if hs_shape[0] % pad_length:
        output = output[: hs_shape[0]]
      return output

    # Currently, we only support data and tensor parallelism with Megablox.
    # We all gather the input activations over tensor parallelism to follow strategy
    # in https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.
    input_partition_spec = nn.logical_to_mesh_axes(("activation_batch", None, None))
    gate_logits_pspec = nn.logical_to_mesh_axes(("activation_batch", None, None))
    w0_pspec = nn.logical_to_mesh_axes((None, None, "mlp"))
    w1_pspec = nn.logical_to_mesh_axes((None, None, "mlp"))
    wo_pspec = nn.logical_to_mesh_axes((None, "mlp", None))

    if isinstance(w0_kernel, QTensor):
      w0_pspec = aqt_tensor.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
    if isinstance(w1_kernel, QTensor):
      w1_pspec = aqt_tensor.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
    if isinstance(wo_kernel, QTensor):
      wo_pspec = aqt_tensor.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)

    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(input_partition_spec, gate_logits_pspec, w0_pspec, w1_pspec, wo_pspec),
        out_specs=(nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed"))),
        check_rep=False,
    )
    def wrapper(x, logits, w0, w1, wo):
      batch_size, sequence_length, _ = x.shape
      x, sorted_selected_experts, weights, group_sizes = self.permute(x, logits)
      layer_w0 = gmm(x, w0, group_sizes)
      layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      layer_w1 = gmm(x, w1, group_sizes)
      layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      intermediate_layer = jnp.multiply(layer_act, layer_w1)
      intermediate_output = gmm(intermediate_layer, wo, group_sizes)
      intermediate_output = checkpoint_name(intermediate_output, "mlpwo")
      tensor_parallelism = self.config.ici_tensor_parallelism * self.config.dcn_tensor_parallelism
      if tensor_parallelism > 1:
        intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)
      output = self.unpermute(
          intermediate_output, sorted_selected_experts, weights, batch_size=batch_size, sequence_length=sequence_length
      )
      return output, None

    return wrapper(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

  def reshape_and_update_weights(self, weights, indices):
    # input of weights & indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (
        jnp.arange(weights.shape[0])[:, None, None],
        jnp.arange(weights.shape[1])[:, None],
        indices,
    )
    update_weights = update_weights.at[index_update].set(weights)
    return update_weights

  def generate_masks(self, top_k_indices, softmax_probs):
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log(f"Applying potential token dropping with a batch expert_capacity of {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to expert [0, 1] & [1, 3],
    # then expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(expert_mask, (batch_size, seq_len * self.num_experts_per_tok, self.num_experts))
    expert_mask_fused = nn.with_logical_constraint(expert_mask_fused, ("activation_batch", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = nn.with_logical_constraint(
        expert_token_count, ("activation_batch", "activation_length", None, None)
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    # reshape & update weights
    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)
    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits):
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(self, rhs_mesh_axes: Tuple[Optional[str], ...] = (), einsum_name=None):

    # the check is to prevent aqteinsum as einsum op for dispatch and combine einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and (einsum_name == DISPATCH or einsum_name == COMBINE):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):
        # simply skip kwargs, since aqt einsum doesn't support any kwargs like precision
        is_aqt = not isinstance(self.quant, quantizations.Fp8Quantization)
        kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
        return self.quant.einsum(**kw)(*args)

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def is_expert_parallelism_enabled(self):
    return self.config.ici_expert_parallelism > 1 or self.config.dcn_expert_parallelism > 1

  def maybe_all_gather_kernel_weight_in_expert_parallelism(self, kernel, kernel_axes):
    if self.is_expert_parallelism_enabled():
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead of weight_dtype
      kernel = nn.with_logical_constraint(kernel, kernel_axes)
    return kernel

  def dense_matmul(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    # gate_logits: batch, length, expert
    gate_logits = nn.with_logical_constraint(gate_logits, ("activation_batch", "activation_length", None))
    # shape of top_k_weights & top_k_indices: (batch, sequence, num_experts_per_tok)
    top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block == "deepseek":
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    else:
      top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

    weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
    matmul_precision = lax.Precision(self.config.matmul_precision)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      dispatch_mask, combine_mask = self.generate_masks(top_k_indices, weights)
      mask_axes = ("activation_batch", "activation_length", None, None)
      dispatch_mask = nn.with_logical_constraint(dispatch_mask, mask_axes)
      combine_mask = nn.with_logical_constraint(combine_mask, mask_axes)
      if self.config.model_call_mode != "inference":
        loss = self.load_balance_loss(top_k_indices, weights)
      else:
        loss = None
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("dispatch"):
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            "BSM,BSEC -> EBCM", inputs, dispatch_mask, precision=matmul_precision
        )
        dispatch = nn.with_logical_constraint(
            dispatch,
            ("activation_exp", "activation_batch_no_exp", None, "activation_embed"),
        )
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            "EBCM,EMH -> EBCH", dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = nn.with_logical_constraint(
            layer_w0,
            ("activation_exp", "activation_batch_no_exp", None, "activation_mlp"),
        )
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            "EBCM,EMH -> EBCH", dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = nn.with_logical_constraint(
            layer_w1,
            ("activation_exp", "activation_batch_no_exp", None, "activation_mlp"),
        )
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_w0_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            "EBCH,EHM -> EBCM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        intermediate_layer = nn.with_logical_constraint(
            intermediate_layer,
            ("activation_exp", "activation_batch_no_exp", None, "activation_embed"),
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            "EBCM,BSEC -> BSM",
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, loss
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_w0_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("w_sum"):
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
        ).astype(self.dtype)
      return output, None

  def retrieve_quantized_weight(
      self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel
  ) -> tuple[QTensor, QTensor, QTensor]:
    # This is called only during tracing. This is to invoke creation of quantized tensor inside AqtEinsum.
    # After jit, this will become no-op and will not affect performance.
    _ = self.dense_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  @nn.compact
  def __call__(self, inputs, padding=None): # lsp
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
        self.num_experts,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        name="gate",
        use_bias=self.config.routed_bias,
        bias_norm=self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
    )(inputs)

    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts, cfg.emb_dim, self.intermediate_dim)
    if cfg.sparse_matmul:
      max_logging.log("Running MoE sparse matmul implementation.")
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel
        )
      return self.sparse_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)
    else:
      max_logging.log("Running MoE dense matmul implementation.")
      return self.dense_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)


class DeepSeekMoeBlock(nn.Module):
  """DeepSeek MoE block, combining shared and routed experts.

  Attributes:
    config: Model configs.
    mesh: Mesh, device mesh.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    weight_dtype: Type for the weights.
    dtype: Type for the dense layer.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[Optional[str], ...]
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    routed_experts, _ = MoeBlock(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
    )(inputs)

    shared_experts = MlpBlock(
        intermediate_dim=cfg.shared_experts * cfg.moe_mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"shared_experts",
        config=cfg,
        quant=self.quant,
    )(inputs)

    return routed_experts + shared_experts


# ==================================================Add Mgate、OpenMoe code==========================================================
class Mgate(nn.Module):
  config: Config
  kernel_init: NdInitializer
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None

  def setup(self):
    if self.config.mgate_dim < 2: return
    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    # kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
    kernel_axes = ("exp", "embed_no_exp", "mlp")
    
    num_experts = 1 if self.config.num_experts <= 1 else self.config.num_experts
    mgate_kernel = self.param(
                      'kernel',
                      nn.with_logical_partitioning(self.kernel_init, kernel_axes),
                      (num_experts, self.config.emb_dim, self.config.mgate_dim),
                      self.weight_dtype,
                      kernel_in_axis,
                      kernel_out_axis,
                      )
    self.kernel = jnp.asarray(mgate_kernel, self.dtype)
    
  @nn.compact
  def __call__(self,layer_inputs, hidden, expert_index=0, compute_n_expert=1, unsqueeze=False):

    if self.config.mgate_dim < 2: return hidden

    print(f'mgate layer_inputs: {layer_inputs.shape} hidden: {hidden.shape} expert_index: {expert_index} compute_n_expert: {compute_n_expert}')
    if unsqueeze:
      layer_inputs = layer_inputs[:, None] # add a dimension when not moe 
      hidden = hidden[:, None] # add a dimension when not moe 

    inner_gate = self.kernel[expert_index: expert_index + compute_n_expert]
    # x = jnp.einsum('BTE,BTEM->BTEM', gate_scores, x)  # 这里是多个专家一起计算mgate分数
    mgate_scores = jnp.einsum('gecm,emi->geci', layer_inputs, inner_gate)
    # mgate_scores = nn.with_logical_constraint(mgate_scores, ("activation_batch", "exp", "activation_length", None))
    max_logging.log(f'mgate is True  mgate_scores: {mgate_scores.shape}')
    mgate_scores = jax.nn.softmax(mgate_scores.astype(jnp.float32), axis=-1)
    mgate_scores = mgate_scores.astype(self.dtype)
    # if self.config.record_internal_nn_metrics:
    #   record_gate(self, 'mgate', mgate_scores, axis=(0, 1, 2))
    G, E, C, H = hidden.shape
    hidden = hidden.reshape(G, E, C, self.config.mgate_dim, H // self.config.mgate_dim)
    # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", None, "tensor"))
    hidden = jnp.einsum('geci,gecif->gecif', mgate_scores, hidden)
    
    hidden = hidden.reshape(G, C, H) if unsqueeze else hidden.reshape(G, E, C, H)
    # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", "tensor"))
    return hidden

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
        return jax.lax.convert_element_type(result, array.dtype), one_hot_indices.max(2)
    else:
        return jnp.take_along_axis(array, indices, axis=axis), None
        
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
        top_k_values, one_hot_indices = _take_along_axis(array, top_k_indices, axis=-1)
        return top_k_values, top_k_indices, one_hot_indices
    else:
        return jax.lax.top_k(array, k), None


@flax.struct.dataclass
class AuxLossStruct:
    value: Array
    weight: Array


def log(t, eps = 1e-20):
    return jnp.log(t.clip(min = eps))
    

def gumbel_noise(inputs, seed=9876, minval=0, maxval=1):
    noise = jax.random.uniform(jax.random.PRNGKey(seed), minval=minval, maxval=maxval, shape=inputs.shape,  dtype=inputs.dtype)
    return -log(-log(noise))


def _entroy(probs):
  log_probs = jnp.log2(jnp.maximum(1.0e-30, probs))
  mean_sum_plogp = jnp.mean(- jnp.sum(log_probs * probs, axis=-1))
  return mean_sum_plogp


def record_gate(self, key, gate_scores, axis=(0, 1)):
    expert_to_token_score = gate_scores.mean(axis=axis)
    sum_value = jnp.sum(expert_to_token_score, axis=-1)
    expert_to_token_score = expert_to_token_score / (sum_value + 1e-6)
    self.sow('intermediates', f'{key}/expert_to_token_score', _entroy(expert_to_token_score)) # 熵越大越好 max: 5.45
    self.sow('intermediates', f'{key}/token_to_expert_score', _entroy(gate_scores)) # 熵越小越好


class OpenMoeBlock(nn.Module):
    config: Config
    num_experts: int
    num_experts_per_tok: int
    mesh: Mesh
    kernel_init: NdInitializer
    kernel_axes: Tuple[str, ...]
    intermediate_dim: int = 4096
    weight_dtype: DType = jnp.float32
    dtype: DType = jnp.bfloat16
    quant: Optional[Quant] = None

    def setup(self):

        kernel_in_axis = np.arange(1)
        kernel_out_axis = np.arange(1, 2)
        kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
        # self.kernel_init = kernel_init
        # The first axes is expert
        kernel_axes = ("exp", "embed_no_exp", "mlp")
        wo_kernel_axes = ("exp", "mlp", "embed_no_exp")
  
        # self.num_experts = self.config.num_experts - shared_experts
        mlp_dim = self.intermediate_dim # moe dim
        emb_dim = self.config.base_emb_dim  # model dim

        self.expert_capacity_factor = self.config.expert_capacity_factor
        self.min_group_size = 1.0
        self.router_z_loss_coef = self.config.router_z_loss_coef
        self.aux_loss_coef = self.config.load_balance_loss_weight

        self.expert_chunk_size = self.config.expert_chunk_size

        # lsp：务必注意wi_0是需要过激活函数的，在这里称之为gate，小心别和dense的mlp搞反了
        w0_kernel = self.param(
            'wi_0',
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            (self.num_experts, emb_dim, mlp_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
        self.wi_gate_0 = jnp.asarray(w0_kernel, self.dtype)
        
        w1_kernel = self.param(
            'wi_1',
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            (self.num_experts, emb_dim, mlp_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
        self.wi_0 = jnp.asarray(w1_kernel, self.dtype)

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
    def __call__(self, inputs, paddings, deterministic=False):
        inputs = inputs.astype(self.dtype)
        combined_outputs, aux_loss = self._dispatch_and_combine_expert_outputs_openmoe(inputs, paddings, deterministic=deterministic)
        return combined_outputs, aux_loss

    # @nn.nowrap
    # def add_aux_loss(self, name: str, value: Array, weight=None):
    #     # Accumulate by summing aux_loss.
    #     if weight is None:
    #         weight = jnp.ones_like(value)

    #     def reduce_fn(x, y):
    #         assert isinstance(x, AuxLossStruct)
    #         assert isinstance(y, AuxLossStruct)
    #         return AuxLossStruct(value=x.value + y.value, weight=x.weight + y.weight)

    #     self.sow(
    #         'intermediates',  # 会在最后的结果中返回
    #         name,
    #         AuxLossStruct(value, weight),
    #         init_fn=lambda: AuxLossStruct(
    #             0.0, 0.0
    #         ), 
    #         reduce_fn=reduce_fn,
    #     )

    def _call_experts(self, expert_inputs, expert_index, compute_n_expert, deterministic=False):
        """
        expert_inputs: gecm
        """
      
        theta_wi, theta_wo = self.wi_0[expert_index: expert_index + compute_n_expert], self.wo_0[expert_index: expert_index + compute_n_expert]

        if self._is_ffn1_gated:
            theta_wi_gated = self.wi_gate_0[expert_index: expert_index + compute_n_expert]

        num_groups, num_experts, capacity, *hidden_dims = expert_inputs.shape
        assert num_experts == theta_wi.shape[0]
       
        # expert_inputs = nn.with_logical_constraint(expert_inputs, ("activation_batch", "exp", "activation_length", "tensor"))

        if self._is_ffn1_gated:
            max_logging.log(f'expert_inputs: {expert_inputs.shape} theta_wi: {theta_wi.shape}')
            hidden0 = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi)
            hidden1 = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi_gated)
            hidden1 = self.activation(hidden1)
            hidden = hidden1 * hidden0
            # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", "tensor"))
        else:
            hidden = jnp.einsum("gecm,emh->gech", expert_inputs, theta_wi)
            hidden = self.activation(hidden)
        #  Broadcast along length.
        hidden = nn.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))(hidden, deterministic=deterministic) 

        # expert_inputs: gecm,  mgatew: meh  -> 
        mgate_layer = Mgate(config=self.config,
              kernel_init=self.kernel_init,
              weight_dtype=self.weight_dtype,
              dtype=self.dtype,
              quant=self.quant,
              name='mgate',
            )
        hidden = mgate_layer(layer_inputs=expert_inputs, 
                            hidden=hidden, 
                            expert_index=expert_index, 
                            compute_n_expert=compute_n_expert)

        hidden = jnp.einsum("gech,ehm->gecm", hidden, theta_wo)
        # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", "tensor"))
        
        return hidden
        
    def _dispatch_and_combine_expert_outputs_openmoe(self, inputs, paddings, deterministic=False):

        max_logging.log(f'Enter openmoe top2 router.....')
        topn = self.num_experts_per_tok
        token_shape = inputs.shape[:-1]
        num_tokens = np.prod(token_shape)
        m_dim = inputs.shape[-1]
       
        num_groups = inputs.shape[0]
        tokens_per_group = num_tokens // num_groups
        assert num_tokens % num_groups == 0, max_logging.log(f'‘num_tokens % num_groups -> {num_tokens} % {num_groups} != 0’')

        max_logging.log(f'expert_capacity_factor: {self.expert_capacity_factor}')
        # expert_capacity = int(self.expert_capacity_factor * tokens_per_group / self.num_experts)
        expert_capacity = int(self.expert_capacity_factor * tokens_per_group / self.num_experts)
        max_group_size = int(inputs.shape[1])
        expert_capacity = min(expert_capacity, max_group_size)
        expert_capacity = max(expert_capacity, self.min_group_size)
        max_logging.log(f'expert_capacity: {expert_capacity}')
       
        # gsm
        grouped_inputs = jnp.reshape(inputs, (num_groups, tokens_per_group, self.config.base_emb_dim))
        token_inputs = jax.lax.convert_element_type(grouped_inputs, jnp.float32)
        max_logging.log(f'token_inputs: {token_inputs.shape}')

        router_logits = DenseGeneral(
                self.num_experts,
                dtype=self.dtype,
                weight_dtype=self.weight_dtype,
                kernel_init=self.kernel_init,
                kernel_axes=self.kernel_axes,
                name='gate')(token_inputs)

        # if self.config.record_internal_nn_metrics:
        #   self.sow('intermediates', 'router_logits/noiso_before/max', router_logits.max())
        #   self.sow('intermediates', 'router_logits/noiso_before/min', router_logits.min())

        if self.config.gate_noise_coef > 0.0:
          max_logging.log(f'gate_noise_coef: {self.config.gate_noise_coef}')
          noise = gumbel_noise(router_logits, seed=self.config.init_weights_seed)
          router_logits += noise * self.config.gate_noise_coef

          # if self.config.record_internal_nn_metrics:
          #   self.sow('intermediates', 'router_logits/noiso_after/max', router_logits.max())
          #   self.sow('intermediates', 'router_logits/noiso_after/min', router_logits.min())

        _, expert_index, one_hot_indices = _top_k(router_logits, k=topn)
        # NVIDIA：Upcycling Large Language Models into Mixture of Experts做法：
        # router_logits: b s * e -> b * s * G * e,  11组，每组8个专家，G11T11 one_hot_indices
        # one_hot_indices: b * s * top * e -> b * s * G * top * e , reshape -> b * s * (G * top) * e
        # expert_index: b s top -> b s G top,
        # 如果按照之前不分组的做法的话，router_logits  reshape：b s * (G e)
        # expert_index + range(0, 88, 8)

        if self.config.sfm_after_topn:
          assert one_hot_indices is not None
          max_logging.log(f'one_hot_indices is not None and sfm_after_topn is {self.config.sfm_after_topn}')
          router_mask = (1 - one_hot_indices) * jnp.finfo(self.dtype).min
          _router_logits = router_logits + router_mask
          router_probs = jax.nn.softmax(_router_logits.astype(jnp.float32), axis=-1)
          # router_probs /= router_probs.sum(-1, keepdims=True)
        else:
            # gse
          router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
        router_probs = router_probs.astype(self.dtype) # ble

        # router_probs = nn.with_logical_constraint(router_probs, ("activation_batch", "activation_length", "exp"))

        if self.config.record_internal_nn_metrics:
          # lsp note: slowly
          # l2norm = jnp.linalg.norm(router_logits.reshape(-1, router_logits.shape[-1]), ord=2, axis=(0, 1))
          l2norm = jnp.sqrt(jnp.sum(jnp.square(router_logits)))
          self.sow('intermediates', 'router_logits/l2norm', l2norm)
          record_gate(self, 'router_logits', router_logits, axis=(0, 1))
          # 解释：
          # expert2token： router_probs = [[0.] * 6 + [0.5, 0.5]], 极端均匀选择2个专家，熵最大，为1.0，
          # router_probs = [[0.] * 6 + [1.0, 0.0]]，极端不均匀选择2个专家，熵最大，为0.0。
          # token2expert： router_probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], 每个专家极端均匀选择token，熵最大，为3.0，
          # router_probs = [0] * 7 + [1.0, 0.]，每个专家极端不均匀选择token，熵最大，为0.0。
          record_gate(self, 'sfm_after_topn', router_probs, axis=(0, 1)) 
          # top2, expert2token: E=8, max: 3, min:0.5
          top_values = jnp.array([(expert_index == i).sum() for i in jnp.arange(0, self.num_experts, 1)])
          self.sow('intermediates', f'top/selected_expert_token_nums', top_values)
        
        # 有padding的时候放开, 一般预训练没有pad
        if paddings is not None:
            max_logging.log(f'paddings: {paddings.shape}')
            max_logging.log(f'token_shape: {token_shape}')
            
            assert paddings.shape == token_shape
            # 如果paddings中的0表示保留，则 nonpaddings = 1.0 - paddings  
            nonpaddings = paddings
            nonpaddings = jnp.reshape(nonpaddings, grouped_inputs.shape[:2])
            gate_mask = jnp.expand_dims(nonpaddings, axis=-1)
            # expert_gate *= gate_mask
    
            expert_index *= (2 * gate_mask - 1.) # lsp:将被mask的专家的所以变为负值，这样在之后转为one hot形式的时候就不会考虑
            expert_index += jnp.repeat(gate_mask - 1., topn, axis=-1)
            router_probs *= gate_mask # ble

        aux_loss, router_z_loss = 0.0, 0.0
        if self.aux_loss_coef is not None:
            aux_loss = _load_balancing_loss(router_probs, expert_index)  # 各个专家之间实现均衡的负载分配
            aux_loss *= self.aux_loss_coef
        if self.router_z_loss_coef is not None:  # 目的是避免路由器的输出变得过于极端或不稳定，确保概率分布不会集中在极少数的专家上  防止过大的logits
            # <=> torch.logsumexp(logits, dim = -1)
            router_z_loss = jnp.log(jnp.sum(jnp.exp(router_logits), axis=-1))
            router_z_loss = jnp.square(router_z_loss)            
            router_z_loss = self.router_z_loss_coef * router_z_loss.mean()
        aux_loss = aux_loss + router_z_loss

        # expert_index = nn.with_logical_constraint(expert_index, ("activation_batch", "activation_length",  None))
        # g * 2 * s
        expert_index = jnp.swapaxes(expert_index, 1, 2)
        # g * 2s
        expert_index = expert_index.reshape(num_groups, -1)
        # expert_index = nn.with_logical_constraint(expert_index, ("activation_batch", "activation_length"))

        # g * 2s * e, expert_index 负值的地方忽略了?
        expert_mask = jax.nn.one_hot(expert_index, self.num_experts, dtype=jnp.int32)
        # expert_mask = nn.with_logical_constraint(expert_mask, ("activation_batch", "activation_length", "exp"))
        # g * 2s * e 
        token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
        # g * 2 * s * e
        token_priority = token_priority.reshape(num_groups, topn, -1, self.num_experts)
        # g * s * 2 * e   ls: 每个token选择了2个专家，专家对应的位置的值表示当前编号专家选择的token数量
        token_priority = jnp.swapaxes(token_priority, 1, 2)

        '''token_priority
        lsp: 每个专家选择的token对应在原始token的位置索引, 类似于
          # e=4的例子
          Array([[[-1.,  0.,  1., -1.],
                  [ 0., -1., -1.,  1.],
                  [-1.,  1., -1.,  2.],
                  [-1.,  2.,  2., -1.],
                  [ 2., -1.,  0., -1.],
                  [ 3., -1., -1.,  0.],
                  [ 1.,  3., -1., -1.],
                  [-1., -1., -1., -1.]]], dtype=float32
                  )
          也可以这么理解，每个token选择了2个专家，被选中的专家的位置处是对应的token索引。
                  '''
        # 在topn那一维度选择max：就是提取当前专家选择了当前token的数量，因为1个专家只能被一个token选择一次，
        # 因此topn这一维度肯定只有一个是正数，这样原来，得到的矩阵就是：如果当前专家选择了当前token，这个token被选中了多少次，如果没有选择当前专家，那么就是一个负数
        # 此外，e这个维度，肯定只有topn个正数。如果取 1 * 1 * 1那么这个值不一定正数, 意味着没选中这个专家
        token_priority = jnp.max(token_priority, axis=2) 
        # g * s *  e
        # token_priority = nn.with_logical_constraint(token_priority, ("activation_batch", "activation_length", "exp"))
    
        if self.expert_chunk_size is None:
            compute_n_expert = self.num_experts
        else:
            compute_n_expert = self.num_experts // self.expert_chunk_size
            assert self.num_experts % self.expert_chunk_size == 0

        combined_outputs = None
        max_logging.log(f'compute_n_expert: {compute_n_expert}')
        for expert_index in range(0, token_priority.shape[2], compute_n_expert):
            # max_logging.log(f'expert_index: {expert_index}')
            _token_priority = token_priority[..., expert_index: expert_index+compute_n_expert]
            _router_probs = router_probs[..., expert_index: expert_index+compute_n_expert]
            # lsp： g * s * e * c  # 如果当前token选择了当前专家后，当前token被选中的总次数的one hot体现
            _dispatch_mask = jax.nn.one_hot(_token_priority, expert_capacity, dtype=jnp.bool_)
            # _dispatch_mask = nn.with_logical_constraint(_dispatch_mask, ("activation_batch", "activation_length", "exp", None))

            # 把token选择专家的概率赋值到one_hot矩阵上
            _combine_array = jnp.einsum('...se,...sec->...sec', _router_probs, _dispatch_mask)
            _combine_array = jax.lax.convert_element_type(_combine_array, self.dtype)
            # _combine_array = nn.with_logical_constraint(_combine_array, ("activation_batch", "activation_length", "exp", None))

            # 专家的输入mask：gsm x gsec -> gecm，  _dispatch_mask可以将多出容量之外的toke进行丢弃
            _expert_inputs = jnp.einsum('gs...,gsec->gec...', token_inputs, _dispatch_mask)
            _expert_inputs = jax.lax.convert_element_type(_expert_inputs, self.dtype)
            # gecm
            # max_logging.log(f'_expert_inputs: {_expert_inputs.shape}')
            # g * e * c * m
            _expert_outputs = self._call_experts(_expert_inputs, expert_index, compute_n_expert, deterministic=deterministic)
            # _expert_outputs = nn.with_logical_constraint(_expert_outputs, ("activation_batch", "exp", "activation_length", None))

            _combined_outputs = jnp.einsum('gec...,gsec->gs...', _expert_outputs, _combine_array)

            combined_outputs = _combined_outputs if combined_outputs is None else combined_outputs + _combined_outputs
            # max_logging.log(f'combined_outputs-{expert_index}: {combined_outputs}')

        # Return to batched shape.
        combined_outputs = combined_outputs.reshape(*inputs.shape)
        return combined_outputs, aux_loss
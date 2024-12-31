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
import max_utils

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
    # max_logging.log(f'name: {self.name} kernel_in_axis: {kernel_in_axis} kernel_out_axis: {kernel_out_axis} kernel: {kernel.shape} inputs: {inputs.shape}')

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


def record_gate(self, key, gate_scores, axis=(0, 1)):
    expert_to_token_score = gate_scores.mean(axis=axis)
    sum_value = jnp.sum(expert_to_token_score, axis=-1)
    expert_to_token_score = expert_to_token_score / (sum_value + 1e-6)
    self.sow('intermediates', f'{key}/expert_to_token_score', _entroy(expert_to_token_score)) # 熵越大越好 max: 5.45
    self.sow('intermediates', f'{key}/token_to_expert_score', _entroy(gate_scores)) # 熵越小越好


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
    # if self.config.record_internal_nn_metrics:
    #   record_gate(self, 'mgate', gate_scores)
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
    max_logging.log(f'self.intermediate_dim: {self.intermediate_dim}')
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
    max_logging.log(f'inputs.shape: {inputs.shape} xshape: {x.shape}')

    if gate_scores is not None:
      assert isinstance(cfg.mgate_dim, int)
      max_logging.log(f'gate_scores is not None and mgate_dim={cfg.mgate_dim}')
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


def log(t, eps = 1e-20):
    return jnp.log(t.clip(min = eps))
    

def gumbel_noise(inputs, seed=9876, minval=0, maxval=1):
    noise = jax.random.uniform(jax.random.PRNGKey(seed), minval=minval, maxval=maxval, shape=inputs.shape,  dtype=inputs.dtype)
    return -log(-log(noise))


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
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  mesh: Mesh
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
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

    if self.config.mgate:
      kernel_axes = (None, 'embed', 'mlp')
      inner_gate = self.param(
        'mgate',
        nn.with_logical_partitioning(kernel_init, kernel_axes),
        (self.num_experts, emb_dim, self.config.mgate_dim),
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    else:
      inner_gate = None

    return w0_kernel, w1_kernel, wo_kernel

  def permute(self, inputs, gate_logits):
    """Permute tokens to group by expert to fit gmm call."""

    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    inputs_2d = jnp.reshape(inputs, (inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    weights, selected_experts = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(self.dtype)

    flatten_selected_experts = jnp.ravel(selected_experts) # 按行展开, 默认 order='C' order='F'的话是按列展开
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    max_logging.log(f'sorted_selected_experts: {sorted_selected_experts.shape}')
    sorted_indices = sorted_selected_experts // self.num_experts_per_tok
    # sort inputs for number of selected experts lsp: 将每个token的向量根据选择专家的索引从0~end排序
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)
    # group_size记录了每个专家选择的token数量
    group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size, selected_experts

  def unpermute(self, intermediate, sorted_selected_experts, weights):
    """Unpermute tokens to original order and combine weights."""

    unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
    reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
    tensor_parallelism = self.config.ici_tensor_parallelism * self.config.dcn_tensor_parallelism
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate, (-1, self.num_experts_per_tok, self.config.emb_dim // tensor_parallelism)
    )
    with jax.named_scope("weight_sum"):
      matmul_precision = lax.Precision(self.config.matmul_precision)
      output = jnp.einsum(
          "BKE,BK -> BE",
          reshaped_intermediate.astype(jnp.float32),
          reshaped_weights.astype(jnp.float32),
          precision=matmul_precision,
      )
    return output.reshape(-1, (self.config.max_target_length - 1) // self.config.megablox_chunks, self.config.emb_dim // tensor_parallelism).astype(self.dtype)

  # def fake_make(self, expert_inputs, hidden, inner_gate):
  #   if inner_gate is not None:
  #     assert isinstance(self.config.mgate_dim, int)
  #     # x = jnp.einsum('BTE,BTEM->BTEM', gate_scores, x)  # 这里是多个专家一起计算mgate分数
  #     mgate_scores = jnp.einsum('gecm,emi->geci', expert_inputs, inner_gate)
  #     max_logging.log(f'megablox mgate is True  mgate_scores: {mgate_scores.shape}')
  #     mgate_scores = jax.nn.softmax(mgate_scores.astype(jnp.float32), axis=-1)
  #     mgate_scores = mgate_scores.astype(self.dtype)
  #     if self.config.record_internal_nn_metrics:
  #       record_gate(self, 'mgate', mgate_scores, axis=(0, 1, 2))
  #     G, E, C, H = hidden.shape
  #     x = hidden.reshape(G, E, C, self.config.mgate_dim, H // self.config.mgate_dim)
  #     x = jnp.einsum('geci,gecif->gecif', mgate_scores, x)
  #     hidden = x.reshape(G, E, C, H)
  #   return hidden

  def megablox(self, inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel):
    tile_size = (512, 1024, 1024) # 矩阵的小方块大小

    def gmm(inputs, kernel, group_sizes):
      hs_shape = inputs.shape
      # pad length is the 1st dimension of tiling size in gmm call
      pad_length = 512
      if hs_shape[0] % pad_length:
        pad_length = pad_length - hs_shape[0] % pad_length
        inputs = jax.lax.pad(inputs.astype(jnp.float32), 0.0, [(0, pad_length, 0), (0, 0, 0)])

      inputs = inputs.astype(self.dtype)
      kernel = kernel.astype(self.dtype)
      # 180224(44 * 4096) * 4096， 这里面进行chunk编译太慢了，且容易出问题
      # chunks = 4
      # l1 = kernel.shape[0]
      # c1 = l1 // chunks
      # l2 = group_sizes.shape[0]
      # c2 = l2 // chunks
      # output = []
      # for i in range(chunks):
      #   _kernel = kernel[i * c1: (i+1) * c1]
      #   _group_sizes = group_sizes[i * c2: (i+1) * c2]
      #   s0 = group_sizes[: i * c2].sum()
      #   s1 = group_sizes[: (i+1) * c2].sum()
      #   _inputs = inputs[s0: s1]
      #   _output = mblx.gmm(
      #       lhs=_inputs, rhs=_kernel, group_sizes=_group_sizes, preferred_element_type=jnp.bfloat16, tiling=tile_size
      #   )
      #   max_logging.log(f'_output: {_output.shape}')
      #   output.append(_output)
      # output = jnp.concatenate(output, axis=0)
      output = mblx.gmm(
            lhs=inputs, rhs=kernel, group_sizes=group_sizes, preferred_element_type=jnp.bfloat16, tiling=tile_size
        )
      max_logging.log(f'output11: {output.shape}')
      if hs_shape[0] % pad_length:
        output = output[: hs_shape[0]]
      return output

    # Currently, we only support data and tensor parallelism with Megablox.
    # We all gather the input activations over tensor parallelism to follow strategy
    # in https://parsa.epfl.ch/course-info/cs723/papers/Megatron.pdf.
    @functools.partial(
        shard_map.shard_map,
        mesh=self.mesh,
        in_specs=(
            (nn.logical_to_mesh_axes(("activation_batch", None, None))),
            (nn.logical_to_mesh_axes(("activation_batch", None, None))),
            (nn.logical_to_mesh_axes((None, None, "mlp"))),
            (nn.logical_to_mesh_axes((None, None, "mlp"))),
            (nn.logical_to_mesh_axes((None, "mlp", None))),
        ),
        out_specs=(nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed")), 
                  nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed")),
                  nn.logical_to_mesh_axes(("activation_batch", None, "activation_embed")),
                  ),
        check_rep=False,
    )
    def wrapper(inp, logits, w0, w1, wo):
      x, sorted_selected_experts, weights, group_sizes, selected_experts = self.permute(inp, logits)
      layer_w0 = gmm(x, w0, group_sizes)
      layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      layer_w1 = gmm(x, w1, group_sizes)
      # layer_w1 = checkpoint_name(layer_w0, "mlpwi_1")
      layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      intermediate_layer = jnp.multiply(layer_act, layer_w1) # 
      # inp: (4, 4096, 4096) intermediate_layer: (32768, 5632) inner_gate: (8, 4096, 44)
      # inp: (4, 4096, 4096) intermediate_layer: (180224, 1024) group_sizes: (44,)
      max_logging.log(f'inp: {inp.shape} intermediate_layer: {intermediate_layer.shape} group_sizes: {group_sizes.shape}')
      # x: (180224, 4096) w0: (44, 4096, 1024) w1: (44, 4096, 1024) wo: (44, 1024, 4096)
      max_logging.log(f'x: {x.shape} w0: {w0.shape} w1: {w1.shape} wo: {wo.shape}')
      intermediate_output = gmm(intermediate_layer, wo, group_sizes)
      intermediate_output = checkpoint_name(intermediate_output, "mlpwo")
      tensor_parallelism = self.config.ici_tensor_parallelism * self.config.dcn_tensor_parallelism
      if tensor_parallelism > 1:
        intermediate_output = jax.lax.psum_scatter(intermediate_output, "tensor", scatter_dimension=1, tiled=True)
      # intermediate_output: (180224, 4096)
      output = self.unpermute(intermediate_output, sorted_selected_experts, weights)
      return output, selected_experts, weights

    return wrapper(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)

  def reshape_and_update_weights(self, weights, indices):
    # input of weights & indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (jnp.arange(weights.shape[0])[:, None, None], jnp.arange(weights.shape[1])[:, None], indices)
    update_weights = update_weights.at[index_update].set(weights)
    return update_weights

  def generate_masks(self, top_k_indices, softmax_probs):
    # calculate expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    tokens_per_batch = seq_len * self.num_experts_per_tok
    expert_capacity_per_batch = int((tokens_per_batch / self.num_experts) * self.config.expert_capacity_factor)
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
        expert_token_count_fused, ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts))
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
        expert_token_position_fused, (batch_size, seq_len, self.num_experts_per_tok, self.num_experts)
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position, num_classes=expert_capacity_per_batch + 1, dtype=jnp.int32
    )

    # shape of combine_mask is (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)
    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits):
    max_logging.log(f'aux_loss_coef: {self.config.aux_loss_coef}')
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.aux_loss_coef
    return loss

  def get_einsum(self, rhs_mesh_axes: Tuple[Optional[str], ...] = ()):
    if self.quant:

      def aqt_einsum(*args, **kwargs):
        # simply skip kwargs, since aqt einsum doesn't support any kwargs like precision
        return self.quant.einsum(rhs_mesh_axes)(*args)

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
    gate_logits = nn.with_logical_constraint(gate_logits, ("activation_batch", "activation_length", "activation_embed"))
    softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
    # shape of top_k_weights & top_k_indices: (batch, sequence, num_experts_per_tok)
    top_k_weights, top_k_indices = jax.lax.top_k(softmax_probs, self.num_experts_per_tok)
    matmul_precision = lax.Precision(self.config.matmul_precision)

    if self.config.expert_capacity_factor > 0:
      max_logging.log(f'expert_capacity_factor: {self.config.expert_capacity_factor}')
      # token dropping if needed
      dispatch_mask, combine_mask = self.generate_masks(top_k_indices, softmax_probs)
      mask_axes = ("activation_batch", "activation_length", None, None)
      dispatch_mask = nn.with_logical_constraint(dispatch_mask, mask_axes)
      combine_mask = nn.with_logical_constraint(combine_mask, mask_axes)
      loss = self.load_balance_loss(top_k_indices, softmax_probs)
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("dispatch"):
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes)(
            "BSM,BSEC -> EBCM", inputs, dispatch_mask, precision=matmul_precision
        )
        dispatch = nn.with_logical_constraint(
            dispatch, ("activation_exp", "activation_batch_no_exp", None, "activation_embed")
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
            layer_w0, ("activation_exp", "activation_batch_no_exp", None, "activation_mlp")
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
            layer_w1, ("activation_exp", "activation_batch_no_exp", None, "activation_mlp")
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
            intermediate_layer, ("activation_exp", "activation_batch_no_exp", None, "activation_embed")
        )
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes)(
            "EBCM,BSEC -> BSM", intermediate_layer, combine_mask, precision=matmul_precision
        )
      return output, loss
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        ).astype(jnp.float32)
        layer_w0 = checkpoint_name(layer_w0, "mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        ).astype(jnp.float32)
        layer_w1 = checkpoint_name(layer_w1, "mlpwi_1")
      layer_w0_act = _convert_to_activation_function(self.config.mlp_activations[0])(layer_w0)
      layer_multiply = jnp.multiply(layer_w0_act, layer_w1).astype(self.dtype)
      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM", layer_multiply, wo_kernel, precision=matmul_precision
        )
        intermediate_layer = checkpoint_name(intermediate_layer, "mlpwo")
      with jax.named_scope("w_sum"):
        output = jnp.einsum("BSEM,BSE -> BSM", intermediate_layer.astype(jnp.float32), weights.astype(jnp.float32)).astype(
            self.dtype
        )
      return output, None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_logits = DenseGeneral(
        self.num_experts,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        name="router_gate",
        matmul_precision=self.config.matmul_precision,
    )(inputs)

    if self.config.gate_noise_coef > 0.0:
        max_logging.log(f'megablox gate_noise_coef: {self.config.gate_noise_coef}')
        noise = gumbel_noise(gate_logits, seed=self.config.init_weights_seed)
        gate_logits += noise * self.config.gate_noise_coef

    if self.config.record_internal_nn_metrics:
      l2norm = jnp.sqrt(jnp.sum(jnp.square(gate_logits)))
      self.sow('intermediates', 'router_gate/l2norm', l2norm)
      # router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
      # record_gate(self, 'router_gate', router_probs)
    w0_kernel, w1_kernel, wo_kernel = self.generate_kernels(cfg.num_experts, cfg.emb_dim, cfg.mlp_dim)

    if cfg.megablox:
      max_logging.log("Running MoE megablox implementation.")
      # inputs: b * l * d _gate_logits: ble, _output: bld  _selected_experts: bl(top)  _weights: bl(top) w0_kernel, w1_kernel: edf    wo_kernel: efd
      output, selected_experts, weights = [], [], []
      chunks = cfg.megablox_chunks
      chunk_size = inputs.shape[1] // chunks
      for inx in range(chunks):
        _inputs = inputs[:, inx * chunk_size: (inx + 1) * chunk_size]
        _gate_logits = gate_logits[:, inx * chunk_size: (inx + 1) * chunk_size]
        _output, _selected_experts, _weights = self.megablox(_inputs, _gate_logits, w0_kernel, w1_kernel, wo_kernel)
        output.append(_output)
        selected_experts.append(_selected_experts)
        weights.append(_weights)
      output = jnp.concatenate(output, axis=1)
      selected_experts = jnp.concatenate(selected_experts, axis=1)
      weights = jnp.concatenate(weights, axis=1)

      # if self.config.record_internal_nn_metrics: # lsp
      #   max_logging.log(f'router weights: {weights.shape} selected_experts: {selected_experts.shape}')
      #   record_gate(self, 'router_gate', weights)
      #   expert_index_record = selected_experts.reshape(-1, self.num_experts_per_tok)
      #   for i in range(0, self.config.num_experts, 4): # 只记录1/4的专家
      #     top = 0
      #     for j in range(2): # 只取top2记录
      #       _top = (expert_index_record[:, j] == i).sum()
      #       top += _top
      #       self.sow('intermediates', f'top{j}/selected_expert_{i}_token_nums', _top)
      #     self.sow('intermediates', f'top/selected_expert_{i}_token_nums', top)
      return output, None
    else:
      max_logging.log("Running MoE matmul implementation.")
      return self.dense_matmul(inputs, gate_logits, w0_kernel, w1_kernel, wo_kernel)


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
    dtype: DType = jnp.bfloat16
    num_experts: int = 0
    intermediate_dim: int = 4096
    intermediate_dropout_rate: float = 0.1

    def setup(self):

        kernel_in_axis = np.arange(1)
        kernel_out_axis = np.arange(1, 2)
        kernel_init = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
        # self.kernel_init = kernel_init
        # The first axes is expert
        kernel_axes = ("exp", "embed_no_exp", "mlp")
        wo_kernel_axes = ("exp", "mlp", "embed_no_exp")
  
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

        if self.config.mgate:
          inner_gate_kernel = self.param(
            'mgate',
            nn.with_logical_partitioning(kernel_init, kernel_axes),
            (self.num_experts, emb_dim, self.config.mgate_dim),
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
          )
          self.inner_gate = jnp.asarray(inner_gate_kernel, self.dtype)
          self.router_name = "router_gate"
        else:
          self.inner_gate = None
          self.router_name = "router_gate"
           
    @nn.compact
    def __call__(self, inputs, paddings, deterministic=False):
        inputs = inputs.astype(self.dtype)
        combined_outputs, aux_loss = self._dispatch_and_combine_expert_outputs_openmoe(inputs, paddings, deterministic=deterministic)
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
        max_logging.log(f'self.intermediate_dropout_rate: {self.intermediate_dropout_rate} deterministic: {deterministic}')
        hidden = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(hidden, deterministic=deterministic) 
        # expert_inputs: gecm,  mgatew: meh  -> 
        if self.config.mgate:
          assert isinstance(self.config.mgate_dim, int)

          inner_gate = self.inner_gate[expert_index: expert_index + compute_n_expert]
          # x = jnp.einsum('BTE,BTEM->BTEM', gate_scores, x)  # 这里是多个专家一起计算mgate分数
          mgate_scores = jnp.einsum('gecm,emi->geci', expert_inputs, inner_gate)
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
          hidden = hidden.reshape(G, E, C, H)
          # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", "tensor"))

        hidden = jnp.einsum("gech,ehm->gecm", hidden, theta_wo)
        # hidden = nn.with_logical_constraint(hidden, ("activation_batch", "exp", "activation_length", "tensor"))
        
        return hidden
        
    def _dispatch_and_combine_expert_outputs_openmoe(self, inputs, paddings, deterministic=False):

        max_logging.log(f'Enter openmoe top2 router.....')
        topn = self.num_experts_per_tok
        token_shape = inputs.shape[:-1]
        num_tokens = np.prod(token_shape)
        m_dim = inputs.shape[-1]
       
        num_groups = self.num_groups
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
                name=self.router_name)(token_inputs)

        if self.config.gate_noise_coef > 0.0:
          max_logging.log(f'gate_noise_coef: {self.config.gate_noise_coef}')
          noise = gumbel_noise(router_logits, seed=self.config.init_weights_seed)
          router_logits += noise * self.config.gate_noise_coef

        _, expert_index, one_hot_indices = _top_k(router_logits, k=topn)

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
          record_gate(self, 'sfm_after_topn', router_probs, axis=(0, 1))
          top_values = jnp.array([(expert_index == i).sum() for i in jnp.arange(0, self.num_experts, 1)])
          self.sow('intermediates', f'top/selected_expert_token_nums', top_values)
        
        # 有padding的时候放开, 一般预训练没有pad
        # if paddings is not None:
        #     max_logging.log(f'paddings: {paddings.shape}')
        #     max_logging.log(f'token_shape: {token_shape}')
            
        #     assert paddings.shape == token_shape
        #     # 如果paddings中的0表示保留，则 nonpaddings = 1.0 - paddings  
        #     nonpaddings = paddings
        #     nonpaddings = jnp.reshape(nonpaddings, grouped_inputs.shape[:2])
        #     gate_mask = jnp.expand_dims(nonpaddings, axis=-1)
        #     # expert_gate *= gate_mask
    
        #     expert_index *= (2 * gate_mask - 1.) # lsp:将被mask的专家的所以变为负值，这样在之后转为one hot形式的时候就不会考虑
        #     expert_index += jnp.repeat(gate_mask - 1., topn, axis=-1)
        #     router_probs *= gate_mask # ble

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

        self.add_aux_loss("aux_loss", aux_loss)
        # Return to batched shape.
        combined_outputs = combined_outputs.reshape(*inputs.shape)
        return combined_outputs, aux_loss

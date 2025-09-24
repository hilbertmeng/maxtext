import re
from einops import rearrange
from layers import initializers
from layers import normalizations
from layers import embeddings
from flax import linen as nn
import jax
import jax.numpy as jnp
import common_types


Config = common_types.Config
NdInitializer = initializers.NdInitializer
DType = common_types.DType

class DeepEmbedBlock(nn.Module):
  """Transformer Deep Embed Block."""
  config: Config
  kernel_init: NdInitializer = initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  input_dim: int = None
  output_dim: int = None
  de_d1_d2_dims: tuple = None # suggesgt fix first dimension to 32

  def setup(self):
    if 'gemma3n' in self.config.deep_embed_type:
      return
    self.d1, self.d2 = self.de_d1_d2_dims
    s1_axes = ("embed", None)
    s2_axes = (None, "embed")
    s2_bias_axes = (None, None)
    s1_kernel_init = nn.with_logical_partitioning(self.kernel_init, s1_axes)
    s2_kernel_init = nn.with_logical_partitioning(self.kernel_init, s2_axes)
    s2_bias_kernel_init = nn.with_logical_partitioning(self.kernel_init, s2_bias_axes)
    self.s1 = self.param('s1', s1_kernel_init, (self.input_dim, self.d1), self.weight_dtype)
    self.s2 = self.param('s2', s2_kernel_init, (self.d2, self.output_dim), self.weight_dtype)
    self.s2_bias = None
    if self.config.use_s2_bias:
      self.s2_bias = self.param('s2.bias', s2_bias_kernel_init, (1, self.output_dim), self.weight_dtype)
    print(f'[DEshape] s1: {self.s1.shape} s2: {self.s2.shape} s2_bias: {self.s2_bias} de_d1_d2_dims: {self.de_d1_d2_dims}')

  @nn.compact
  def __call__(self, inputs, output, decoder_input_tokens, deep_embedding=None):
    cfg = self.config
    if cfg.deep_embed_init == 'inside' and deep_embedding is None:
      deep_embedding = embeddings.Embed(
            name="token_embedder",
            num_embeddings=cfg.vocab_size,
            features=self.d1 * self.d2, # Don't need to follow mudd mlp dim.
            dtype=cfg.dtype,
            embedding_init=initializers.get_init_method(cfg.init_method),
            config=cfg,
          )(decoder_input_tokens.astype("int32"))
      deep_embedding = deep_embedding.reshape(*output.shape[:2], self.d1, self.d2)
    print(f'inputs: {inputs.shape} output: {output.shape} deep_embedding: {deep_embedding.shape}')
    print(f'DeepEmbedBlock deep_embed_type: {self.config.deep_embed_type}')

    # btD x Dd -> btd -> bt1d
    deep_w = jnp.expand_dims(inputs @ self.s1, axis=2)
    
    # bt1d @ btdd -> bt1d @ dD -> bt1D + bt1D -> bt1D
    deep_w = deep_w @ deep_embedding
    if cfg.de_gate == 'tanh':
      deep_w = jax.nn.tanh(deep_w)
    elif cfg.de_gate == 'sigmoid':
      deep_w = jax.nn.sigmoid(deep_w)
    elif cfg.de_gate == 'relu':
      deep_w = jax.nn.relu(deep_w)
    elif cfg.de_gate == 'gelu':
      deep_w = jax.nn.gelu(deep_w)
    elif cfg.de_gate == 'silu':
      deep_w = jax.nn.silu(deep_w)

    if self.s2_bias is not None:
      deep_w = (deep_w @ self.s2 + self.s2_bias).reshape(*output.shape)
    else:
      deep_w = (deep_w @ self.s2).reshape(*output.shape)

      # if cfg.de_gate == 'tanh':
      #   deep_w = jax.nn.tanh(deep_w)
      # elif cfg.de_gate == 'sigmoid':
      #   deep_w = jax.nn.sigmoid(deep_w)
      # elif cfg.de_gate == 'relu':
      #   deep_w = jax.nn.relu(deep_w)
      # elif cfg.de_gate == 'gelu':
      #   deep_w = jax.nn.gelu(deep_w)
      # elif cfg.de_gate == 'silu':
      #   deep_w = jax.nn.silu(deep_w)

    output *= deep_w

    if cfg.deep_embed_norm or cfg.mlp_post_norm:
      output = normalizations.RMSNorm(
          name="norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm", ),
          epsilon=cfg.normalization_layer_epsilon,
      )(output) # suggest to use post_norm

    return output

def get_deep_embedding(cfg, de):
  deep_embeddings = {}
  print(f'de: {de.shape}')
  if cfg.deep_embed_init == 'outside':
    for de_type in ['devalue', 'deattnout', '1xmlp', '4xmlp']:
      if de_type in cfg.deep_embed_type:
        offset_dim = cfg.mlp_dim if de_type == '4xmlp' else cfg.emb_dim
        print(f'de_type: {de_type} offset_dim: {offset_dim}')
        offset_dim *= cfg.num_decoder_layers
        deep_embeddings[de_type] = rearrange(
            de[..., :offset_dim], 'B T (L d e) -> L B T d e',
            L=cfg.num_decoder_layers, 
            d=32 if cfg.emb_dim < 4096 else 64
            )
        de = de[..., offset_dim: ]
        print(f'{de_type}: {deep_embeddings[de_type].shape}')
 
  return deep_embeddings

def compute_embed_dim(cfg):
  emb_dim = 0
  if cfg.deep_embed_init == 'outside':
    if 'devalue' in cfg.deep_embed_type:
      emb_dim += cfg.num_decoder_layers * cfg.emb_dim
    if 'deattnout' in cfg.deep_embed_type:
      emb_dim += cfg.num_decoder_layers * cfg.emb_dim
    if '1xmlp' in cfg.deep_embed_type:
      emb_dim  += cfg.num_decoder_layers * cfg.emb_dim
    if '4xmlp' in cfg.deep_embed_type:
      emb_dim +=  cfg.num_decoder_layers * cfg.mlp_dim
  return emb_dim
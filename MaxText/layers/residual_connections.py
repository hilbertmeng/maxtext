"""Full Attention Residuals and Manifold-Constrained Hyper-Connections.

References and initialization choices: experiments/attnres_mhc_review.md.
Routing math is FP32; branch reads and persistent states use model dtype.
"""
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp


def sinkhorn(logits, iterations, epsilon):
  """DeepSeek TileKernels sinkhorn_normalize_ref, including epsilon order."""
  matrix = jax.nn.softmax(logits.astype(jnp.float32), axis=-1) + epsilon
  matrix /= matrix.sum(axis=-2, keepdims=True) + epsilon
  for _ in range(iterations - 1):
    matrix /= matrix.sum(axis=-1, keepdims=True) + epsilon
    matrix /= matrix.sum(axis=-2, keepdims=True) + epsilon
  return matrix


def mhc_update(state, branch, post, mixing):
  """mixing[..., source, destination], as in official mhc_post_ref."""
  residual = jnp.einsum('...ij,...id->...jd', mixing, state.astype(jnp.float32))
  return (residual + post[..., :, None] * branch.astype(jnp.float32)[..., None, :]).astype(state.dtype)


class FullAttentionResidual(nn.Module):
  config: Any

  @nn.compact
  def __call__(self, sources):
    if len(sources) == 1:
      return sources[0]  # Softmax over one source is identically one.
    cfg = self.config
    width = sources[0].shape[-1]
    query = self.param('query', nn.initializers.zeros, (width,), cfg.weight_dtype)
    scale = self.param('key_scale', nn.initializers.ones, (width,), cfg.weight_dtype)
    # Keep the large history as separate buffers. Stacking [B,T,S,D] at every
    # read made XLA retain many growing, multi-GiB history copies under remat.
    # Only the scalar depth scores need a new source axis.
    scores = []
    for source in sources:
      value = source.astype(jnp.float32)
      key = value * jax.lax.rsqrt(jnp.mean(value**2, axis=-1, keepdims=True) + cfg.attnres_norm_epsilon)
      key *= scale.astype(jnp.float32)
      scores.append(jnp.einsum('...d,d->...', key, query.astype(jnp.float32)))
    scores = jnp.stack(scores, axis=-1)
    weights = jax.nn.softmax(scores, axis=-1)  # Depth axis; no sqrt(D).
    result = jnp.zeros_like(sources[0], dtype=jnp.float32)
    for index, source in enumerate(sources):
      result += weights[..., index, None] * source.astype(jnp.float32)
    return result.astype(sources[0].dtype)


class ManifoldHyperConnection(nn.Module):
  config: Any
  layer_index: int = 0  # Connector index: ATTN=2*l, MLP=2*l+1.

  @nn.compact
  def __call__(self, state):
    cfg = self.config
    n, d = state.shape[-2:]
    if n != cfg.mhc_num_streams:
      raise ValueError('mHC state has an incorrect stream count')
    outputs = n * n + 2 * n
    flat = state.astype(jnp.float32).reshape(*state.shape[:-2], n * d)
    normalized = flat * jax.lax.rsqrt(jnp.mean(flat**2, axis=-1, keepdims=True) + cfg.mhc_norm_epsilon)
    if cfg.mhc_norm_affine:
      scale = self.param('norm_scale', nn.with_logical_partitioning(nn.initializers.ones, ('embed',)),
                         (n * d,), cfg.weight_dtype)
      normalized *= scale.astype(jnp.float32)
    kernel_init = (nn.initializers.zeros if cfg.mhc_router_init_std == 0
                   else nn.initializers.normal(cfg.mhc_router_init_std))
    # The flattened stream/embedding axis is still a model-weight axis.
    # Without this annotation every FSDP rank replicates the full controller.
    phi = self.param('router_kernel', nn.with_logical_partitioning(kernel_init, ('embed', None)),
                     (n * d, outputs), cfg.weight_dtype)
    alpha = self.param('router_scale', nn.initializers.constant(cfg.mhc_alpha_init), (3,), cfg.weight_dtype)

    def bias_init(_key, shape, dtype):
      pre = jnp.full((n,), cfg.mhc_pre_bias_init).at[self.layer_index % n].set(cfg.mhc_pre_selected_bias_init)
      residual = (jnp.full((n, n), cfg.mhc_residual_offdiag_bias)
                  + jnp.eye(n) * (cfg.mhc_residual_identity_bias - cfg.mhc_residual_offdiag_bias))
      return jnp.concatenate((
          pre,
          jnp.full((n,), cfg.mhc_post_bias_init),
          residual.reshape(-1))).astype(dtype)

    bias = self.param('router_bias', bias_init, (outputs,), cfg.weight_dtype)
    alphas = jnp.concatenate((jnp.repeat(alpha[0], n), jnp.repeat(alpha[1], n), jnp.repeat(alpha[2], n*n)))
    logits = (normalized @ phi.astype(jnp.float32)) * alphas.astype(jnp.float32) + bias.astype(jnp.float32)
    pre = jax.nn.sigmoid(logits[..., :n]) + cfg.mhc_pre_epsilon
    post = 2 * jax.nn.sigmoid(logits[..., n:2*n])
    mixing = sinkhorn(logits[..., 2*n:].reshape(*state.shape[:-2], n, n),
                     cfg.mhc_sinkhorn_iters, cfg.mhc_sinkhorn_epsilon)
    read = jnp.sum(pre[..., :, None] * state.astype(jnp.float32), axis=-2).astype(state.dtype)
    return read, post, mixing


class MHCLearnedStreamReduce(nn.Module):
  """V4-style independent sigmoid gates before the existing final RMSNorm.

  Matches depth-memory's learned-head equations and initialization, without
  reproducing the CUDA kernels' TF32 rounding. This is separate from mHC A.6.
  """
  config: Any

  @nn.compact
  def __call__(self, state):
    cfg = self.config
    n, d = state.shape[-2:]
    if n != cfg.mhc_num_streams:
      raise ValueError('mHC head state has an incorrect stream count')
    flat = state.astype(jnp.float32).reshape(*state.shape[:-2], n * d)
    scale = self.param('norm_scale', nn.with_logical_partitioning(nn.initializers.ones, ('embed',)),
                         (n * d,), cfg.weight_dtype)
    kernel = self.param('head_kernel', nn.with_logical_partitioning(
        nn.initializers.normal(cfg.mhc_head_init_std), ('embed', None)),
                        (n * d, n), cfg.weight_dtype)
    alpha = self.param('head_scale', nn.initializers.constant(cfg.mhc_head_alpha_init), (1,), cfg.weight_dtype)
    bias = self.param('head_bias', nn.initializers.constant(cfg.mhc_head_bias_init), (n,), cfg.weight_dtype)
    normalized = flat * jax.lax.rsqrt(jnp.mean(flat**2, axis=-1, keepdims=True) + cfg.mhc_head_norm_epsilon)
    logits = (normalized * scale.astype(jnp.float32)) @ kernel.astype(jnp.float32)
    gates = jax.nn.sigmoid(logits * alpha.astype(jnp.float32) + bias.astype(jnp.float32)) + cfg.mhc_pre_epsilon
    return jnp.sum(state.astype(jnp.float32) * gates[..., :, None], axis=-2).astype(state.dtype)


class ResidualDecoderStack(nn.Module):
  config: Any
  mesh: Any
  quant: Any = None

  @nn.compact
  def __call__(self, embedding, segment_ids, positions, tokens, deterministic, model_mode):
    # Import here to share all existing Llama ATTN/MLP machinery without a
    # models -> fusion -> models module initialization cycle.
    from layers.fusion import SubDecoderLayer
    from layers.models import get_remat_policy

    cfg = self.config
    Layer = SubDecoderLayer
    if cfg.remat_policy != 'none':
      Layer = nn.remat(Layer, prevent_cse=cfg.remat_prevent_cse,
                       policy=get_remat_policy(cfg), static_argnums=(6, 7))
    if cfg.residual_connection == 'attnres':
      state = (embedding,)
    else:
      state = jnp.broadcast_to(embedding[..., None, :], (*embedding.shape[:-1], cfg.mhc_num_streams, embedding.shape[-1]))
    windows = cfg.sliding_window_size
    windows = windows if isinstance(windows, list) else [windows]
    for index in range(cfg.num_decoder_layers):
      window = windows[index % len(windows)]
      window = cfg.max_target_length if window is None else window
      state, _ = Layer(cfg, self.mesh, self.quant, window, index, name=f'layers_{index}')(
          state, segment_ids, positions, tokens, None, deterministic, model_mode, None)
    if cfg.residual_connection == 'attnres':
      return FullAttentionResidual(cfg, name='output_connection')(state)
    if cfg.mhc_learned_head:
      return MHCLearnedStreamReduce(cfg, name='output_connection')(state)
    # Explicit mean-adapter ablation. The normal output RMSNorm still follows.
    return jnp.mean(state.astype(jnp.float32), axis=-2).astype(embedding.dtype)

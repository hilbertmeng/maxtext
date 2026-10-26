"""Post-softmax diagonal removal on the pinned JAX0.8.1 Splash implementation.

Keep its causal softmax denominator: Y_off = Y - p_ii V_i. In reverse, use
Y_off in Splash's row delta and remove the remaining diagonal Q/K/V terms.
This uses one normal Splash forward/backward, without an explicit T x T alpha.
"""
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

NEGATIVE_DIAGONAL_LOGIT = -1e9


def make_negative_diagonal_splash(kernel):
  """Use a strict-causal Splash mask and return zero for empty history rows.

  The caller constructs its mask with offset=-1, suppressing the diagonal in
  logits BEFORE softmax. A fixed, zero-valued sink at -1e9 stabilizes empty
  rows; its probability underflows to zero on nonempty rows. No parameter is
  introduced, and no post-softmax renormalization/subtraction is used here.
  """
  def apply(q, k, v, segment_ids=None):
    return kernel(q, k, v, segment_ids=segment_ids,
                  sinks=jnp.full((q.shape[0],), NEGATIVE_DIAGONAL_LOGIT, jnp.float32))
  return apply


def _collapse(info):
  # Match SplashAttentionKernel.__call__'s JAX0.8.1 mask-info adapter.
  if info is None or info.partial_mask_blocks is None:
    return info
  return info._replace(partial_mask_blocks=info.partial_mask_blocks.reshape(
      -1, *info.partial_mask_blocks.shape[-2:]))


def make_zero_diagonal_splash(kernel):
  """Wrap a causal, equal-head MHA kernel whose mask includes its diagonal."""
  options = dict(kernel.kwargs)
  if options['is_mqa'] or options['save_residuals']:
    raise ValueError('Diagonal-zero adapter requires equal-head MHA without auxiliary outputs')
  masks = tuple(_collapse(info) for info in (
      kernel.fwd_mask_info, kernel.dq_mask_info, kernel.dkv_mask_info))
  cap = options['attn_logits_soft_cap']

  def forward(q, k, v, segment_ids, mask_infos):
    if q.shape != k.shape or q.shape[:2] != v.shape[:2]:
      raise ValueError('Diagonal-zero attention requires aligned Q/K/V token and head axes')
    out, residual = splash._splash_attention_fwd(
        *mask_infos, q, k, v, segment_ids, None, **options)
    raw_diagonal = jnp.sum(q.astype(jnp.float32) * k.astype(jnp.float32), axis=-1)
    diagonal = (jnp.tanh(raw_diagonal / cap) * cap if cap else raw_diagonal)
    probability = jnp.exp(diagonal - residual[6].astype(jnp.float32))
    off = (out.astype(jnp.float32) - probability[..., None] * v.astype(jnp.float32)).astype(out.dtype)
    # Enforce the requested exact zero at each packed segment's first token,
    # including BF16 roundoff in Y - p_ii*V_i. Segment IDs are contiguous runs.
    has_history = jnp.arange(q.shape[-2]) > 0
    if segment_ids is not None:
      has_history &= segment_ids.q == jnp.roll(segment_ids.kv, 1)
    off = jnp.where(has_history[None, :, None], off, 0)
    # Splash's D_i=<dY_i,Y_i> must use the modified output, preserving the
    # diagonal logit's contribution to the ORIGINAL softmax denominator.
    residual = residual[:5] + (off,) + residual[6:]
    return off, (residual, probability, has_history)

  @jax.custom_vjp
  def apply(q, k, v, segment_ids, mask_infos):
    return forward(q, k, v, segment_ids, mask_infos)[0]

  def backward(saved, dout):
    residual, probability, has_history = saved
    dout = jnp.where(has_history[None, :, None], dout, 0)
    q, k, v = residual[:3]
    gradients = splash._splash_attention_bwd(res=residual, do=dout, **options)
    dq, dk, dv = gradients[3:6]
    upstream = dout.astype(jnp.float32)
    correction = probability * jnp.sum(upstream * v.astype(jnp.float32), axis=-1)
    if cap:
      raw = jnp.sum(q.astype(jnp.float32) * k.astype(jnp.float32), axis=-1)
      correction *= 1 - jnp.square(jnp.tanh(raw / cap))
    dq = (dq.astype(jnp.float32) - correction[..., None] * k.astype(jnp.float32)).astype(q.dtype)
    dk = (dk.astype(jnp.float32) - correction[..., None] * q.astype(jnp.float32)).astype(k.dtype)
    dv = (dv.astype(jnp.float32) - probability[..., None] * upstream).astype(v.dtype)
    return dq, dk, dv, None, None

  apply.defvjp(forward, backward)
  def call(q, k, v, segment_ids=None):
    # Mask arrays can be tracers created inside shard_map/scan. Pass them as
    # explicit custom-VJP operands rather than capturing them in its closure.
    return apply(q, k, v, segment_ids, masks)
  return call

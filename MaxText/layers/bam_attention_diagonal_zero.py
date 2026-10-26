"""Pre-softmax diagonal suppression using the pinned Splash public interface."""
import jax.numpy as jnp

NEGATIVE_DIAGONAL_LOGIT = -1e9


def make_negative_diagonal_splash(kernel):
  """Wrap strict-causal Splash, returning zero for empty-history weighted V.

  The caller uses a causal mask with offset=-1. A fixed zero-valued sink at
  -1e9 stabilizes empty rows; its probability underflows on nonempty rows.
  This adds no parameter and uses Splash's existing forward and backward.
  """
  def apply(q, k, v, segment_ids=None):
    return kernel(q, k, v, segment_ids=segment_ids,
                  sinks=jnp.full((q.shape[0],), NEGATIVE_DIAGONAL_LOGIT, jnp.float32))
  return apply

"""Analysis-only stable FP32 CE; preserve learned projections and their dtype."""
import jax
import jax.numpy as jnp


def cross_entropy(logits, targets, z_loss=0.):
  z = logits.astype(jnp.float32)
  logp = jax.nn.log_softmax(z, axis=-1)
  loss = -jnp.sum(targets.astype(jnp.float32)*logp, axis=-1)
  aux = jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,axis=-1)**2
  return loss+aux,aux


def install():
  import max_utils
  max_utils.cross_entropy_with_logits = cross_entropy

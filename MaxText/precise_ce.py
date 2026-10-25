"""Analysis-only FP32 CE statistics; learned forward and logits dtype unchanged."""
import jax,jax.numpy as jnp
_INSTALLED=False

def cross_entropy(logits,targets,z_loss=0.):
 z=logits.astype(jnp.float32);t=targets.astype(jnp.float32)
 logp=jax.nn.log_softmax(z,axis=-1);loss=-jnp.sum(t*logp,axis=-1)
 log_z=jax.scipy.special.logsumexp(z,axis=-1);aux=jnp.asarray(z_loss,jnp.float32)*log_z**2
 return loss+aux,aux

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True
 import max_utils
 max_utils.cross_entropy_with_logits=cross_entropy

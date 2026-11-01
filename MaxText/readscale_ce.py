"""FP32 diagnostic CE from unchanged native BF16 logits."""
import jax
import jax.numpy as jnp

def cross_entropy(logits,targets,z_loss=0.):
    z=logits.astype(jnp.float32)
    loss=-jnp.sum(targets.astype(jnp.float32)*jax.nn.log_softmax(z,axis=-1),axis=-1)
    aux=jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,axis=-1)**2
    return loss+aux,aux

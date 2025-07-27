import jax
import jax.numpy as jnp


@jax.custom_vjp
def tpu_friendly_gather(inputs, indices):
  return jnp.take(inputs, indices, axis=0, unique_indices=False, indices_are_sorted=False)


def _tpu_friendly_gather_fwd(inputs, indices):
  outputs = tpu_friendly_gather(inputs, indices)
  residuals = (inputs.shape, indices)
  return outputs, residuals


def _tpu_friendly_gather_bwd(residuals, grad_outputs):
  input_shape, indices = residuals
  inverse_permutation = jnp.argsort(indices)
  grad_outputs_permuted = grad_outputs[inverse_permutation] # (BTK)D
  grad_inputs = grad_outputs_permuted.reshape(input_shape[0], -1, input_shape[-1]).sum(-2)
  return (grad_inputs, None)


tpu_friendly_gather.defvjp(_tpu_friendly_gather_fwd, _tpu_friendly_gather_bwd)


@jax.custom_vjp
def tpu_gather_by_permutation(inputs, permutation_indices):
  return jnp.take(inputs, permutation_indices, axis=0)

def _tpu_gather_by_permutation_fwd(inputs, permutation_indices):
  outputs = tpu_gather_by_permutation(inputs, permutation_indices)
  residuals = permutation_indices
  return outputs, residuals

def _tpu_gather_by_permutation_bwd(residuals, grad_outputs):
  permutation_indices = residuals
  inverse_permutation = jnp.argsort(permutation_indices)
  grad_inputs = jnp.take(grad_outputs, inverse_permutation, axis=0)
  
  return (grad_inputs, None)

tpu_gather_by_permutation.defvjp(_tpu_gather_by_permutation_fwd, _tpu_gather_by_permutation_bwd)
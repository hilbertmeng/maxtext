"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=bare-except, consider-using-generator, ungrouped-imports, too-many-positional-arguments
"""Utils that are only interesting to MaxText. """
from typing import Any, Callable, Optional, Union
import chex
from functools import partial

import jax

import flax.traverse_util as traverse_util
import optax
import jax.numpy as jnp
from optax.contrib._muon import scale_by_muon
from optax._src import combine
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform


def scale_by_learning_rate(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    *,
    flip_sign: bool = True,
    scale: float = 1.0,
) -> base.GradientTransformation:
  if learning_rate is None:
    return base.identity()
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count) * scale)
  return transform.scale(m * learning_rate)


def muon(
    learning_rate_schedule: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_optimizer: Optional[Any] = None,
) -> base.GradientTransformation:
  
  def build_param_labels(params):
    flat_params = traverse_util.flatten_dict(params)
    param_labels = {}
    for _k, v in flat_params.items():
      k = "/".join(_k)
      ndim = v.ndim if hasattr(v, 'ndim') else v.value.ndim
      if 'embedding' in k: # embedding 和 head lr不同
      # if 'embedding' in k or 'logits_dense' in k:
        label = 'adam_embed'
      elif ndim >= 2 and 'embedding' not in k and 'logits_dense' not in k:
        label = 'muon'
      elif ndim == 1 and 'scale' in k:
        label = 'adam_rms'
      elif ndim == 1 and 'bias' in k:
        label = 'adam_bias'
      else:
        label = 'adam'
      print(f'k: {k}, label: {label} ndim: {ndim}')
      param_labels[_k] = label
    return traverse_util.unflatten_dict(param_labels)

  muon_mask = jax.tree_util.tree_map(lambda x: False if x == 0.0 else True, weight_decay_mask) \
                                    if weight_decay_mask is not None else None # lsp
  return combine.partition(
      transforms={
          'muon': combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
              ),
              transform.add_decayed_weights(weight_decay, muon_mask),
              scale_by_learning_rate(learning_rate_schedule, scale=8),
          ),
          'adam_embed': adam_optimizer(weight_decay=0.0, lr_coef=100),
          'adam_rms': adam_optimizer(weight_decay=0.0, lr_coef=1.0),
          'adam_bias': adam_optimizer(weight_decay=0.0, lr_coef=1.0),
          'adam': adam_optimizer(weight_decay=0.0, lr_coef=1.0),
      },
      # lsp: Only two dims use muon, other use adam
      param_labels=lambda params: build_param_labels(params),
  )

def get_optimizer(config, learning_rate_schedule, wd_tree=None):
  """create optimizer"""
  print(f'opt_type: {config.opt_type}')
  if config.opt_type == "adamw":
    mask = jax.tree_util.tree_map(lambda x: False if x == 0.0 else True, wd_tree) if wd_tree else None
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    optimizer = optax.adamw(
        learning_rate_schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        eps_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=mask,
    )
  elif config.opt_type == "adam_pax":
    optimizer = adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        wd_tree=wd_tree, # lsp
    )
  elif config.opt_type == "sgd":
    optimizer = optax.sgd(learning_rate_schedule)
  elif config.opt_type == "muon":
    adam_optimizer = partial(adam_pax,
        learning_rate_schedule,
        beta1=0.8,
        beta2=0.95,
        epsilon=1e-10,
        epsilon_root=config.adam_eps_root,
        # weight_decay=config.adam_weight_decay,
        wd_tree=None, # lsp
    )
    optimizer = muon(
        learning_rate_schedule,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        weight_decay_mask=wd_tree,
        adaptive=False, # default is False, 开启 adaptive 不只是变方向，也会变“有效步长”。如果开启比较稳定，可以适当调大学习率
        adam_optimizer=adam_optimizer,
      )
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")
  
  if config.gradient_clipping_threshold > 0:
    if config.clip_by_global_norm:
      print(f'clip_by_global_norm: {config.gradient_clipping_threshold}')
      return optax.chain(
          optax.clip_by_global_norm(config.gradient_clipping_threshold),
          optimizer,
      )
    else:
      print(f'clip: {config.gradient_clipping_threshold}')
      return optax.chain(
          optax.clip(config.gradient_clipping_threshold),
          optimizer,
      )
  return optimizer


def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    wd_tree=None,  # lsp
    lr_coef=1.0,
) -> optax.GradientTransformation:
  """Standard Adam optimizer that supports weight decay.

  Follows the implementation in pax/praxis sharded_adam
  https://github.com/google/praxis/blob/545e00ab126b823265d70c715950d39333484f38/praxis/optimizers.py#L621

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `optax.GradientTransformation`.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.0
    return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))

  def update_fn(updates, state, params=None):
    # Sanitize updates just in case.
    if weight_decay > 0:
      assert params is not None
    count = state.count

    class _slot_opt_state:

      def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu

    def _update_momentum(update, mu, nu):
      # The conversion to the data type of the update ensures that bfloat16 remains
      # bfloat16 in the optimizer state. This conversion has to be done after
      # `bias_corrected_dacay` is calculated as calculating `jnp.power(decay, t)` in low
      # precision can result in it being rounded to 1 and subsequently a
      # "division by zero" error.
      beta1_decay = bias_corrected_decay(count, beta1).astype(update)
      beta2_decay = bias_corrected_decay(count, beta2).astype(update)
      mu = (1.0 - beta1_decay) * update + beta1_decay * mu
      nu = (1.0 - beta2_decay) * (update**2) + beta2_decay * nu
      return _slot_opt_state(mu=mu, nu=nu)

    updated_moments = jax.tree_util.tree_map(_update_momentum, updates, state.mu, state.nu)

    mu = jax.tree_util.tree_map(lambda x: x.mu, updated_moments)
    nu = jax.tree_util.tree_map(lambda x: x.nu, updated_moments)

    updates = jax.tree_util.tree_map(lambda mu, nu: mu / (jnp.sqrt(nu + epsilon_root) + epsilon), mu, nu)

    if wd_tree is None:
        updates = jax.tree_util.tree_map(lambda x, v: x + weight_decay * v, updates, params)
    else: # lsp
        updates = jax.tree_util.tree_map(lambda x, v, wd: x + wd * v, updates, params, wd_tree)

    step_size = -1.0 * learning_rate_fn(count) * lr_coef # lsp
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)

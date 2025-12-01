"""Utils that are only interesting to MaxText. """
from typing import Any, Callable, Optional, Union
import chex
from functools import partial
import math

import jax
import optax
import jax.numpy as jnp
import flax.traverse_util as traverse_util

from optax.contrib._muon import scale_by_muon
from optax._src import combine
from optax._src import base
from optax._src import transform

import max_utils
import max_logging


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

# muon must decay
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
    config: Optional[Any] = None,
) -> base.GradientTransformation:

  def build_param_labels(params):
    flat_params = traverse_util.flatten_dict(params)
    param_labels = {}

    def get_ndim_shape(v):
        if hasattr(v, "ndim"):
            return v.ndim, v.shape
        if hasattr(v, "value"):
            return getattr(v.value, "ndim", 1), v.value.shape
        return 1

    def get_label(k, ndim, vshape):
        if 'bias' in k:
          return 'adam_nowd'
        elif (ndim == 1 and 'scale' in k):
          return 'adam_nowd' if config.direct_scale else 'adam_default'
        elif (ndim == 2 and 'scale' in k and vshape[1] in [1, 2, 3] and config.partial_scan_layers):
          return 'adam_nowd' if config.direct_scale else 'adam_default'
        if any(x in k for x in ['embedding', 'logits_dense']):
          return 'adam_default'
        if 'dyn_w_proj' in k:
          return 'muon_attn' if config.dc_use_muon else 'adam_default'
        if 'compose' in k:
          return 'muon_attn' if config.mudd_use_muon else 'adam_default'
        if ndim == 2 and 'attention' in k:
          return 'muon_attn'
        if ndim == 2 and 'mlp' in k and 'compose' not in k:
          return 'muon_mlp'
        if config.partial_scan_layers and ndim == 3 and vshape[1] in [1, 2, 3]:
          if 'attention' in k:
            return 'muon_attn'
          if 'mlp' in k and 'compose' not in k:
            return 'muon_mlp'

        return 'adam_default'

    for _k, v in flat_params.items():
        k = "/".join(_k)
        ndim, vshape = get_ndim_shape(v)
        label = get_label(k, ndim, vshape)
        max_logging.log(f"k: {k} label: {label} ndim: {ndim} vshape: {vshape}")
        param_labels[_k] = label

    return traverse_util.unflatten_dict(param_labels)

  def weight_dim_nums_fn(params): # optax>=0.2.6
    def get_dim_nums(x):
        if x.ndim == 2:
          # Standard 2D matrix: [dim1, dim2]
          return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=1)
        elif x.ndim == 3:
          # Scanned layers: [dim1, L, dim2], L will be treated as batch axis
          # lsp: reduction_axis: row, output_axis: column, other axis will be treated as batch axis
          return optax.contrib.MuonDimensionNumbers(reduction_axis=0, output_axis=2)
        else:
          raise ValueError(f'Unsupported dimension: {x.ndim}')
    
    return jax.tree.map(get_dim_nums, params)

  # muon_mask = _build_wd_bool_mask_from_tree(weight_decay_mask)
  muon_kwargs = {
    'ns_coeffs': ns_coeffs,
    'ns_steps': ns_steps,
    'beta': beta,
    'eps': eps,
    'mu_dtype': mu_dtype,
    'nesterov': nesterov,
    'adaptive': adaptive,
  }
  if optax.__version__ >= '0.2.6': # speed up
    muon_kwargs['weight_dimension_numbers'] = weight_dim_nums_fn
  muon_base = scale_by_muon(**muon_kwargs)

  default_scale = 1.0
  attn_scale = math.sqrt(max(config.num_query_heads * config.head_dim, config.emb_dim)) * config.muon_scale
  mlp_scale = math.sqrt(max(config.num_query_heads * config.head_dim, config.mlp_dim)) * config.muon_scale
  max_logging.log(f'attn_scale: {attn_scale}, mlp_scale: {mlp_scale} weight_decay: {weight_decay}')

  muon_final_lr = config.final_muon_scale * config.learning_rate * config.cosine_learning_rate_final_fraction / config.muon_scale
  muon_learning_rate_schedule = max_utils.create_learning_rate_schedule(config, final_lr=muon_final_lr)
  max_logging.log(f'final_muon_scal: {config.final_muon_scale} muon_final_lr: {muon_final_lr}')

  return combine.partition(
      transforms={
          'muon_attn': combine.chain(
              muon_base,
              transform.add_decayed_weights(weight_decay, mask=None), # Can use muon_mask to control wd
              scale_by_learning_rate(muon_learning_rate_schedule, scale=attn_scale),
          ),
          'muon_mlp': combine.chain(
              muon_base,
              transform.add_decayed_weights(weight_decay, mask=None), # Can use muon_mask to control wd
              scale_by_learning_rate(muon_learning_rate_schedule, scale=mlp_scale),
          ),
          # Small model rms wd set 0.0 better, bigger model unknow. but muon paper suggest wd=0.1
          # 1+scale mode need wd, other not need.
          'adam_nowd': adam_optimizer(weight_decay=0.0),
          'adam_default': adam_optimizer(weight_decay=weight_decay),
      },
      # lsp: Only two dims use muon, other use adam
      param_labels=lambda params: build_param_labels(params),
  )


def _build_adamw(config, learning_rate_schedule, wd_tree):

  def build_param_labels(params):
    flat_params = traverse_util.flatten_dict(params)
    param_labels = {}
    for _k, v in flat_params.items():
      k = "/".join(_k)
      ndim = v.ndim if hasattr(v, 'ndim') else v.value.ndim
      label = 'adam_default'
      if 'deep' in k:
        label = 'adam_deep'
      max_logging.log(f'k: {k}, label: {label} ndim: {ndim}')
      param_labels[_k] = label
    return traverse_util.unflatten_dict(param_labels)

  # mask = _build_wd_bool_mask_from_tree(wd_tree)
  adam_optimizer = partial(
      adam_pax,
      learning_rate_schedule,
      beta1=config.adam_b1,
      beta2=config.adam_b2,
      epsilon=config.adam_eps,
      epsilon_root=config.adam_eps_root,
      wd_tree=None,
  )
  adam_base = transform.scale_by_adam(
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
  )
  weight_decay = config.adam_weight_decay
  deep_scale = 100 # lr scale for deep
  adam_one_dim_scale = 1.0
  adam_default_scale = 1.0
  return combine.partition(
      transforms={
          'adam_deep': combine.chain(
              adam_base,
              transform.add_decayed_weights(weight_decay, mask=None), # Can use muon_mask to control wd
              scale_by_learning_rate(learning_rate_schedule, scale=deep_scale),
          ),
          'adam_one_dim': adam_optimizer(weight_decay=0.0, lr_coef=adam_one_dim_scale),
          'adam_default': adam_optimizer(weight_decay=weight_decay, lr_coef=adam_default_scale),
      },
      # lsp: Only two dims use muon, other use adam
      param_labels=lambda params: build_param_labels(params),
  )


def _build_adam_pax(config, learning_rate_schedule, wd_tree):
  return adam_pax(
      learning_rate_schedule,
      beta1=config.adam_b1,
      beta2=config.adam_b2,
      epsilon=config.adam_eps,
      epsilon_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
      wd_tree=wd_tree,
  )


def _build_sgd(_config, learning_rate_schedule, _wd_tree):
  return optax.sgd(learning_rate_schedule)


def _build_muon(config, learning_rate_schedule, wd_tree):
  adam_optimizer = partial(
      adam_pax,
      learning_rate_schedule,
      beta1=config.adam_b1,
      beta2=config.adam_b2,
      epsilon=config.adam_eps,
      epsilon_root=config.adam_eps_root,
      wd_tree=None,
  )
  return muon(
      learning_rate_schedule,
      eps=config.adam_eps,
      weight_decay=config.adam_weight_decay,
      weight_decay_mask=wd_tree,
      adaptive=False,
      adam_optimizer=adam_optimizer,
      config=config,
  )


_OPTIMIZER_BUILDERS = {
  'adamw': _build_adamw,
  'adam_pax': _build_adam_pax,
  'sgd': _build_sgd,
  'muon': _build_muon,
}


def get_optimizer(config, learning_rate_schedule, wd_tree=None):
  max_logging.log(f'opt_type: {config.opt_type}')
  try:
    optimizer_builder = _OPTIMIZER_BUILDERS[config.opt_type]
  except KeyError:
    raise ValueError(f"{config.opt_type=} is not a supported.")

  optimizer = optimizer_builder(config, learning_rate_schedule, wd_tree)
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

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
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
      beta1_decay = bias_corrected_decay(count, beta1).astype(update.dtype)
      beta2_decay = bias_corrected_decay(count, beta2).astype(update.dtype)
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

    step_size = -lr_coef * learning_rate_fn(count) # lsp
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)
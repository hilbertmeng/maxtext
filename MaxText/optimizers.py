"""Utils that are only interesting to MaxText. """
from typing import Any, Callable, NamedTuple, Optional, Union
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


# ================================= [NorMuon] =================================
# NorMuon: Making Muon more efficient and scalable
# Paper: https://arxiv.org/abs/2510.05491
# Core idea: Add neuron-level adaptive learning rate after Muon's orthogonalization
# to address the issue of non-uniform neuron norms caused by orthogonalization.

class ScaleByNeuronNormState(NamedTuple):
  """State for scale_by_neuron_norm."""
  count: chex.Array  # step count
  nu: base.Updates   # second moment (neuron-level, stored as per-row mean)


def scale_by_neuron_norm(
    beta2: float = 0.999,
    eps: float = 1e-8,
    reduction_axis: int = -1,
) -> base.GradientTransformation:
  """NorMuon's neuron-level normalization after Muon orthogonalization.
  
  This transformation computes the second moment of the updates at the neuron
  level (row-wise mean of squared updates) and normalizes each row accordingly.
  
  Args:
    beta2: Decay rate for the second moment (EMA of squared updates).
    eps: Small constant for numerical stability.
    reduction_axis: Axis to compute the mean over (default -1, i.e., columns).
  
  Returns:
    A GradientTransformation that applies neuron-level normalization.
  """
  
  def init_fn(params):
    # Initialize second moment as per-neuron (per-row) vectors
    def _init_nu(p):
      if p.ndim >= 2:
        # For 2D: shape [rows, cols] -> nu shape [rows, 1]
        # For 3D: shape [rows, batch, cols] -> nu shape [rows, batch, 1]
        shape = list(p.shape)
        shape[reduction_axis] = 1
        return jnp.zeros(shape, dtype=p.dtype)
      else:
        # For 1D params, store scalar second moment
        return jnp.zeros((), dtype=p.dtype)
    
    nu = jax.tree_util.tree_map(_init_nu, params)
    return ScaleByNeuronNormState(count=jnp.zeros([], jnp.int32), nu=nu)
  
  def update_fn(updates, state, params=None):
    del params
    count = state.count
    nu = state.nu
    
    # Bias correction for second moment
    count_inc = count + 1
    bias_correction = 1.0 - jnp.power(beta2, count_inc)
    
    def _update_neuron_norm(update, nu_prev):
      if update.ndim >= 2:
        # Compute per-row mean of squared updates: mean_cols(O_t ⊙ O_t)
        sq_mean = jnp.mean(update ** 2, axis=reduction_axis, keepdims=True)
        # EMA update: v_t = beta2 * v_{t-1} + (1 - beta2) * sq_mean
        nu_new = beta2 * nu_prev + (1 - beta2) * sq_mean
        # Bias-corrected second moment
        nu_hat = nu_new / bias_correction
        # Normalize: O_hat_t = O_t / sqrt(v_t + eps)
        normalized = update / (jnp.sqrt(nu_hat) + eps)
        return normalized, nu_new
      else:
        # For 1D params, apply scalar normalization
        sq_mean = jnp.mean(update ** 2)
        nu_new = beta2 * nu_prev + (1 - beta2) * sq_mean
        nu_hat = nu_new / bias_correction
        normalized = update / (jnp.sqrt(nu_hat) + eps)
        return normalized, nu_new
    
    updates_and_nu = jax.tree_util.tree_map(_update_neuron_norm, updates, nu)
    new_updates = jax.tree_util.tree_map(lambda x: x[0], updates_and_nu)
    new_nu = jax.tree_util.tree_map(lambda x: x[1], updates_and_nu)
    
    return new_updates, ScaleByNeuronNormState(count=count_inc, nu=new_nu)
  
  return base.GradientTransformation(init_fn, update_fn)


def scale_by_normuon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta1: float = 0.95,
    beta2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    weight_dimension_numbers: Optional[
        Union[
            optax.contrib.MuonDimensionNumbers,
            Callable[[base.Params], optax.contrib.MuonDimensionNumbers],
        ]
    ] = None,
) -> base.GradientTransformation:
  """NorMuon optimizer: Muon with neuron-level normalization.
  
  NorMuon improves upon Muon by adding a neuron-level adaptive learning rate
  after the orthogonalization step. This addresses the issue of non-uniform
  neuron norms caused by Muon's orthogonalization.
  
  Algorithm:
    1. First moment: M_t = beta1 * M_{t-1} + (1 - beta1) * G_t
    2. Orthogonalization: O_t = NS5(M_t) (Newton-Schulz iteration)
    3. Second moment (neuron-level): v_t = beta2 * v_{t-1} + (1 - beta2) * mean_cols(O_t ⊙ O_t)
    4. Normalize: O_hat_t = O_t / sqrt(v_t + eps)
  
  Args:
    ns_coeffs: Coefficients for Newton-Schulz iteration.
    ns_steps: Number of Newton-Schulz iterations.
    beta1: Decay rate for first moment (momentum).
    beta2: Decay rate for second moment (neuron-level normalization).
    eps: Small constant for numerical stability.
    mu_dtype: Data type for momentum.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to use adaptive scaling (from original Muon).
    weight_dimension_numbers: Dimension specification for Muon.
  
  Returns:
    A GradientTransformation implementing NorMuon.
  """
  muon_kwargs = {
      'ns_coeffs': ns_coeffs,
      'ns_steps': ns_steps,
      'beta': beta1,
      'eps': eps,
      'mu_dtype': mu_dtype,
      'nesterov': nesterov,
      'adaptive': adaptive,
  }
  if weight_dimension_numbers is not None:
    muon_kwargs['weight_dimension_numbers'] = weight_dimension_numbers
  
  return combine.chain(
      scale_by_muon(**muon_kwargs),
      scale_by_neuron_norm(beta2=beta2, eps=eps, reduction_axis=-1),
  )


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


def muon_scale_schedule(config):
  def schedule(step):
      # 务必注意，这个传入的step不是实际的步数，是从0开始的
      pct = (step) / max(1, config.learning_rate_schedule_steps)
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      # lr * a + final_lr * (1 - a)
      scale = config.muon_scale * a + config.final_muon_scale * (1 - a)
      return scale
  return schedule
    

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
    normuon_beta2: float = 0.999,
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
  # Original Muon
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

  attn_dim_sqrt = math.sqrt(max(config.num_query_heads * config.head_dim, config.emb_dim))
  mlp_dim_sqrt = math.sqrt(max(config.num_query_heads * config.head_dim, config.mlp_dim))
  max_logging.log(f'attn_dim_sqrt: {attn_dim_sqrt}, mlp_dim_sqrt: {mlp_dim_sqrt} weight_decay: {weight_decay}')

  muon_final_lr = config.final_muon_scale * config.learning_rate * config.cosine_learning_rate_final_fraction / config.muon_scale
  max_logging.log(f'final_muon_scal: {config.final_muon_scale} muon_final_lr: {muon_final_lr}')

  return combine.partition(
      transforms={
          'muon_attn': combine.chain(
              muon_base,
              scale_by_learning_rate(muon_scale_schedule(config), scale=attn_dim_sqrt, flip_sign=False),
              transform.add_decayed_weights(weight_decay, mask=None), # Can use muon_mask to control wd
              scale_by_learning_rate(learning_rate_schedule),
          ),
          'muon_mlp': combine.chain(
              muon_base,
              scale_by_learning_rate(muon_scale_schedule(config), scale=mlp_dim_sqrt, flip_sign=False),
              transform.add_decayed_weights(weight_decay, mask=None), # Can use muon_mask to control wd
              scale_by_learning_rate(learning_rate_schedule),
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
  # Check if NorMuon is enabled via config
  normuon_beta2 = getattr(config, 'normuon_beta2', 0.999)
  
  return muon(
      learning_rate_schedule,
      eps=config.adam_eps,
      weight_decay=config.adam_weight_decay,
      weight_decay_mask=wd_tree,
      adaptive=False,
      adam_optimizer=adam_optimizer,
      config=config,
      normuon_beta2=normuon_beta2,
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
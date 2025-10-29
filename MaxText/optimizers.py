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


def _build_wd_bool_mask_from_tree(wd_tree):
  """Converts a weight-decay coefficient tree into a boolean mask tree.

  Any value equal to 0.0 becomes False, otherwise True. If `wd_tree` is None,
  returns None.
  """
  if wd_tree is None:
    return None
  return jax.tree_util.tree_map(lambda x: False if x == 0.0 else True, wd_tree)


def _apply_clipping(optimizer: optax.GradientTransformation, config) -> optax.GradientTransformation:
  """Optionally wraps optimizer with gradient clipping according to config."""
  if getattr(config, 'gradient_clipping_threshold', 0) > 0:
    if getattr(config, 'clip_by_global_norm', False):
      max_logging.log(f'clip_by_global_norm: {config.gradient_clipping_threshold}')
      return optax.chain(
          optax.clip_by_global_norm(config.gradient_clipping_threshold),
          optimizer,
      )
    else:
      max_logging.log(f'Error clip: {config.gradient_clipping_threshold}')
      return optax.chain(
          optax.clip(config.gradient_clipping_threshold),
          optimizer,
      )
  return optimizer

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
    param_labels, weight_dim_nums = {}, {}
    for _k, v in flat_params.items():
      k = "/".join(_k)
      if hasattr(v, 'ndim'):
        ndim = v.ndim
      elif hasattr(v, 'value'):
        ndim = v.value.ndim
      else:
        ndim = 1
      label = 'adam_default'
      # if 'bias' in k or (ndim == 1 and 'scale' in k) or 'dynamic_dense_conn2' in k: # rms no wd better(0.002), but muon paper suggest wd
      if 'bias' in k or (ndim == 1 and 'scale' in k): # rms no wd better(0.002), but muon paper suggest wd
        label = 'adam_one_dim'
      elif 'embedding' in k:
        label = 'adam_default'
      elif 'dyn_w_proj' in k:
        if config.dc_use_muon:
          label = 'muon_attn'
        else:
          label = 'adam_default'
      elif 'compose' in k:
        if config.mudd_use_muon:
          label = 'muon_attn'
        else:
          label = 'adam_default'
      elif ndim == 2 and 'attention' in k:
        label = 'muon_attn'
      elif ndim == 2 and 'mlp' in k and 'compose' not in k: # remove mudd params
        label = 'muon_mlp'
      max_logging.log(f'k: {k}, label: {label} ndim: {ndim}')
      param_labels[_k] = label
    return traverse_util.unflatten_dict(param_labels)

  def weight_dim_nums_fn(params): # optax>=0.2.6
    return jax.tree.map(lambda x: optax.contrib.MuonDimensionNumbers((0,), (1,)), params)

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

  if config.muon_decay_ratio and config.muon_decay_ratio > 0.0:
    muon_final_lr = config.muon_decay_ratio * config.learning_rate * 0.2 / config.muon_scale # 不论muon_scale是多少，均按照 muon_scale=0.1 计算的最终学习率
    muon_learning_rate_schedule = max_utils.create_learning_rate_schedule(config, final_lr=muon_final_lr)
    max_logging.log(f'muon_final_lr: {muon_final_lr}')
  else:
    muon_learning_rate_schedule = learning_rate_schedule     

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
          'adam_one_dim': adam_optimizer(weight_decay=0.0, lr_coef=default_scale),
          'adam_default': adam_optimizer(weight_decay=weight_decay, lr_coef=default_scale),
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
# below no add muon scale
# beta1=0.8, beta2=0.95, epsilon=1e-10, qkvo+mlp wd 0.1, other wd=0.0, qkvo+mlp lr 8x, embed lr 100x, other lr 1x, eval loss: 2.4190
# beta1=0.8, beta2=0.95, epsilon=1e-10, all wd=0.1 exp rms wd=0.0, qkvo+mlp lr 8x, embed lr 100x, other lr 1x, eval loss: 2.4191
# beta1=0.8, beta2=0.95, epsilon=1e-10, all wd=0.0 , qkvo+mlp lr 8x, embed lr 100x, other lr 1x, eval loss: 2.4221
# below add muon scale
# beta1=0.8, beta2=0.95, epsilon=1e-10, all wd=0.1, val loss: 2.4233
# beta1=0.8, beta2=0.95, epsilon=1e-10, all  wd=0.1, exp Rms wd=0.0, eval loss: 2.4178
# beta1=0.9, beta2=0.95, epsilon=1e-8,  all wd=0.1, val loss: 2.4216
# beta1=0.9, beta2=0.95, epsilon=1e-8, all  wd=0.1, exp Rms wd=0.0, eval loss: 2.4158
# 总结：
# 1、如果某个参数（非1维矩阵）设置了wd，那么对于模型而言，前期loss会低，但是后期乏力，因此，wd会较大的影响后期模型的性能，最终设置了wd会好一些
# 2、如果某个参数的学习率设置的较大，那么对于模型而言，前期loss会低，但是后期乏力，因此，需要一个合适的学习率。过大或者过小都不是很好。
# 3、如果某个参数因为学得慢需要设置较大学习率，最后尽量和正常参数学习率的比率保持一致。最好不要decay到一样的学习率。也就是大学习率和小学习率始终保持一个固定比例较好

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

    step_size = -lr_coef * learning_rate_fn(count) # lsp
    # Finally, fold in step size.
    updates = jax.tree_util.tree_map(lambda x: step_size * x, updates)

    updated_states = optax.ScaleByAdamState(count=count + 1, mu=mu, nu=nu)
    return updates, updated_states

  return optax.GradientTransformation(init_fn, update_fn)

#  Copyright 2023 Google LLC

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#       https://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Initializers."""

from typing import Callable, Tuple, Union

from flax import linen as nn
import jax
import common_types
import jax.numpy as jnp

Array = common_types.Array
DType = common_types.DType
PRNGKey = common_types.PRNGKey
Shape = common_types.Shape

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)

default_bias_init = jax.nn.initializers.constant(0.0)


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""

  def init_fn(key, shape, dtype, in_axis=0, out_axis=1):
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn


def contant_dense_init(value):
  """Initializer with in_axis, out_axis set at call time."""

  def init_fn(key, shape, dtype, in_axis=0, out_axis=1):
    fn = jax.nn.initializers.constant(value)
    return fn(key, shape, dtype)

  return init_fn

def nd_dense_init_normal(scale, min_val=None, max_val=None):
  def init_fn(key, shape, dtype, in_axis=0, out_axis=1):
    fn = nn.initializers.normal(scale)
    return jnp.clip(fn(key, shape, dtype), min=min_val, max=max_val) # lsp: add min, max

  return init_fn


def get_init_method(init_method):
  if init_method == 'olmoe':
    init_fun = nd_dense_init_normal(0.02, min_val=-0.06, max_val=0.06)
  elif init_method == 'truncated_normal':
    init_fun = nd_dense_init(1.0, "fan_in", "truncated_normal")
  else:
    init_fun = nd_dense_init_normal(0.006)

  return init_fun
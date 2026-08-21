#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Normalization Layers."""

from typing import Any, Tuple, Optional

from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from layers import initializers

Initializer = initializers.Initializer
DEFAULT_RMS_EPSILON = 1e-6


def rms_norm(
    x: jnp.ndarray,
    *,
    dtype: Any,
    epsilon: float = DEFAULT_RMS_EPSILON,
    axis: int = -1,
    statistics_dtype: Any = jnp.float32,
) -> jnp.ndarray:
    """Parameter-free RMS normalization with configurable statistics dtype."""
    x_stats = jnp.asarray(x, statistics_dtype)
    mean2 = jnp.mean(lax.square(x_stats), axis=axis, keepdims=True)
    return jnp.asarray(x_stats * lax.rsqrt(mean2 + epsilon), dtype)


class RMSNorm(nn.Module):
    """RMS normalization."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    weight_dtype: Any = jnp.float32
    kernel_axes: Tuple[Optional[str], ...] = ()
    scale_init: Initializer = nn.initializers.zeros
    direct_scale: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        if not self.scale_init:
            return y
        scale = self.param(
            "scale",
            nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
            (features,),
            self.weight_dtype,
        )
        scale = jnp.asarray(scale, self.dtype)
        if not self.direct_scale:
            assert self.scale_init == nn.initializers.zeros
        return y * scale if self.direct_scale else y * (scale + 1.0)


def get_rmsnorm(name, cfg, **kwargs):
    rms_kwargs = {'kernel_axes': ('norm',),}
    for item in ['dtype', 'weight_dtype', 'normalization_layer_epsilon', 'direct_scale']:
        key = 'epsilon' if item == 'normalization_layer_epsilon' else item
        rms_kwargs[key] = kwargs.get(key, getattr(cfg, item))
    base_scale_init = nn.initializers.ones if rms_kwargs['direct_scale'] else nn.initializers.zeros
    rms_kwargs['scale_init'] = kwargs.get("scale_init", base_scale_init)
    return RMSNorm(name=name, **rms_kwargs)

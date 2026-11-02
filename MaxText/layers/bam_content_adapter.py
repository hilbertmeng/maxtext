"""Identity-initialized, per-head BAM content recoding (the identity has no WD)."""
import math
import jax
import jax.numpy as jnp
from flax import linen as nn


class HeadContentAdapter(nn.Module):
  num_heads: int
  head_dim: int
  rank: int = 0
  dtype: object = jnp.bfloat16
  weight_dtype: object = jnp.float32
  precision: object = None

  @nn.compact
  def __call__(self, x):
    if x.shape[-2:] != (self.num_heads, self.head_dim):
      raise ValueError(f'Content adapter expects heads/data, got {x.shape}')
    if self.rank == 0:
      delta = self.param('delta', nn.with_logical_partitioning(
          nn.initializers.zeros, ('q_heads', None, None)),
          (self.num_heads, self.head_dim, self.head_dim), self.weight_dtype)
      update = jnp.einsum('...nd,ndk->...nk', x, delta.astype(self.dtype), precision=self.precision)
    else:
      if not 0 < self.rank <= self.head_dim:
        raise ValueError('Content adapter rank must lie in [1, head_dim]')
      down = self.param('down', nn.with_logical_partitioning(
          nn.initializers.normal(1. / math.sqrt(self.head_dim)), ('q_heads', None, None)),
          (self.num_heads, self.head_dim, self.rank), self.weight_dtype)
      up = self.param('up', nn.with_logical_partitioning(
          nn.initializers.zeros, ('q_heads', None, None)),
          (self.num_heads, self.rank, self.head_dim), self.weight_dtype)
      hidden = jnp.einsum('...nd,ndr->...nr', x, down.astype(self.dtype), precision=self.precision)
      update = jnp.einsum('...nr,nrk->...nk', hidden, up.astype(self.dtype), precision=self.precision)
    return x + update.astype(x.dtype)


def relative_update_norm(parameters):
  """RMS over heads of ||Delta||_F / ||I||_F; no activation capture needed."""
  p = jax.tree.map(lambda v: jax.lax.stop_gradient(v).astype(jnp.float32), parameters)
  if 'delta' in p:
    square = jnp.sum(jnp.square(p['delta']), axis=(-2, -1))
    width = p['delta'].shape[-1]
  else:
    # tr((down^T down)(up up^T)), forming only rank x rank matrices.
    gram_down = jnp.einsum('ndr,nds->nrs', p['down'], p['down'])
    gram_up = jnp.einsum('nrd,nsd->nrs', p['up'], p['up'])
    square = jnp.maximum(jnp.sum(gram_down * gram_up, axis=(-2, -1)), 0.)
    width = p['up'].shape[-1]
  return jnp.sqrt(jnp.mean(square) / width)


def parameter_health(params, config):
  """One effective-update statistic per layer/route, including scanned layers."""
  from flax.traverse_util import flatten_dict
  flat = flatten_dict(params)
  groups = {}
  for path, value in flat.items():
    if len(path) > 1 and path[-2] in ('content_write_adapter', 'content_o_adapter', 'content_v_adapter'):
      groups.setdefault(path[:-1], {})[path[-1]] = value
  metrics = {}
  for path, p in groups.items():
    route = path[-1].removeprefix('content_').removesuffix('_adapter')
    scanned = 'layers' in path
    if scanned:
      p = jax.tree.map(lambda v: jnp.moveaxis(v, config.param_scan_axis, 0), p)
      count = next(iter(p.values())).shape[0]
      values = jax.vmap(relative_update_norm)(p)
      for i in range(count):
        metrics[f'bam/content_adapter/{route}/layer_{i:03d}/update_over_identity'] = values[i]
    else:
      layer = config.num_decoder_layers - 1
      metrics[f'bam/content_adapter/{route}/layer_{layer:03d}/update_over_identity'] = relative_update_norm(p)
  return metrics

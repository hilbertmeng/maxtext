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
  activation: str = "linear"
  shared_heads: bool = False
  flatten_params: bool = False

  @nn.compact
  def __call__(self, x):
    if self.activation not in ('linear', 'gelu') or (self.activation != 'linear' and self.rank == 0):
      raise ValueError(f'Unsupported content adapter activation={self.activation!r}, rank={self.rank}')
    if self.shared_heads and (self.rank or self.flatten_params):
      raise ValueError('Shared heads require full, unflattened adapter parameters')
    if x.shape[-2:] != (self.num_heads, self.head_dim):
      raise ValueError(f'Content adapter expects heads/data, got {x.shape}')
    # Shard independent heads over the existing parameter-FSDP (embed) axis.
    # Data width75 is not divisible by the8-device Medium mesh; q_heads alone
    # maps only to tensor parallelism and would replicate these parameters.
    if self.rank == 0:
      # A single small shared matrix is replicated. Independent head matrices
      # retain the verified FSDP layout; width75 itself cannot shard across8.
      shape = (self.head_dim, self.head_dim) if self.shared_heads else (self.num_heads, self.head_dim, self.head_dim)
      axes = (None, None) if self.shared_heads else ('embed', None, None)
      if self.flatten_params:
        # Fold head and input-data axes for FSDP64/128; neither32 heads nor
        # data width80 alone divides those meshes. The per-head map is unchanged.
        shape, axes = (self.num_heads * self.head_dim, self.head_dim), ('embed', None)
      delta = self.param('delta', nn.with_logical_partitioning(
          nn.initializers.zeros, axes), shape, self.weight_dtype)
      if self.flatten_params:
        delta = delta.reshape(self.num_heads, self.head_dim, self.head_dim)
      contraction = '...nd,dk->...nk' if self.shared_heads else '...nd,ndk->...nk'
      update = jnp.einsum(contraction, x, delta.astype(self.dtype), precision=self.precision)
    else:
      if not 0 < self.rank <= self.head_dim:
        raise ValueError('Content adapter rank must lie in [1, head_dim]')
      down_shape = (self.num_heads, self.head_dim, self.rank)
      up_shape = (self.num_heads, self.rank, self.head_dim)
      axes = ('embed', None, None)
      if self.flatten_params:
        down_shape = (self.num_heads * self.head_dim, self.rank)
        up_shape = (self.num_heads * self.rank, self.head_dim)
        axes = ('embed', None)
      down = self.param('down', nn.with_logical_partitioning(
          nn.initializers.normal(1. / math.sqrt(self.head_dim)), axes),
          down_shape, self.weight_dtype)
      up = self.param('up', nn.with_logical_partitioning(
          nn.initializers.zeros, axes), up_shape, self.weight_dtype)
      if self.flatten_params:
        down = down.reshape(self.num_heads, self.head_dim, self.rank)
        up = up.reshape(self.num_heads, self.rank, self.head_dim)
      hidden = jnp.einsum('...nd,ndr->...nr', x, down.astype(self.dtype), precision=self.precision)
      if self.activation == 'gelu':
        hidden = nn.gelu(hidden)
      update = jnp.einsum('...nr,nrk->...nk', hidden, up.astype(self.dtype), precision=self.precision)
    return x + update.astype(x.dtype)


def relative_update_norm(parameters, num_heads=None):
  """RMS over heads of ||Delta||_F / ||I||_F; no activation capture needed."""
  p = jax.tree.map(lambda v: jax.lax.stop_gradient(v).astype(jnp.float32), parameters)
  if num_heads is not None:
    if 'delta' in p:
      width = p['delta'].shape[-1]
      p['delta'] = p['delta'].reshape(num_heads, width, width)
    else:
      width, rank = p['up'].shape[-1], p['down'].shape[-1]
      p = dict(down=p['down'].reshape(num_heads, width, rank),
               up=p['up'].reshape(num_heads, rank, width))
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
  heads = config.num_query_heads if getattr(config, 'bam_content_adapter_flatten_params', False) else None
  norm = lambda values: relative_update_norm(values, num_heads=heads)
  nonlinear = getattr(config, 'bam_content_adapter_activation', 'linear') == 'gelu'
  # GELU'(0)=1/2: this is the local adapter Jacobian, not a global linear Delta.
  metric = 'jacobian0_update_over_identity' if nonlinear else 'update_over_identity'
  derivative = .5 if nonlinear else 1.
  for path, p in groups.items():
    route = path[-1].removeprefix('content_').removesuffix('_adapter')
    scanned = 'layers' in path
    if scanned:
      p = jax.tree.map(lambda v: jnp.moveaxis(v, config.param_scan_axis, 0), p)
      count = next(iter(p.values())).shape[0]
      values = jax.vmap(norm)(p)
      for i in range(count):
        metrics[f'bam/content_adapter/{route}/layer_{i:03d}/{metric}'] = derivative * values[i]
    else:
      layer = config.num_decoder_layers - 1
      metrics[f'bam/content_adapter/{route}/layer_{layer:03d}/{metric}'] = derivative * norm(p)
  return metrics

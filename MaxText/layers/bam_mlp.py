"""Generalized BAM column reads used as SwiGLU value channels."""
import dataclasses
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from layers import attentions, initializers, linears


def read_enabled(config, layer_index):
  location = getattr(config, 'bam_mlp_read_location', 'none')
  if location not in ('none', 'write', 'nonwrite', 'all'):
    raise ValueError(f'Unknown MLP BAM-read location: {location}')
  if location == 'none':
    return False
  every = int(config.bam_mlp_write_every)
  writes = (layer_index + 1) % every == int(config.bam_mlp_write_offset) % every
  return location == 'all' or (writes if location == 'write' else not writes)


class BamMlpRead(nn.Module):
  """An independent R256 address branch, with the parent's V-read semantics."""
  config: object
  quant: object = None

  @nn.compact
  def __call__(self, inputs, matrix):
    cfg = self.config
    assert cfg.bam_general_column_read and cfg.bam_m_read_norm == 'none'
    assert isinstance(cfg.num_query_heads, int) and cfg.num_query_heads % 2 == 0
    heads, width = cfg.num_query_heads // 2, cfg.bam_v
    rank = int(cfg.bam_mlp_read_rank)
    dtype, wdtype = cfg.dtype, cfg.weight_dtype
    kernel_init = initializers.get_model_init(cfg)
    def dense(name, features, axes, init=kernel_init):
      return linears.DenseGeneral(features=features, axis=-1, use_bias=False,
          kernel_init=init, kernel_axes=axes, dtype=dtype, weight_dtype=wdtype,
          quant=self.quant, matmul_precision=cfg.matmul_precision, name=name)(inputs)
    hidden = nn.gelu(dense('read_down', rank, ('embed', None)))
    key = linears.DenseGeneral(features=(heads,width), axis=-1, use_bias=False,
        kernel_init=kernel_init, kernel_axes=('embed','q_heads','v_factor'),
        dtype=dtype, weight_dtype=wdtype, quant=self.quant,
        matmul_precision=cfg.matmul_precision, name='read_up')(hidden)
    bias = self.param('read_pre_bias', nn.with_logical_partitioning(
        nn.initializers.zeros, ('q_heads','v_factor')), (heads,width), wdtype)
    key = key + jnp.asarray(bias,dtype)
    epsilon = cfg.bam_read_key_epsilon
    epsilon = cfg.normalization_layer_epsilon if epsilon is None else epsilon
    use_read_gates = bool(getattr(cfg, 'bam_mlp_read_gates', True))
    if use_read_gates:
      opening = cfg.bam_read_gate_init
      opening = math.sqrt(epsilon)/cfg.bam_read_key_scale if opening is None else opening
      zero = initializers.contant_dense_init(0.)
      gate = dense('read_gate',(heads,1),('embed','q_heads',None),zero)
      gate_bias = self.param('read_gate_b0',nn.with_logical_partitioning(
          nn.initializers.constant(math.log(opening/(1-opening))),('q_heads',None)),(heads,1),wdtype)
    arm = attentions._BamReadArm(name='mlp',k_dim=cfg.bam_k,v_dim=width,
        num_heads=heads,read_side='col',prune_row=True,key_mode='rms_gate',
        key_scale=float(cfg.bam_read_key_scale),rms_epsilon=float(epsilon),
        rms_statistics_dtype=jnp.float32 if cfg.bam_read_rms_statistics_dtype=='float32' else dtype,
        gate_activation=jax.nn.sigmoid,implementation=cfg.bam_read_implementation)
    if use_read_gates:
      dynamic_key = attentions._transform_bam_read_key(
          key, arm, gate+jnp.asarray(gate_bias,dtype))
    else:
      dynamic_key = jnp.asarray(arm.key_scale,dtype) * attentions._transform_bam_read_key(
          key, dataclasses.replace(arm,key_mode='rms'))
    static_init = getattr(cfg, 'bam_mlp_read_static_key_init', 'normal')
    assert static_init in ('normal', 'zero')
    static_key = self.param('static_key',nn.with_logical_partitioning(
        nn.initializers.zeros if static_init == 'zero' else nn.initializers.normal(width**-.5),
        ('v_factor','q_heads')),(width,heads),wdtype)
    if use_read_gates:
      static_opening = float(cfg.bam_static_read_gate_init)
      logits = dense('static_gate',heads,('embed','q_heads'),zero)
      static_bias = self.param('static_gate_b0',nn.with_logical_partitioning(
          nn.initializers.constant(math.log(static_opening/(1.1-static_opening))),('q_heads',)),(heads,),wdtype)
      static_gate = 1.1*jax.nn.sigmoid(logits+jnp.asarray(static_bias,dtype))
      effective_key = dynamic_key + jnp.asarray(static_key,dtype).T*(static_gate/static_opening)[...,None]
    else:
      effective_key = dynamic_key + jnp.asarray(static_key,dtype).T
    # Both terms are column keys: combine before one matrix contraction.
    result,_ = attentions.bam_read(matrix,effective_key,dataclasses.replace(arm,key_mode='none'))
    return result.reshape(result.shape[:-2]+(heads*cfg.bam_k,))

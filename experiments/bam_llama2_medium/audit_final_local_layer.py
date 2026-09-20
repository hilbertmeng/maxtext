"""Numerically verify the eight-block scan -> final L boundary on a small model."""
from pathlib import Path
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
from layers import models
from layers.fusion import BamLayerPair

EXP = 'BamMediumIndependentLLFQKConcatStaticLocalVOSharedC8K64QK48Truncate25Layer'
with tempfile.TemporaryDirectory() as out:
  Path(out, 'audit').mkdir()
  cfg = pyconfig.initialize(
      [None, 'MaxText/configs/base.yml'], exp_class=EXP, run_name='audit',
      enable_checkpointing=False, base_output_directory=out+'/', jax_cache_dir='',
      log_config=False, dataset_type='synthetic', base_emb_dim=128,
      base_num_query_heads=2, base_num_kv_heads=2, base_mlp_dim=128,
      vocab_size=128, max_target_length=4, max_prefill_predict_length=4,
      query_chunk_size=4, per_device_batch_size=1., dtype='float32')
  cfg.get_keys()['bam_write_v_bottleneck_dim'] = 32
  cfg.get_keys()['mlp_dim_by_block'] = [128, 128, 120]
  cfg.get_keys()['bam_final_local_mlp_dim'] = 136
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.Transformer(cfg, mesh, quant=None)
  tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
  args = (tokens, jnp.arange(4)[None], tokens+1, jnp.ones_like(tokens), jnp.ones_like(tokens))
  rngs = {'params': jax.random.key(8), 'aqt': jax.random.key(9)}
  variables = model.init(rngs, *args, enable_dropout=False)
  params = variables['params']

  def capture(next_fun, call_args, kwargs, context):
    mod = context.module
    if context.method_name == '__call__' and mod.name == 'final_local_layer':
      mod.sow('intermediates', 'test_input_m', call_args[0][1])
    result = next_fun(*call_args, **kwargs)
    if context.method_name == '__call__' and isinstance(mod, BamLayerPair):
      mod.sow('intermediates', 'test_output_m', result[0][1])
    return result

  with nn.intercept_methods(capture):
    result, captured = model.apply({'params': params}, *args, enable_dropout=False,
                                   mutable=['intermediates'])
  dec = captured['intermediates']['decoder']
  last_f_m = dec['layers']['test_output_m'][0][-1]
  tail_m = dec['final_local_layer']['test_input_m'][0]
  np.testing.assert_array_equal(last_f_m, tail_m)
  assert tail_m.shape == (1, 4, 64, 32)
  assert float(jnp.linalg.norm(tail_m)) > 0
  assert dec['layers']['test_output_m'][0].shape[0] == 8

  def clear_tail_m(next_fun, call_args, kwargs, context):
    if context.method_name == '__call__' and context.module.name == 'final_local_layer':
      h, m = call_args[0]
      call_args = ((h, jnp.zeros_like(m)),) + call_args[1:]
    return next_fun(*call_args, **kwargs)
  with nn.intercept_methods(clear_tail_m):
    ablated = model.apply({'params': params}, *args, enable_dropout=False)
  delta = float(jnp.max(jnp.abs(result[0] - ablated[0])))
  assert delta > 1e-7, delta

  loss = lambda p: jnp.mean(model.apply({'params': p}, *args, enable_dropout=False)[0])
  grads = jax.jit(jax.grad(loss))(params)
  flat = flatten_dict(grads)
  assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(grads))
  tail_norm = sum(float(jnp.sum(v.value**2 if hasattr(v, 'value') else v**2))
                  for p, v in flat.items() if 'final_local_layer' in p)
  assert tail_norm > 0
  print('FINAL_LOCAL_NUMERIC_OK', 'M_continuity=exact', 'blocks=8',
        'tail_M_ablation_delta=', delta, 'tail_grad_squared_norm=', tail_norm, flush=True)

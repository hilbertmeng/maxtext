"""Check one frozen XL layer at saved formal inputs and upstream cotangents.

Capturing a whole-network backward can change BF16 lowering. This runner instead
holds both boundary states and both upstream cotangents fixed, checks native vs
Pallas locally, and decomposes the MLP/norm paths without propagating a changed
forward through the remaining network.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--tensor-root', required=True)
  parser.add_argument('--checkpoint', required=True, help='Checkpoint items directory')
  parser.add_argument('--cohort', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip() == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import tensorflow as tf
  import orbax.checkpoint as ocp
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers import fusion, linears, normalizations

  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'single-layer-vjp').mkdir(exist_ok=True)
  with tf.io.gfile.GFile(args.tensor_root.rstrip('/') + '/tensor-fidelity.json') as f:
    fidelity = json.load(f)
  assert fidelity['passed'], 'Boundary tensors must pass the formal model fidelity gate'
  layer, token = fidelity['target_layer'], fidelity['target_token']
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class='BamXLPropV5FastKernelFullRemat',
      run_name='single-layer-vjp', base_output_directory=str(args.output) + '/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  cp = ocp.PyTreeCheckpointer()
  metadata = cp.metadata(args.checkpoint).item_metadata.tree['params']['params']['decoder']['layers']['block']
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), metadata)
  target = {'params': {'params': {'decoder': {'layers': {'block': abstract}}}}}
  restored = cp.restore(args.checkpoint, item=target, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args(target))
  cp.close()
  block = unfreeze(nn.unbox(restored['params']['params']['decoder']['layers']['block']))
  assert all(x.shape[1] == cfg.num_decoder_layers - 1 for x in jax.tree.leaves(block))
  params = jax.tree.map(lambda x: jnp.take(x, layer, axis=1), block)
  jax.block_until_ready(params)
  del restored, block, target, abstract
  print('LAYER_PARAMETERS_READY', layer, sum(x.size for x in jax.tree.leaves(params)), flush=True)

  def load(name):
    with tf.io.gfile.GFile(args.tensor_root.rstrip('/') + '/' + name + '.npy', 'rb') as f:
      return jnp.asarray(np.load(f), jnp.bfloat16)

  states = {name: load('forward-state-' + name) for name in ('M', 'residual')}
  cotangents = {name: load('backward-state-' + name) for name in ('M', 'residual')}
  with tf.io.gfile.GFile(args.cohort, 'rb') as f:
    with np.load(f) as saved:
      data = {k: jnp.asarray(saved[k]) for k in saved.files}
  x, matrix = states['residual'][0], states['M'][0]
  gx, gm = cotangents['residual'][1], cotangents['M'][1]
  probes = {tag + '_' + side: jnp.zeros_like(x)
            for tag in ('mlp_norm', 'mlp', 'memory_mlp') for side in ('input', 'output')}
  active = None
  forward_stats = None

  def rms(value):
    value = value.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(value))
    return maximum * jnp.sqrt(jnp.mean((value / jnp.maximum(maximum, 1e-30)) ** 2))

  def intercept(next_fun, call_args, kwargs, context):
    tag = None
    if context.method_name == '__call__':
      if isinstance(context.module, normalizations.RMSNorm) and context.module.name == 'post_self_attention_layer_norm':
        tag = 'mlp_norm'
      elif isinstance(context.module, linears.MlpBlock) and context.module.name in ('mlp', 'memory_mlp'):
        tag = context.module.name
    if tag is None:
      return next_fun(*call_args, **kwargs)
    a = list(call_args)
    a[0] = a[0] + active[tag + '_input']
    forward_stats[tag + '_input_target_rms'] = rms(a[0][:, token])
    out = next_fun(*a, **kwargs) + active[tag + '_output']
    forward_stats[tag + '_output_target_rms'] = rms(out[:, token])
    return out

  def layer_call(params, x, matrix, probes, gx, gm, *, pallas, block_probes):
    nonlocal active, forward_stats
    cfg.get_keys().update(bam_pallas_core=pallas, bam_pallas_write=pallas)
    module = fusion.SubDecoderLayer(cfg, mesh, None, cfg.max_target_length, 0)
    active, forward_stats = probes, {}
    m = matrix if pallas else jnp.swapaxes(matrix, -1, -2)
    def apply():
      return module.apply({'params': params}, x,
          data['inputs_segmentation'], data['inputs_position'], data['inputs'], None,
          True, 'train', None, M_in=m, layer_index=jnp.int32(layer),
          rngs={'dropout': jax.random.PRNGKey(0), 'params': jax.random.PRNGKey(0)})
    try:
      if block_probes:
        with nn.intercept_methods(intercept):
          y, result_m = apply()
      else:
        y, result_m = apply()
    finally:
      active = None
    result_m = result_m if pallas else jnp.swapaxes(result_m, -1, -2)
    objective = jnp.sum(y.astype(jnp.float32) * gx.astype(jnp.float32))
    objective += jnp.sum(result_m.astype(jnp.float32) * gm.astype(jnp.float32))
    return objective, (y, result_m, forward_stats)

  def measure(params, x, matrix, probes, gx, gm, *, pallas, block_probes):
    (_, (y, m, stats)), (dx, dm, dp) = jax.value_and_grad(
        layer_call, argnums=(1, 2, 3), has_aux=True)(params, x, matrix, probes, gx, gm,
                                                   pallas=pallas, block_probes=block_probes)
    return y, m, dx, dm, dp, stats

  def host_rms(a):
    a = np.asarray(a, np.float64)
    return float(np.sqrt(np.mean(a * a)))

  def relative(actual, reference):
    return host_rms(np.asarray(actual, np.float64) - np.asarray(reference, np.float64)) / max(host_rms(reference), 1e-30)

  results = dict(model_fidelity=fidelity, layer=layer, target_token=token, arms=[])
  previous = None
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(measure, static_argnames=('pallas', 'block_probes'))
    for pallas, block_probes in ((True, False), (True, True), (False, True)):
      y, m, dx, dm, dp, stats = jax.device_get(compiled(params, x, matrix, probes, gx, gm,
                                                     pallas=pallas, block_probes=block_probes))
      result = dict(pallas=pallas, block_probes=block_probes,
          output_relative_error=dict(residual=relative(y, states['residual'][1]), M=relative(m, states['M'][1])),
          cotangent_relative_error=dict(residual=relative(dx, cotangents['residual'][0]), M=relative(dm, cotangents['M'][0])),
          gradient_rms=dict(residual_in=host_rms(dx), residual_out=host_rms(gx), M_in=host_rms(dm), M_out=host_rms(gm)),
          forward_target_rms=jax.tree.map(float, stats))
      if block_probes:
        result['block_vjp_gains'] = {name: host_rms(dp[name + '_input']) / max(host_rms(dp[name + '_output']), 1e-30)
                                    for name in ('mlp_norm', 'mlp', 'memory_mlp')}
        control = (np.asarray(dp['mlp_norm_output'], np.float64)
                   - np.asarray(dp['mlp_input'], np.float64) - np.asarray(dp['memory_mlp_input'], np.float64))
        result['normalized_h_gradient_rms'] = dict(total=host_rms(dp['mlp_norm_output']),
            ordinary_mlp=host_rms(dp['mlp_input']), memory_mlp=host_rms(dp['memory_mlp_input']),
            memory_write_controls=host_rms(control))
      if previous is not None:
        result['previous_arm_gradient_relative_error'] = dict(residual=relative(dx, previous[0]), M=relative(dm, previous[1]))
      previous = (dx, dm)
      results['arms'].append(result)
      print('SINGLE_LAYER_VJP', json.dumps(result), flush=True)
      (args.output / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
  main()

"""Check saved Q/K/V/context cotangents against standalone Splash and FP32.

The tensors must come from a forward/gradient fidelity-checked model probe.
This measures a local VJP at those tensors; it does not replay the whole model.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import os


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--tensor-root', required=True)
  parser.add_argument('--reference-script', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip() == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import tensorflow as tf
  from flax import linen as nn
  import max_utils
  import pyconfig
  from layers import attentions
  native_attention = runpy.run_path(str(args.reference_script))['native_attention']
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'captured-splash-check').mkdir(exist_ok=True)
  with tf.io.gfile.GFile(args.tensor_root.rstrip('/') + '/tensor-fidelity.json') as f:
    fidelity = json.load(f)
  assert fidelity['passed'], 'Refusing to interpret tensors from a failed model fidelity gate'
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class='BamXLPropV5FastKernelFullRemat',
      run_name='captured-splash-check', base_output_directory=str(args.output) + '/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  arrays = {}
  for direction in ('forward', 'backward'):
    for name in ('q', 'k', 'v', 'context'):
      key = direction + '-' + name
      with tf.io.gfile.GFile(args.tensor_root.rstrip('/') + '/' + key + '.npy', 'rb') as f:
        arrays[key] = jnp.asarray(np.load(f), jnp.bfloat16)

  def rms(x):
    x = x.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(x))
    return maximum * jnp.sqrt(jnp.mean((x / jnp.maximum(maximum, 1e-30)) ** 2))

  def check(q, k, v, dy, observed_q, observed_k, observed_v, observed_context, *, fused):
    # A mutable config closure alone is not a JIT cache key.
    cfg.get_keys()['sa_use_fused_bwd_kernel'] = fused
    segments = jnp.ones(q.shape[:2], jnp.int32)
    call = lambda q, k, v: attentions._bam_splash_attention(q, k, v, segments, cfg, mesh,
        window=q.shape[1], seq_minor=attentions._bam_splash_seq_minor(cfg, False))
    y, pullback = jax.vjp(call, q, k, v)
    gradients = pullback(dy)
    ref, ref_pullback = jax.vjp(lambda q, k, v: native_attention(q, k, v).astype(jnp.bfloat16), q, k, v)
    references = ref_pullback(dy)
    result = dict(forward_reference_error_rms=rms(y.astype(jnp.float32) - ref.astype(jnp.float32)),
                  forward_observed_error_rms=rms(y.astype(jnp.float32) - observed_context.astype(jnp.float32)),
                  forward_reference_rms=rms(ref), context_cotangent_rms=rms(dy))
    for name, actual, expected, observed in zip(('q', 'k', 'v'), gradients, references,
                                                (observed_q, observed_k, observed_v)):
      denom = jnp.maximum(rms(expected), 1e-30)
      result[name] = dict(standalone_rms=rms(actual), reference_rms=rms(expected), observed_rms=rms(observed),
          standalone_relative_error=rms(actual.astype(jnp.float32) - expected.astype(jnp.float32)) / denom,
          observed_relative_error=rms(observed.astype(jnp.float32) - expected.astype(jnp.float32)) / denom,
          standalone_observed_relative_error=rms(actual.astype(jnp.float32) - observed.astype(jnp.float32)) / denom)
    # Actual dV and upstream G have the same shape, so RMS ratios equal Frobenius ratios.
    result['observed_dv_bound_fraction'] = rms(observed_v) / (jnp.sqrt(jnp.float32(q.shape[1])) * jnp.maximum(rms(dy), 1e-30))
    return result

  results = dict(model_fidelity=fidelity, arms=[])
  inputs = [arrays[x] for x in ('forward-q', 'forward-k', 'forward-v', 'backward-context',
                                'backward-q', 'backward-k', 'backward-v', 'forward-context')]
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled_check = jax.jit(check, static_argnames=('fused',))
    for fused in (True, False):
      measured = jax.device_get(compiled_check(*inputs, fused=fused))
      measured = jax.tree.map(float, measured)
      measured['fused_backward'] = fused
      results['arms'].append(measured)
      print('CAPTURED_SPLASH_CHECK', json.dumps(measured), flush=True)
  (args.output / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
  main()

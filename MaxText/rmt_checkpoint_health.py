"""Read-only, paired-cohort RMT checkpoint forward/gradient health diagnosis.

Restores params only, writes no checkpoint and applies no optimizer updates.
Record per-sequence loss/calibration and scalar activation/cotangent taps.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from absl import app
from flax.traverse_util import flatten_dict
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator


def parameter_stats(params, grads=None, scan_axis=1):
  p = flatten_dict(params)
  g = flatten_dict(grads) if grads is not None else None
  result = {}
  for path, x in p.items():
    name = '/'.join(path)
    axes = tuple(i for i in range(x.ndim) if not ('layers' in path and i == scan_axis))
    xf = x.astype(jnp.float32)
    row = {'parameter_rms': jnp.sqrt(jnp.mean(xf**2, axis=axes)),
           'parameter_l2': jnp.sqrt(jnp.sum(xf**2, axis=axes))}
    if g is not None:
      gf = g[path].astype(jnp.float32)
      row.update(gradient_rms=jnp.sqrt(jnp.mean(gf**2, axis=axes)),
                 gradient_l2=jnp.sqrt(jnp.sum(gf**2, axis=axes)),
                 parameter_gradient_dot=jnp.sum(xf * gf, axis=axes))
    result[name] = row
  return result


def serializable(tree):
  return jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), jax.device_get(tree))


def main(argv):
  cfg = pyconfig.initialize(argv)
  if not cfg.only_eval or cfg.enable_checkpointing or not cfg.load_parameters_path or cfg.load_full_state_path:
    raise ValueError('Require read-only parameter restore, no optimizer/checkpoint writes')
  if not cfg.base_output_directory.startswith('/tmp/'):
    raise ValueError('Output must be local /tmp')
  if cfg.get_keys().get('rmt_remat_policy', 'full') != 'full' or cfg.get_keys().get('rmt_block_scan', False):
    raise ValueError('Probe requires the trained plain layer-scan remat path')
  cfg.get_keys().update(rmt_crossscale_health_probe=False, rmt_norm_probe=False)
  rng, writer, manager, mesh, model, _, _ = train.setup_mesh_and_model(cfg)
  state, _ = max_utils.setup_decode_state(model, cfg, rng, mesh, manager)
  params = state.params
  output = Path(os.environ.get('RMT_HEALTH_OUTPUT', '/tmp/rmt-checkpoint-health'))
  output.mkdir(parents=True, exist_ok=True)
  count = int(os.environ.get('RMT_HEALTH_SEQUENCES', '32'))
  grad_count = int(os.environ.get('RMT_HEALTH_GRAD_SEQUENCES', '4'))
  source, _ = create_data_iterator(cfg, mesh)
  batches = [next(source) for _ in range(count)]
  order = np.random.default_rng(261001).permutation(count).tolist()
  batches = [batches[i] for i in order]
  hashes = [{k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in b.items()} for b in batches]
  meta = {'exp': cfg.exp_class, 'checkpoint': cfg.load_parameters_path,
          'layers': cfg.num_decoder_layers, 'heads': cfg.num_query_heads,
          'head_dim': cfg.head_dim, 'reskey_dim': cfg.rmt_reskey_dim,
          'sequence_length': cfg.max_target_length, 'global_batch': cfg.global_batch_size_to_load,
          'cohort_seed': 261001, 'order': order, 'cohort_hashes': hashes,
          'jax': jax.__version__, 'source_commit': os.environ.get('RMT_HEALTH_COMMIT')}
  (output/'metadata.json').write_text(json.dumps(meta, indent=2))
  print('PARAMS_AND_COHORT_READY '+json.dumps(meta), flush=True)
  stats = jax.jit(parameter_stats, static_argnames=('scan_axis',))(params, scan_axis=cfg.param_scan_axis)
  (output/'parameters.json').write_text(json.dumps(serializable(stats), indent=2))
  del stats

  def objective(p, batch):
    return train.loss_fn(model, cfg, dict(batch), rng, p, is_train=False)[0]
  # Paired gate: same restored params and first cohort sequence before/after taps.
  baseline = jax.jit(objective)
  with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
    ref = float(baseline(params, batches[0]))
  del baseline
  jax.clear_caches()
  cfg.get_keys().update(rmt_crossscale_health_probe=True, rmt_norm_probe=True)
  forward = jax.jit(objective)
  started = time.monotonic()
  for i, batch in enumerate(batches):
    os.environ['RMT_HEALTH_FILE'] = str(output/f'forward-{i:03d}-health.jsonl')
    os.environ['RMT_NORM_TAP_FILE'] = str(output/f'forward-{i:03d}-taps.jsonl')
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      loss = float(forward(params, batch))
    jax.effects_barrier()
    if i == 0 and abs(loss-ref) > 1e-5:
      raise ValueError(f'Instrumentation altered CE: baseline={ref}, instrumented={loss}')
    row = {'sequence': i, 'loss': loss, 'elapsed': time.monotonic()-started}
    with (output/'losses.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print('HEALTH_FORWARD '+json.dumps(row), flush=True)
  del forward
  jax.clear_caches()

  def gradient(p, batch):
    loss, grads = jax.value_and_grad(objective)(p, batch)
    stats = parameter_stats(p, grads, cfg.param_scan_axis)
    norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32)**2) for x in jax.tree_util.tree_leaves(grads)))
    return loss, norm, stats
  backward = jax.jit(gradient)
  for i, batch in enumerate(batches[:grad_count]):
    os.environ['RMT_HEALTH_FILE'] = str(output/f'gradient-{i:03d}-health.jsonl')
    os.environ['RMT_NORM_TAP_FILE'] = str(output/f'gradient-{i:03d}-taps.jsonl')
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      loss, norm, stats = backward(params, batch)
    row = serializable({'loss': loss, 'gradient_norm': norm, 'parameters': stats})
    jax.effects_barrier()
    (output/f'gradient-{i:03d}.json').write_text(json.dumps(row, indent=2))
    print('HEALTH_GRADIENT '+json.dumps({'sequence': i, 'loss':row['loss'], 'gradient_norm':row['gradient_norm']}), flush=True)
  if writer: writer.close()
  print('HEALTH_COMPLETE', flush=True)


if __name__ == '__main__':
  app.run(main)

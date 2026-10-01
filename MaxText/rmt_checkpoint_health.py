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
from flax import core
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import pyconfig
import train
from input_pipeline._pile_data_processing import PileDatasets, extract_pythia_datapath


def parameter_stats(params, grads=None, scan_axis=1):
  p = flatten_dict(params)
  g = flatten_dict(grads) if grads is not None else None
  result = {}
  for path, x in p.items():
    name = '/'.join(path)
    axes = tuple(i for i in range(x.ndim) if not ('layers' in path and i == scan_axis))
    xf = x.astype(jnp.float32)
    row = {'parameter_rms': jnp.sqrt(jnp.mean(xf**2, axis=axes)),
           'parameter_l2': jnp.sqrt(jnp.sum(xf**2, axis=axes)),
           'parameter_mean': jnp.mean(xf, axis=axes)}
    if g is not None:
      gf = g[path].astype(jnp.float32)
      row.update(gradient_rms=jnp.sqrt(jnp.mean(gf**2, axis=axes)),
                 gradient_l2=jnp.sqrt(jnp.sum(gf**2, axis=axes)),
                 parameter_gradient_dot=jnp.sum(xf * gf, axis=axes),
                 gradient_mean=jnp.mean(gf, axis=axes))
    if 'layers' in path and 'mlp' in path and path[-1] == 'kernel' and x.ndim == 3:
      xm = jnp.moveaxis(xf,scan_axis,0)
      unit_axis = 2 if path[-2] in ('wi_0','wi_1') else 1
      if path[-2] in ('wi_0','wi_1','wo'):
        ids = [i for i in (5363,3073,4107,5817) if i<xm.shape[unit_axis]]
        row['inspected_unit_ids'] = jnp.asarray(ids,dtype=jnp.int32)
        if unit_axis==2:
          row['inspected_unit_weight_l2'] = jnp.sqrt(jnp.sum(xm[:,:,jnp.asarray(ids,dtype=jnp.int32)]**2,axis=1))
        else:
          row['inspected_unit_weight_l2'] = jnp.sqrt(jnp.sum(xm[:,jnp.asarray(ids,dtype=jnp.int32),:]**2,axis=2))
        if g is not None:
          gm = jnp.moveaxis(g[path].astype(jnp.float32),scan_axis,0)
          selected = gm[:,:,jnp.asarray(ids,dtype=jnp.int32)] if unit_axis==2 else gm[:,jnp.asarray(ids,dtype=jnp.int32),:]
          row['inspected_unit_gradient_l2'] = jnp.sqrt(jnp.sum(selected**2,axis=1 if unit_axis==2 else 2))
          row['inspected_unit_gradient_energy_fraction'] = jnp.sum(selected**2,axis=(1,2))/jnp.maximum(jnp.sum(gm**2,axis=(1,2)),1e-30)
    if path[-1] == 'logits_dense':
      row['vocab_common_weight_rms'] = jnp.sqrt(jnp.mean(jnp.mean(xf,-1)**2))
      if g is not None:
        row['vocab_common_gradient_rms'] = jnp.sqrt(jnp.mean(jnp.mean(gf,-1)**2))
    result[name] = row
  return result


def serializable(tree):
  return jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), jax.device_get(tree))


def main(argv):
  cfg = pyconfig.initialize(argv)
  # Generic config validation couples restore to checkpoint-manager creation.
  # This standalone decoder restore owns neither a manager nor a save path.
  checkpoint = os.environ.get('RMT_HEALTH_CHECKPOINT')
  if checkpoint:
    if cfg.load_parameters_path or cfg.load_full_state_path:
      raise ValueError('Pass the diagnostic checkpoint only through RMT_HEALTH_CHECKPOINT')
    cfg.get_keys()['load_parameters_path'] = checkpoint
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
  # Use future training shards outside both restored checkpoints' seen prefix.
  # Ordinary first training batches can be memorized; legacy validation pads to
  # T4096 and would confound the actual-context comparison.
  paths, _ = extract_pythia_datapath(cfg.dataset_path, cfg.eval_split)
  source_paths = paths[-4:]
  source = PileDatasets(mesh=mesh, name='rmt.health', path=source_paths,
      batch_size=int(cfg.per_device_batch_size * jax.local_device_count()),
      seq_len=cfg.max_target_length, repeat=1, seed=261001,
      task_features=cfg.task_features, shuffle_buffer_size=1024,
      only_eval=True, zero_loss=cfg.zero_loss, iter_file_nums=2,
      mix_attn=cfg.mix_attn, pad_id=cfg.pad_id)
  batches = [next(source) for _ in range(count)]
  order = np.random.default_rng(261001).permutation(count).tolist()
  batches = [batches[i] for i in order]
  hashes = [{k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in b.items()} for b in batches]
  meta = {'exp': cfg.exp_class, 'checkpoint': cfg.load_parameters_path,
          'layers': cfg.num_decoder_layers, 'heads': cfg.num_query_heads,
          'head_dim': cfg.head_dim, 'reskey_dim': cfg.rmt_reskey_dim,
          'sequence_length': cfg.max_target_length, 'global_batch': cfg.global_batch_size_to_load,
          'cohort_seed': 261001, 'source_paths': source_paths, 'dataset_path': cfg.dataset_path,
          'cohort_role': 'unseen TruePile tail-four shards', 'order': order, 'cohort_hashes': hashes,
          'jax': jax.__version__, 'source_commit': os.environ.get('RMT_HEALTH_COMMIT')}
  (output/'metadata.json').write_text(json.dumps(meta, indent=2))
  for i,batch in enumerate(batches):
    (output/f'cohort-{i:03d}.json').write_text(json.dumps({
        k:np.asarray(batch[k]).reshape(-1).tolist() for k in ('inputs','targets','targets_segmentation')}))
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
  cfg.get_keys().update(rmt_crossscale_health_probe=True, rmt_norm_probe=True,
                       rmt_crossscale_numeric_probe=bool(int(os.environ.get('RMT_HEALTH_NUMERIC','0'))))
  forward = jax.jit(objective)
  started = time.monotonic()
  for i, batch in enumerate(batches):
    os.environ['RMT_HEALTH_FILE'] = str(output/f'forward-{i:03d}-health.jsonl')
    os.environ['RMT_NORM_TAP_FILE'] = str(output/f'forward-{i:03d}-taps.jsonl')
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      loss = float(forward(params, batch))
    jax.effects_barrier()
    if i == 0:
      delta = loss-ref
      (output/'instrumentation-gate.json').write_text(json.dumps(
          {'baseline_ce': ref, 'instrumented_ce': loss, 'delta': delta,
           'cpu_forward_gradient_identity': True, 'bf16_tpu_tolerance': .002}))
      if abs(delta) > .002:
        raise ValueError(f'Instrumentation CE drift too large: baseline={ref}, instrumented={loss}')
    row = {'sequence': i, 'loss': loss, 'elapsed': time.monotonic()-started}
    with (output/'losses.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print('HEALTH_FORWARD '+json.dumps(row), flush=True)
  # Same compiled computation and cohort for parameter interventions.
  if os.environ.get('RMT_HEALTH_ABLATIONS', '0') == '1':
    original_flat = flatten_dict(params)
    axis = cfg.param_scan_axis
    def changed(variant):
      flat = dict(original_flat)
      prefix = ('params', 'decoder')
      if variant.startswith('last_mlp_static_'):
        factor = float(variant[len('last_mlp_static_'):])
        path = prefix + ('layers', 'mlp_write_key')
        value = flat[path]
        idx = [slice(None)] * value.ndim
        idx[axis] = cfg.num_decoder_layers - 1
        flat[path] = value.at[tuple(idx)].multiply(factor)
      elif variant.startswith('last_mlp_neurons_'):
        ids = [int(v) for v in variant[len('last_mlp_neurons_'):].split('-')]
        path = prefix + ('layers', 'mlp', 'wo', 'kernel')
        value = jnp.moveaxis(flat[path],axis,0)
        if any(i>=value.shape[1] for i in ids):raise ValueError(ids)
        value = value.at[-1,jnp.asarray(ids),:].set(0)
        flat[path] = jnp.moveaxis(value,0,axis)
      elif variant == 'embedding_no_address_bias':
        path = prefix + ('dynamic_embedding_write', 'address_up_bias')
        flat[path] = jnp.zeros_like(flat[path])
      elif variant == 'unembedding_dynamic_off':
        path = prefix + ('dynamic_unembedding_read', 'key_kernel')
        flat[path] = jnp.zeros_like(flat[path])
      elif variant == 'last_mlp_dynamic_off':
        path = prefix + ('layers', 'dynamic_mlp_write', 'gate_bias')
        value = flat[path]
        idx = [slice(None)] * value.ndim
        idx[axis] = cfg.num_decoder_layers-1
        flat[path] = value.at[tuple(idx)].set(-100.)
      elif variant != 'baseline':
        raise ValueError(variant)
      tree = unflatten_dict(flat)
      return core.freeze(tree) if isinstance(params, core.FrozenDict) else tree
    variants = os.environ.get('RMT_HEALTH_VARIANTS',
        'baseline,last_mlp_static_0,last_mlp_static_0.5,last_mlp_static_0.9,last_mlp_static_1.1,last_mlp_dynamic_off,embedding_no_address_bias,unembedding_dynamic_off').split(',')
    for variant in variants:
      replacement = changed(variant)
      for i, batch in enumerate(batches):
        os.environ['RMT_HEALTH_FILE'] = str(output/f'ablation-{variant}-{i:03d}-health.jsonl')
        os.environ['RMT_NORM_TAP_FILE'] = str(output/f'ablation-{variant}-{i:03d}-taps.jsonl')
        with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
          loss = float(forward(replacement, batch))
        jax.effects_barrier()
        with (output/'ablations.jsonl').open('a') as f:
          f.write(json.dumps({'variant':variant,'sequence':i,'loss':loss})+'\n')
      print('HEALTH_ABLATION '+variant,flush=True)
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

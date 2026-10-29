"""Paired, dropout-free checkpoint loss for original versus head-tied O gamma.

Run on an explicitly owned diagnostic TPU. Restore model parameters only; keep
the runtime model and all parameter shapes unchanged. One compiled forward is
reused for every checkpoint and intervention. No optimizer or training writes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dynamic_o_gamma_analysis import coarsen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'MaxText'))


def paired_statistics(values):
  x = np.asarray(values, np.float64)
  if not np.isfinite(x).all() or len(x) < 2:
    raise ValueError('Need finite paired sequence losses')
  rng = np.random.default_rng(9876)
  means = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
  return dict(mean=float(x.mean()), std=float(x.std(ddof=1)),
              se=float(x.std(ddof=1) / np.sqrt(len(x))),
              bootstrap95=[float(v) for v in np.percentile(means, [2.5, 97.5])],
              n=len(x))


def load_cohort(path, count):
  z = np.load(path)
  required = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation')
  batch = {k: z[k][:count] for k in required}
  if batch['inputs'].shape != (count, 2048):
    raise ValueError(batch['inputs'].shape)
  if not np.all(batch['inputs_segmentation'] == 1):
    raise ValueError('This cohort adaptation expects full, unpacked sequences')
  # Pythia token0 is EOS, NOT padding. The historical cohort masked token0.
  # Its sequences are full-length, so every target is supervised here.
  batch['targets_segmentation'] = np.ones_like(batch['targets'], dtype=np.int32)
  proof = dict(
      source=str(path), source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      count=count, length=2048, mask='all targets valid, including EOS token0; Pythia pad_id=-1',
      sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in batch['inputs']],
      batch_sha256=hashlib.sha256(b''.join(batch[k].tobytes() for k in sorted(batch))).hexdigest())
  return batch, proof


def make_config(a):
  import pyconfig
  a.output.mkdir(parents=True, exist_ok=True)
  (a.output / 'gamma-head-diagnostic').mkdir(exist_ok=True)
  return pyconfig.initialize(
      [None, str(REPO / 'MaxText/configs/base.yml')], exp_class=a.exp,
      run_name='gamma-head-diagnostic', only_eval=True, enable_checkpointing=False,
      dataset_type='synthetic', base_output_directory=str(a.output) + '/',
      target_train_batch_size=None, per_device_batch_size=float(a.batch_size),
      eval_per_device_batch_size=float(a.batch_size),
      max_target_length=2048, max_prefill_predict_length=2048,
      jax_cache_dir=str(a.output / 'jax-cache'), profiler='', log_config=False,
      record_training_health_metrics=False, record_internal_nn_metrics=False)


def run(a):
  import jax
  import jax.numpy as jnp
  import orbax.checkpoint as ocp
  from flax.linen import partitioning
  from flax.traverse_util import flatten_dict, unflatten_dict
  import max_utils
  from layers.models import Transformer

  a.output.mkdir(parents=True, exist_ok=True)
  cohort, proof = load_cohort(a.cohort, a.sequences)
  if a.sequences % a.batch_size:
    raise ValueError('cohort size must be divisible by batch size')
  cfg = make_config(a)
  if not (cfg.pad_id == -1 and cfg.num_decoder_layers == 32 and cfg.num_query_heads == 32
          and cfg.bam_v == 64 and cfg.bam_local_o_dynamic_norm_scale_init == .2):
    raise ValueError('Unexpected checkpoint model geometry/recipe')
  if jax.device_count() != 1:
    raise ValueError('This runner is scoped to one diagnostic v6e-1')
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  checkpointer = ocp.PyTreeCheckpointer()
  modes = ('original', 'head_mean', 'head_signed_rms')
  metadata = dict(cohort=proof, runtime_model_commit=a.model_commit,
                  diagnostic_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  exp=a.exp, checkpoint_root=a.checkpoint_root, modes=modes,
                  batch_size=a.batch_size, device=str(jax.devices()[0]),
                  bam_pallas_core=cfg.bam_pallas_core, bam_pallas_write=cfg.bam_pallas_write,
                  interventions='Only params named local_o_dynamic_norm_scale, last axis; no config/shape changes')
  (a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

  def upload(file):
    if a.gcs_output:
      subprocess.run(['gsutil', '-q', 'cp', str(file), a.gcs_output.rstrip('/') + '/' + file.name], check=True)

  upload(a.output / 'metadata.json')

  @jax.jit
  def forward(params, batch):
    xent, _, _ = model.apply(
        params, batch['inputs'], batch['inputs_position'],
        decoder_segment_ids=batch['inputs_segmentation'],
        decoder_target_tokens=batch['targets'],
        decoder_target_mask=batch['targets_segmentation'], enable_dropout=False,
        rngs={'dropout': jax.random.key(9876), 'params': jax.random.key(9876)})
    mask = batch['targets_segmentation'] != 0
    return jnp.sum(xent.astype(jnp.float32) * mask, axis=-1) / jnp.maximum(mask.sum(axis=-1), 1)

  gamma_keys = None
  first_result = False
  for step in a.steps:
    result_path = a.output / f'loss-{step}.json'
    if result_path.exists():
      prior = json.loads(result_path.read_text())
      if prior['cohort_batch_sha256'] != proof['batch_sha256']:
        raise ValueError('Existing result uses a different cohort')
      continue
    path = a.checkpoint_root.rstrip('/') + f'/{step}/items'
    t0 = time.monotonic()
    md = checkpointer.metadata(path)
    tree = md.item_metadata.tree if hasattr(md, 'item_metadata') else md
    abstract = jax.tree.map(lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=sharding), tree['params'])
    restore_args = ocp.checkpoint_utils.construct_restore_args(abstract)
    restored = checkpointer.restore(path, args=ocp.args.PyTreeRestore(
        item={'params': abstract}, restore_args={'params': restore_args}, partial_restore=True))
    params = restored['params']
    del restored
    flat = flatten_dict(params)
    count = sum(int(v.size) for v in flat.values())
    if count != 2775256416:
      raise ValueError(('parameter count', count))
    keys = tuple(k for k in flat if k[-1] == 'local_o_dynamic_norm_scale')
    if sum(flat[k].size for k in keys) != 65536:
      raise ValueError('Wrong gamma leaves')
    if gamma_keys is not None and gamma_keys != keys:
      raise ValueError('Parameter tree changed across checkpoints')
    gamma_keys = keys
    variants = {}
    for mode in modes:
      changed = dict(flat)
      for key in keys:
        changed[key] = coarsen(flat[key], mode, xp=jnp)
      variants[mode] = unflatten_dict(changed)
    losses = {mode: [] for mode in modes}
    print(f'GAMMA_PROBE_RESTORED step={step} seconds={time.monotonic()-t0:.1f}', flush=True)
    for offset in range(0, a.sequences, a.batch_size):
      batch = {k: jax.device_put(v[offset:offset+a.batch_size], sharding) for k, v in cohort.items()}
      for mode in modes:
        with mesh, partitioning.axis_rules(cfg.logical_axis_rules):
          value = np.asarray(jax.device_get(forward(variants[mode], batch)), np.float64)
        if not np.isfinite(value).all():
          raise ValueError((step, mode, 'nonfinite CE'))
        losses[mode].extend(value.tolist())
      if not first_result:
        warm = a.output / 'first-result.json'
        warm.write_text(json.dumps(dict(step=step, sequences=a.batch_size,
                                        losses={m: losses[m] for m in modes})) + '\n')
        upload(warm)
        print('GAMMA_PROBE_FIRST_RESULT', warm, flush=True)
        first_result = True
    baseline = np.asarray(losses['original'])
    result = dict(step=step, checkpoint=path, cohort_batch_sha256=proof['batch_sha256'],
                  sequences=a.sequences, elapsed_seconds=time.monotonic()-t0, modes={})
    for mode in modes:
      values = np.asarray(losses[mode])
      result['modes'][mode] = dict(mean_loss=float(values.mean()), per_sequence=values.tolist(),
                                  delta=paired_statistics(values-baseline))
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    upload(result_path)
    print('GAMMA_PROBE_CHECKPOINT_DONE', json.dumps(
        dict(step=step, modes={m: result['modes'][m]['delta'] for m in modes})), flush=True)
    del params, flat, variants, changed, abstract
    gc.collect()
  print('GAMMA_PROBE_DONE', a.output, flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--exp', required=True)
  p.add_argument('--model-commit', required=True)
  p.add_argument('--checkpoint-root', required=True)
  p.add_argument('--steps', type=int, nargs='+', required=True)
  p.add_argument('--cohort', type=Path, required=True)
  p.add_argument('--sequences', type=int, default=128)
  p.add_argument('--batch-size', type=int, default=4)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--gcs-output')
  run(p.parse_args())


if __name__ == '__main__':
  main()

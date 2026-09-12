"""Per-token LocalO/fetched-O row spectra at the original BAlignedRow runtime.

No production edits. Capture by Linen interception; CPU statistics use bounded
parallel layer tasks with one BLAS thread. All valid positions are included.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText'))
from absl import app
from flax import linen as nn
from flax.linen import partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
import train
from layers import attentions

BASE = 'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow'
KEYS = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation', 'targets_segmentation')


class LocalORowRankProbe(getattr(exp, BASE)):
  only_eval = True
  per_device_batch_size = 1
  eval_per_device_batch_size = 1
  load_parameters_path = ('gs://newproject-1-llm_projects_us-east5/log/' + BASE
                          + '/checkpoints/13500/items')


exp.LocalORowRankProbe = LocalORowRankProbe


def forward(model, params, batch, rng, capture=True):
  owners = []
  saved = {}
  original_project = attentions._project_bam_read_keys
  original_read = attentions.bam_read

  def project(*args, **kwargs):
    result = original_project(*args, **kwargs)
    if owners:
      saved['keys'] = result
    return result

  def read(M, *args, **kwargs):
    result = original_read(M, *args, **kwargs)
    if owners:
      assert M.ndim == 4 and isinstance(result, tuple)
      raw, _, key, _ = saved.pop('keys')
      assert raw.shape[-2:] == (16, 32) and M.shape[-2:] == (32, 8)
      norm = attentions._transform_bam_read_key(
          raw, 'rms', kwargs['key_scale'], rms_epsilon=kwargs['rms_epsilon'],
          rms_statistics_dtype=kwargs['rms_statistics_dtype'])
      for name, value in dict(raw=raw, norm=norm, key=key, matrix=M, out=result[1]).items():
        owners[-1].sow('intermediates', 'orank_' + name, value.astype(jnp.float32))
    return result

  def intercept(next_fun, args, kwargs, ctx):
    if ctx.method_name != '_read_fetched_m':
      return next_fun(*args, **kwargs)
    owners.append(ctx.module)
    try:
      return next_fun(*args, **kwargs)
    finally:
      owners.pop()

  if capture:
    attentions._project_bam_read_keys = project
    attentions.bam_read = read
  try:
    with nn.intercept_methods(intercept):
      output, collections = model.apply(
          params, batch['inputs'], batch['inputs_position'],
          decoder_segment_ids=batch['inputs_segmentation'],
          decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
          enable_dropout=False, rngs={'params': rng, 'dropout': rng}, mutable=['intermediates'])
  finally:
    attentions._project_bam_read_keys = original_project
    attentions.bam_read = original_read
  mask = batch['targets_segmentation'] != 0
  loss = jnp.sum(output[0] * mask, -1) / jnp.maximum(mask.sum(-1), 1)
  raw = {}
  if capture:
    for path, value in flatten_dict(collections['intermediates']).items():
      if not path[-1].startswith('orank_'):
        continue
      offset = next(int(m[1]) for p in path if (m := re.fullmatch(r'(?:local|fetch)_(\d+)', p)))
      while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
      assert value.shape[0] == 8, (path, value.shape)
      for block in range(8):
        raw[f'L{3*block+offset:02d}_{path[-1][6:]}'] = value[block]
    assert len(raw) == 24 * 5, len(raw)
  return loss, raw


def spectrum(x):
  energy = np.linalg.svd(x, compute_uv=False)**2
  valid = energy.sum(-1) > 1e-24
  retained = np.cumsum(energy, -1) / np.maximum(energy.sum(-1, keepdims=True), 1e-30)
  retained = np.where(valid[:, None], retained, np.nan)
  return retained, valid


def layer_stats(layer, raw, mask):
  out = {}
  matrices = {k: np.asarray(raw[f'L{layer:02d}_{k}'])[mask].astype(np.float64)
              for k in ('raw', 'norm', 'key', 'matrix', 'out')}
  for stage in ('raw', 'norm', 'key', 'out'):
    x = matrices[stage]
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    unit = np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)
    for kind, value in [('amplitude', x), ('direction', unit)]:
      retained, valid = spectrum(value)
      prefix = f'{stage}_{kind}'
      out[prefix + '_retained_mean'] = np.nanmean(retained, axis=0)
      out[prefix + '_retained_quantiles'] = np.nanquantile(retained, [.05,.25,.5,.75,.95], axis=0)
      out[prefix + '_valid_fraction'] = np.mean(valid)
      for threshold in (.9, .95, .99):
        ranks = np.argmax(retained >= threshold, axis=-1) + 1
        out[prefix + f'_r{int(threshold*100)}'] = np.mean(ranks[valid]) if valid.any() else np.nan
    cos = unit @ unit.swapaxes(-1, -2)
    pair_valid = (norms[..., 0] > 1e-12)
    cos = np.where(pair_valid[..., :, None] & pair_valid[..., None, :], cos, np.nan)
    out[stage + '_cos'] = np.nanmean(cos, axis=0)
    out[stage + '_abs_cos'] = np.nanmean(abs(cos), axis=0)
    out[stage + '_cos_quantiles'] = np.nanquantile(cos, [.05,.5,.95], axis=0)
  # Token-dependent SVD is an optimistic bound, not a realizable learned A(x)/H(x).
  key, matrix = matrices['key'], matrices['matrix']
  u, s, vh = np.linalg.svd(key, full_matrices=False)
  reference = key @ matrix
  denominator = np.sum(reference**2, axis=(-2,-1))
  approx = np.zeros_like(reference)
  errors = []
  for r in range(16):
    approx += (u[..., :, r:r+1] * s[..., None, r:r+1]) @ (vh[..., r:r+1, :] @ matrix)
    errors.append(np.sum((approx-reference)**2, axis=(-2,-1)))
  errors = np.stack(errors, -1)
  out['key_svd_read_relative_squared_error'] = errors.sum(0) / max(denominator.sum(), 1e-30)
  out['key_svd_read_error_quantiles'] = np.quantile(
      errors / np.maximum(denominator[:, None], 1e-30), [.05,.5,.95], axis=0)
  out['read_energy'] = denominator.mean()
  return {f'L{layer:02d}_{k}': v for k,v in out.items()}


def run(config):
  cohort_path = Path(os.environ.get('ORANK_COHORT', '/tmp/pile_eval_cohort.npz'))
  output = Path(os.environ.get('ORANK_OUTPUT', '/tmp/local-o-row-rank'))
  output.mkdir(parents=True, exist_ok=True)
  with np.load(cohort_path) as data:
    cohort = {k: np.asarray(data[k]) for k in (*KEYS, 'sequence_hashes')}
  hashes = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in cohort['inputs']]
  assert hashes == list(cohort['sequence_hashes']) and len(hashes) == 128
  workers = min(16, len(os.sched_getaffinity(0)))
  print(f'CPU_WORKERS={workers} CPU_AVAILABLE={len(os.sched_getaffinity(0))}', flush=True)
  rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  cursor = SimpleNamespace(meta_dict={'checkpoint_step': None})
  state, _, _, _ = max_utils.setup_training_state(model, cursor, tx, config, rng, mesh, manager)
  capture = jax.jit(lambda p,b: forward(model,p,b,rng))
  ordinary = jax.jit(lambda p,b: forward(model,p,b,rng,False)[0])
  metadata = dict(base_class=BASE, checkpoint=config.load_parameters_path, checkpoint_step=13500,
      training_commit='77401da6f83a5aa6ddd61994e028c3c694221518',
      diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(), sequence_hashes=hashes,
      rank_scan=list(range(1,17)), layer_types=['F' if l%3==2 else 'L' for l in range(24)],
      cpu_workers=workers, scope='per-token geometry and read-error bounds; not training benefit')
  (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
  pending_jobs = []
  def finish(job):
    index, losses, tasks, started = job
    result = {k:v for task in tasks for k,v in task.result().items()}
    pending = output/f'.pending_{index:03d}.npz'
    np.savez_compressed(pending, loss=losses, **result)
    pending.replace(output/f'sample_{index:03d}.npz')
    print(f'SAMPLE_DONE {index} pipeline_seconds={time.perf_counter()-started:.2f}', flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    for index in range(128):
      path = output/f'sample_{index:03d}.npz'
      if path.exists():
        continue
      # Bound resident activation batches, while next TPU inference overlaps CPU SVD.
      if len(pending_jobs) >= 2:
        finish(pending_jobs.pop(0))
      started = time.perf_counter()
      batch = {k: jnp.asarray(cohort[k][index:index+1]) for k in KEYS}
      with mesh, partitioning.axis_rules(config.logical_axis_rules):
        losses, raw = jax.device_get(capture(state.params, batch))
        if index == 0:
          base_loss = np.asarray(ordinary(state.params, batch))
          np.testing.assert_array_equal(losses, base_loss)
          print('CAPTURE_FORWARD_EXACT', flush=True)
      assert np.isfinite(losses).all() and all(np.isfinite(v).all() for v in raw.values())
      print(f'FIRST_STEP sample={index} loss={losses[0]:.8f}', flush=True)
      mask = cohort['targets_segmentation'][index:index+1] != 0
      if index == 0:
        t0 = time.perf_counter()
        serial = [layer_stats(l, raw, mask) for l in range(1,5)]
        serial_seconds = time.perf_counter()-t0
        t0 = time.perf_counter()
        futures = [pool.submit(layer_stats,l,raw,mask) for l in range(1,5)]
        parallel = [f.result() for f in futures]
        parallel_seconds = time.perf_counter()-t0
        for a,b in zip(serial,parallel):
          for key in a:
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)
        print(f'PARALLEL_NUMERICS_OK serial_s={serial_seconds:.2f} parallel_s={parallel_seconds:.2f}', flush=True)
      tasks = [pool.submit(layer_stats,l,raw,mask) for l in range(24)]
      pending_jobs.append((index, losses, tasks, started))
    for job in pending_jobs:
      finish(job)
  if writer:
    writer.flush()
  print('DONE', flush=True)


if __name__ == '__main__':
  app.run(lambda argv: run(pyconfig.initialize(argv)))
